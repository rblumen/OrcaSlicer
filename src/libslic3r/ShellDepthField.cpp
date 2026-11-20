#include "ShellDepthField.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <admesh/stl.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "BoundingBox.hpp"
#include "ClipperUtils.hpp"
#include "ExPolygon.hpp"
#include "Flow.hpp"
#include "Layer.hpp"
#include "MarchingSquares.hpp"
#include "Model.hpp"
#include "Point.hpp"
#include "Polygon.hpp"
#include "Print.hpp"
#include "PrintConfig.hpp"
#include "SurfaceZoneDistance.hpp"
#include "libslic3r.h"

namespace Slic3r {

double shell_depth(int layers, double thickness, double layer_height)
{
    return layers > 0 ? std::max(layers * layer_height, thickness) : 0.;
}

bool ShellRampProfile::active() const { return taper_ratio > 0. && std::abs(depth - base_depth) > EPSILON; }

double ShellRampProfile::ramp_length() const { return std::abs(depth - base_depth) * taper_ratio; }

double ShellRampProfile::at(double distance) const
{
    if (! this->active())
        return base_depth;
    return depth - (depth - base_depth) * std::clamp(distance / this->ramp_length(), 0., 1.);
}

namespace {

// The combination over the zones on one side, see the top of ShellDepthField.hpp, fed one zone at a time.
struct SideCombination
{
    double thicker { - std::numeric_limits<double>::infinity() };
    double thinner { std::numeric_limits<double>::infinity() };
    double reach { - std::numeric_limits<double>::infinity() };
    double nearest_base { 0. };

    void add(const ShellRampProfile &profile, double distance)
    {
        if (! profile.active())
            return;
        const double length = profile.ramp_length();
        if (length - distance > reach) {
            reach        = length - distance;
            nearest_base = profile.base_depth;
        }
        if (! (distance < length))
            return;
        const double depth = profile.at(distance);
        if (depth > profile.base_depth)
            thicker = std::max(thicker, depth);
        else if (depth < profile.base_depth)
            thinner = std::min(thinner, depth);
    }

    std::optional<double> decided() const
    {
        if (thicker > - std::numeric_limits<double>::infinity())
            return thicker;
        if (thinner < std::numeric_limits<double>::infinity())
            return thinner;
        return std::nullopt;
    }
};

struct ZoneField
{
    ShellRampProfile    top;
    ShellRampProfile    bottom;
    SurfaceZoneDistance distance;

    bool active() const { return distance.active(); }
};

} // namespace

struct ShellDepthField::Impl
{
    std::vector<ZoneField> zones;
};

ShellDepthField::ShellDepthField(std::unique_ptr<const Impl> impl) : m_impl(std::move(impl)) {}
ShellDepthField::~ShellDepthField() = default;

std::shared_ptr<const ShellDepthField> ShellDepthField::build(const std::vector<indexed_triangle_set> &parts, const std::vector<ShellRampZone> &zones,
                                                              const ShellDepthFieldParams &params)
{
    if (zones.empty() || ! (params.fine_edge > 0.))
        return nullptr;

    auto impl = std::make_unique<Impl>();
    impl->zones.resize(zones.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, zones.size()), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i < range.end(); ++ i) {
            const ShellRampZone &in   = zones[i];
            ZoneField           &zone = impl->zones[i];
            zone.top    = in.top;
            zone.bottom = in.bottom;
            if (in.part >= parts.size() || ! (in.top.active() || in.bottom.active()))
                continue;
            SurfaceZoneDistanceParams distance_params;
            distance_params.fine_edge         = params.fine_edge;
            distance_params.coarse_edge       = params.coarse_edge;
            distance_params.edge_growth       = std::clamp(params.edge_growth, 0., 1.);
            distance_params.surface_tolerance = params.surface_tolerance;
            // Inside the zone the depth is D1 wherever it is.
            distance_params.inside_limit      = 0.;
            distance_params.outside_limit     = std::max(in.top.active() ? in.top.ramp_length() : 0., in.bottom.active() ? in.bottom.ramp_length() : 0.) +
                std::max(0., params.outside_margin);
            zone.distance = SurfaceZoneDistance::build(parts[in.part], in.mesh, distance_params);
        }
    });
    if (std::none_of(impl->zones.begin(), impl->zones.end(), [](const ZoneField &zone) { return zone.active(); }))
        return nullptr;
    return std::shared_ptr<const ShellDepthField>(new ShellDepthField(std::move(impl)));
}

size_t ShellDepthField::zone_count() const { return m_impl->zones.size(); }
bool   ShellDepthField::zone_active(size_t zone) const { return m_impl->zones[zone].active(); }
double ShellDepthField::signed_distance(size_t zone, const Vec3d &p) const { return m_impl->zones[zone].distance.signed_distance(p); }

std::optional<double> ShellDepthField::depth(ShellSide side, const Vec3d &p) const
{
    SideCombination combination;
    for (const ZoneField &zone : m_impl->zones) {
        const ShellRampProfile &profile = side == ShellSide::Top ? zone.top : zone.bottom;
        if (zone.active() && profile.active())
            combination.add(profile, zone.distance.signed_distance(p));
    }
    return combination.decided();
}

ShellSample ShellDepthField::sample(const Vec3d &p) const
{
    SideCombination top, bottom;
    for (const ZoneField &zone : m_impl->zones) {
        if (! zone.active())
            continue;
        const double distance = zone.distance.signed_distance(p);
        top.add(zone.top, distance);
        bottom.add(zone.bottom, distance);
    }
    auto reach = [](const SideCombination &side) { return std::isfinite(side.reach) ? float(side.reach) : -1.f; };
    ShellSample out;
    out.top_reach    = reach(top);
    out.top_depth    = float(top.decided().value_or(top.nearest_base));
    out.bottom_reach = reach(bottom);
    out.bottom_depth = float(bottom.decided().value_or(bottom.nearest_base));
    return out;
}

bool ShellDepthField::may_affect(const BoundingBoxf3 &box) const
{
    if (! box.defined)
        return false;
    const Eigen::AlignedBox3d query(box.min, box.max);
    return std::any_of(m_impl->zones.begin(), m_impl->zones.end(), [&query](const ZoneField &zone) { return zone.active() && zone.distance.reach().intersects(query); });
}

BoundingBoxf3 ShellDepthField::reach() const
{
    BoundingBoxf3 out;
    for (const ZoneField &zone : m_impl->zones)
        if (zone.active()) {
            out.merge(zone.distance.reach().min());
            out.merge(zone.distance.reach().max());
        }
    return out;
}

double ShellDepthField::max_depth(ShellSide side) const
{
    double out = 0.;
    for (const ZoneField &zone : m_impl->zones) {
        const ShellRampProfile &profile = side == ShellSide::Top ? zone.top : zone.bottom;
        if (zone.active() && profile.active())
            out = std::max({ out, profile.base_depth, profile.depth });
    }
    return out;
}

size_t ShellDepthField::triangle_count() const
{
    size_t count = 0;
    for (const ZoneField &zone : m_impl->zones)
        count += zone.distance.triangle_count();
    return count;
}

const Polygons &ShellLevels::at(size_t k) const
{
    static const Polygons none;
    return k < level.size() ? level[k] : none;
}

} // namespace Slic3r

namespace marchsq {

namespace {

// The field sampled once over a layer, on a grid of gsizef, and read by marching squares on a raster of rsizef: a
// raster value is interpolated between the grid samples around it. One of the four values of ShellSample is read.
struct ShellRaster
{
    static constexpr float gsizef = 0.5f;
    static constexpr float rsizef = 0.05f;
    const coord_t  rsize  = Slic3r::scaled(rsizef);
    const long             gsize  = std::lround(gsizef / rsizef);

    Slic3r::Point                    size;
    Slic3r::Point                    offs;
    // Grid samples, row by row.
    size_t                           grid_rows { 0 };
    size_t                           grid_cols { 0 };
    std::vector<Slic3r::ShellSample> samples;

    ShellRaster(const Slic3r::ShellDepthField &field, const Slic3r::BoundingBox &bbox, double z) : size(bbox.size()), offs(bbox.min)
    {
        // Every grid sample a raster value can be interpolated from, and one more on each axis.
        grid_rows = size_t(rows() / gsize + 2);
        grid_cols = size_t(cols() / gsize + 2);
        samples.resize(grid_rows * grid_cols);
        for (size_t r = 0; r < grid_rows; ++ r)
            for (size_t c = 0; c < grid_cols; ++ c) {
                const Slic3r::Point pt = to_Point(Coord(long(r) * gsize, long(c) * gsize));
                samples[r * grid_cols + c] = field.sample(Slic3r::Vec3d(Slic3r::unscaled(pt.x()), Slic3r::unscaled(pt.y()), z));
            }
    }

    long rows() const { return to_coordr(size.y()); }
    long cols() const { return to_coordr(size.x()); }

    // Bilinear between the grid samples around raster point (row, col).
    float get(float Slic3r::ShellSample::*value, long row, long col) const
    {
        const long   r0 = std::min<long>(row / gsize, long(grid_rows) - 2);
        const long   c0 = std::min<long>(col / gsize, long(grid_cols) - 2);
        const float  tr = std::clamp(float(row - r0 * gsize) / float(gsize), 0.f, 1.f);
        const float  tc = std::clamp(float(col - c0 * gsize) / float(gsize), 0.f, 1.f);
        auto at = [this, value](long r, long c) { return samples[size_t(r) * grid_cols + size_t(c)].*value; };
        const float top    = at(r0, c0) + tc * (at(r0, c0 + 1) - at(r0, c0));
        const float bottom = at(r0 + 1, c0) + tc * (at(r0 + 1, c0 + 1) - at(r0 + 1, c0));
        return top + tr * (bottom - top);
    }

    inline coord_t to_coord(long x) const { return x * rsize; }
    inline long            to_coordr(coord_t x) const { return long(x / rsize); }
    inline Slic3r::Point   to_Point(const Coord &p) const { return Slic3r::Point(to_coord(p.c) + offs.x(), to_coord(p.r) + offs.y()); }
};

// One of the values of a ShellRaster.
struct ShellRasterValue
{
    const ShellRaster *raster;
    float Slic3r::ShellSample::*value;
};

} // namespace

template<> struct _RasterTraits<ShellRasterValue>
{
    using ValueType = float;
    static float  get(const ShellRasterValue &sv, size_t row, size_t col) { return sv.raster->get(sv.value, long(row), long(col)); }
    static size_t rows(const ShellRasterValue &sv) { return size_t(sv.raster->rows()); }
    static size_t cols(const ShellRasterValue &sv) { return size_t(sv.raster->cols()); }
};

} // namespace marchsq

namespace Slic3r {

namespace {

// Where the value exceeds iso.
Polygons contour(const marchsq::ShellRaster &raster, float ShellSample::*value, float iso)
{
    const marchsq::ShellRasterValue view { &raster, value };
    Polygons out;
    for (const marchsq::Ring &ring : marchsq::execute(view, iso, { raster.gsize, raster.gsize })) {
        Polygon &polygon = out.emplace_back();
        polygon.points.reserve(ring.size());
        for (const marchsq::Coord &crd : ring)
            polygon.points.emplace_back(raster.to_Point(crd));
    }
    // The rings of a field never cross, and a hole's ring lies inside the ring around it.
    return union_(out, pftEvenOdd);
}

ShellLevels threshold(const marchsq::ShellRaster &raster, float ShellSample::*reach, float ShellSample::*depth, double max_depth,
                      const std::vector<double> &offsets)
{
    ShellLevels out;
    out.reach = contour(raster, reach, 0.f);
    if (out.reach.empty())
        return out;
    // The depth runs on past the end of the reach, so that clipping the levels to it leaves no gap between the levels
    // and the stock shell outside.
    for (const double offset : offsets) {
        // As the stock projection: the layer is in the shell if its distance is below the depth by more than EPSILON.
        const double iso = offset + EPSILON;
        if (iso >= max_depth)
            break;
        Polygons level = intersection(contour(raster, depth, float(iso)), out.reach);
        if (level.empty())
            break;
        out.level.emplace_back(std::move(level));
    }
    return out;
}

} // namespace

LayerShellLevels make_layer_shell_levels(const ShellDepthField &field, const BoundingBox &extents, double z, const std::vector<double> &top_offsets,
                                         const std::vector<double> &bottom_offsets)
{
    LayerShellLevels out;
    const BoundingBoxf3 reach = field.reach();
    if (! extents.defined || ! reach.defined || z < reach.min.z() || z > reach.max.z())
        return out;
    // The samples cover the layer where the field reaches it, a cell of margin for the rings closed along the border,
    // and lie on a fixed grid, so that neighbouring layers are sampled at the same points.
    BoundingBox bbox(Point(std::max(extents.min.x(), coord_t(scale_(reach.min.x()))), std::max(extents.min.y(), coord_t(scale_(reach.min.y())))),
                     Point(std::min(extents.max.x(), coord_t(scale_(reach.max.x()))), std::min(extents.max.y(), coord_t(scale_(reach.max.y())))));
    if (bbox.min.x() > bbox.max.x() || bbox.min.y() > bbox.max.y())
        return out;
    const coord_t cell = scaled(marchsq::ShellRaster::gsizef);
    bbox.offset(cell);
    bbox.merge(align_to_grid(bbox.min, Point(cell, cell)));
    const marchsq::ShellRaster raster(field, bbox, z);

    out.top    = threshold(raster, &ShellSample::top_reach, &ShellSample::top_depth, field.max_depth(ShellSide::Top), top_offsets);
    out.bottom = threshold(raster, &ShellSample::bottom_reach, &ShellSample::bottom_depth, field.max_depth(ShellSide::Bottom), bottom_offsets);
    return out;
}

std::vector<LayerShellLevels> make_shell_levels(const ShellDepthField &field, const PrintObject &object)
{
    const auto                   layers     = object.layers();
    const double                 top_max    = field.max_depth(ShellSide::Top);
    const double                 bottom_max = field.max_depth(ShellSide::Bottom);
    std::vector<LayerShellLevels> out(layers.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, layers.size()), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i < range.end(); ++ i) {
            const Layer &layer = *layers[i];
            if (layer.lslices.empty())
                continue;
            const BoundingBox extents = get_extents(layer.lslices);
            if (! field.may_affect(BoundingBoxf3(Vec3d(unscaled(extents.min.x()), unscaled(extents.min.y()), layer.bottom_z()),
                                                 Vec3d(unscaled(extents.max.x()), unscaled(extents.max.y()), layer.print_z))))
                continue;
            // The Z distances the stock projection compares with the shell thickness, up to the deepest shell.
            std::vector<double> top_offsets, bottom_offsets;
            for (size_t k = 0; k <= i; ++ k) {
                top_offsets.emplace_back(layer.print_z - layers[i - k]->print_z);
                if (top_offsets.back() >= top_max)
                    break;
            }
            for (size_t k = 0; i + k < layers.size(); ++ k) {
                bottom_offsets.emplace_back(layers[i + k]->bottom_z() - layer.bottom_z());
                if (bottom_offsets.back() >= bottom_max)
                    break;
            }
            out[i] = make_layer_shell_levels(field, extents, layer.slice_z, top_offsets, bottom_offsets);
        }
    });
    return out;
}

std::shared_ptr<const ShellDepthField> build_shell_depth_field(const PrintObject &object, std::vector<const ModelVolume*> *inconsistent_modifiers)
{
    const PrintObjectRegions *regions = object.shared_regions();
    if (regions == nullptr)
        return nullptr;
    const double layer_height = object.config().layer_height.value;

    // One zone per modifier and part. A modifier nested in another one has a region over the part and one over the
    // outer modifier's region; the latter comes later and wins, so its base is the outer modifier's shell. The zone
    // spans the object's whole height, so the modifier gets one only if its profiles are the same in every height
    // range it appears in.
    auto same_profile = [](const ShellRampProfile &a, const ShellRampProfile &b) {
        return a.base_depth == b.base_depth && a.depth == b.depth && a.taper_ratio == b.taper_ratio;
    };
    struct Candidate
    {
        ShellRampZone zone;
        size_t        ranges { 0 };
        bool          consistent { true };
    };
    using Volumes = std::pair<const ModelVolume*, const ModelVolume*>;
    std::map<Volumes, Candidate> candidates;
    const PrintRegion           *first_part_region = nullptr;
    for (const PrintObjectRegions::LayerRangeRegions &layer_range : regions->layer_ranges) {
        std::map<Volumes, ShellRampZone> in_range;
        for (const PrintObjectRegions::VolumeRegion &volume_region : layer_range.volume_regions) {
            if (volume_region.parent < 0 || volume_region.region == nullptr || volume_region.region == layer_range.volume_regions[volume_region.parent].region)
                continue;
            const PrintRegion &modifier = *volume_region.region;
            const RampKey     &key      = modifier.ramp_key();
            if (key.top_base_layers < 0 && key.bottom_base_layers < 0)
                continue;
            int root = volume_region.parent;
            while (layer_range.volume_regions[root].parent >= 0)
                root = layer_range.volume_regions[root].parent;
            if (first_part_region == nullptr)
                first_part_region = layer_range.volume_regions[root].region;
            const PrintRegionConfig &config = modifier.config();
            ShellRampZone            zone;
            if (key.top_base_layers >= 0)
                zone.top = { shell_depth(key.top_base_layers, key.top_base_thickness, layer_height),
                             shell_depth(config.top_shell_layers.value, config.top_shell_thickness.value, layer_height), config.top_shell_ramp_taper_ratio.value };
            if (key.bottom_base_layers >= 0)
                zone.bottom = { shell_depth(key.bottom_base_layers, key.bottom_base_thickness, layer_height),
                                shell_depth(config.bottom_shell_layers.value, config.bottom_shell_thickness.value, layer_height),
                                config.bottom_shell_ramp_taper_ratio.value };
            in_range[{ volume_region.model_volume, layer_range.volume_regions[root].model_volume }] = std::move(zone);
        }
        for (auto &[volumes, zone] : in_range) {
            Candidate &candidate = candidates[volumes];
            if (candidate.ranges > 0 && ! (same_profile(zone.top, candidate.zone.top) && same_profile(zone.bottom, candidate.zone.bottom)))
                candidate.consistent = false;
            candidate.zone = std::move(zone);
            ++ candidate.ranges;
        }
    }

    // The meshes in the frame the layers were sliced in, see PrintObjectSlice.cpp.
    const Transform3d trafo = object.trafo_sliced();
    auto sliced_mesh = [&trafo](const ModelVolume &volume) {
        indexed_triangle_set its = volume.mesh().its;
        its_transform(its, trafo * volume.get_matrix(), true);
        return its;
    };
    std::vector<const ModelVolume*>   parts;
    std::vector<indexed_triangle_set> part_meshes;
    std::vector<ShellRampZone>        zones;
    for (auto &[volumes, candidate] : candidates) {
        if (! candidate.consistent) {
            if (inconsistent_modifiers != nullptr && std::find(inconsistent_modifiers->begin(), inconsistent_modifiers->end(), volumes.first) == inconsistent_modifiers->end())
                inconsistent_modifiers->emplace_back(volumes.first);
            continue;
        }
        if (! candidate.zone.top.active() && ! candidate.zone.bottom.active())
            continue;
        auto it_part = std::find(parts.begin(), parts.end(), volumes.second);
        candidate.zone.part = size_t(it_part - parts.begin());
        if (it_part == parts.end()) {
            parts.emplace_back(volumes.second);
            part_meshes.emplace_back(sliced_mesh(*volumes.second));
        }
        candidate.zone.mesh = sliced_mesh(*volumes.first);
        zones.emplace_back(std::move(candidate.zone));
    }
    if (zones.empty())
        return nullptr;

    const PrintObjectConfig &config = object.config();
    double max_layer_height = 0.;
    for (const Layer *layer : object.layers())
        max_layer_height = std::max(max_layer_height, layer->height);
    // The widest wall band of the object: the ensure vertical shell thickness lookups lie within it, that far off the
    // side faces.
    double wall_band = 0.;
    for (size_t i = 0; i < object.num_printing_regions(); ++ i) {
        const PrintRegion &region = object.printing_region(i);
        wall_band = std::max(wall_band, region.config().wall_loops.value * double(region.flow(object, frPerimeter, layer_height).spacing()) +
                                            double(region.flow(object, frExternalPerimeter, layer_height).width()));
    }
    ShellDepthFieldParams params;
    // A solid infill line is the precision a shell's boundary needs.
    const double spacing     = first_part_region->flow(object, frSolidInfill, config.layer_height.value).spacing();
    params.fine_edge         = 0.5 * spacing;
    params.coarse_edge       = spacing;
    params.surface_tolerance = max_layer_height + wall_band;
    return ShellDepthField::build(part_meshes, zones, params);
}

} // namespace Slic3r
