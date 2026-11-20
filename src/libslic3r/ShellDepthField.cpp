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

// As the stock projection, see PrintObject::discover_vertical_shells() and discover_horizontal_shells().
bool ShellRule::solid(size_t k, double dz) const
{
    auto in_shell = [k, dz](double n, double T) { return double(k) < n || dz < T - EPSILON; };
    return in_shell(layers, thickness) || (in_shell(floor_in_layers, floor_in_thickness) && in_shell(floor_base_layers, floor_base_thickness));
}

bool ShellRampProfile::active() const { return taper_ratio > 0. && std::abs(this->depth() - this->base_depth()) > EPSILON; }

double ShellRampProfile::ramp_length() const { return std::abs(this->depth() - this->base_depth()) * taper_ratio; }

ShellRule ShellRampProfile::at(double distance) const
{
    const ShellRule base { double(std::max(base_layers, 0)), base_layers > 0 ? base_thickness : 0. };
    if (! this->active())
        return base;
    const ShellRule in { double(std::max(layers, 0)), layers > 0 ? thickness : 0. };
    const double    t = std::clamp(distance / this->ramp_length(), 0., 1.);
    // Interpolating n and T separately can put fewer layers in the shell than both ends do. A layer that both ends' rules
    // put in the shell is in it across the whole ramp. Holding the shell of the end with the lesser depth instead would
    // be right only at the nominal layer height: at another one the ends can swap order and the thicker end would be held.
    return { in.layers + (base.layers - in.layers) * t, in.thickness + (base.thickness - in.thickness) * t,
             in.layers, in.thickness, base.layers, base.thickness };
}

namespace {

// The combination over the zones on one side at one point, see the top of ShellDepthField.hpp, fed one zone at a time
// with a value that grows with how far the zone's shell reaches: the largest value over the deciding zones that
// thicken, else the smallest over those that thin, else none. `nearest` is the value of the zone whose ramp ends
// nearest to deciding, and `reach` the largest L_s - d, which crosses 0 where the last ramp ends.
template<class Value> struct SideCombination
{
    std::optional<Value> thicker;
    std::optional<Value> thinner;
    std::optional<Value> nearest;
    double               reach { - std::numeric_limits<double>::infinity() };

    void add(const ShellRampProfile &profile, double distance, Value value)
    {
        const double length = profile.ramp_length();
        if (length - distance > reach) {
            reach   = length - distance;
            nearest = value;
        }
        if (! (distance < length))
            return;
        if (profile.thickens())
            thicker = thicker ? std::max(*thicker, value) : value;
        else
            thinner = thinner ? std::min(*thinner, value) : value;
    }

    std::optional<Value> decided() const { return thicker ? thicker : thinner; }
};

// How many of the leading layers the rule puts in the shell.
size_t solid_prefix(const ShellRule &rule, const std::vector<double> &offsets)
{
    size_t k = 0;
    while (k < offsets.size() && rule.solid(k, offsets[k]))
        ++ k;
    return k;
}

struct ZoneField
{
    ShellRampProfile    top;
    ShellRampProfile    bottom;
    SurfaceZoneDistance distance;

    bool                    active() const { return distance.active(); }
    const ShellRampProfile &profile(ShellSide side) const { return side == ShellSide::Top ? top : bottom; }
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
            // Inside the zone the rule is (n1, T1) wherever it is.
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

size_t                  ShellDepthField::zone_count() const { return m_impl->zones.size(); }
bool                    ShellDepthField::zone_active(size_t zone) const { return m_impl->zones[zone].active(); }
const ShellRampProfile &ShellDepthField::profile(size_t zone, ShellSide side) const { return m_impl->zones[zone].profile(side); }
double                  ShellDepthField::signed_distance(size_t zone, const Vec3d &p) const { return m_impl->zones[zone].distance.signed_distance(p); }

std::optional<size_t> ShellDepthField::solid_layers(ShellSide side, const Vec3d &p, const std::vector<double> &offsets) const
{
    SideCombination<size_t> combination;
    for (const ZoneField &zone : m_impl->zones) {
        const ShellRampProfile &profile = zone.profile(side);
        if (zone.active() && profile.active()) {
            const double distance = zone.distance.signed_distance(p);
            combination.add(profile, distance, solid_prefix(profile.at(distance), offsets));
        }
    }
    return combination.decided();
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

ShellRule ShellDepthField::deepest(ShellSide side) const
{
    ShellRule out;
    for (const ZoneField &zone : m_impl->zones) {
        const ShellRampProfile &profile = zone.profile(side);
        if (zone.active() && profile.active())
            for (const ShellRule &rule : { profile.at(0.), profile.at(profile.ramp_length()) }) {
                out.layers    = std::max(out.layers, rule.layers);
                out.thickness = std::max(out.thickness, rule.thickness);
            }
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

// The zone distances sampled once over a layer, on a grid of gsizef, and values computed from them read by marching
// squares on a raster of rsizef: a raster value is interpolated between the grid values around it.
struct ShellRaster
{
    static constexpr float gsizef = 0.5f;
    static constexpr float rsizef = 0.05f;
    const coord_t          rsize  = Slic3r::scaled(rsizef);
    const long             gsize  = std::lround(gsizef / rsizef);

    Slic3r::Point      size;
    Slic3r::Point      offs;
    size_t             grid_rows { 0 };
    size_t             grid_cols { 0 };
    size_t             zones { 0 };
    // The signed distance of every zone at every grid sample, sample by sample and the samples row by row.
    std::vector<float> distances;

    ShellRaster(const Slic3r::ShellDepthField &field, const Slic3r::BoundingBox &bbox, double z) : size(bbox.size()), offs(bbox.min)
    {
        // Every grid sample a raster value can be interpolated from, and one more on each axis.
        grid_rows = size_t(rows() / gsize + 2);
        grid_cols = size_t(cols() / gsize + 2);
        zones     = field.zone_count();
        distances.assign(grid_rows * grid_cols * zones, std::numeric_limits<float>::infinity());
        for (size_t r = 0; r < grid_rows; ++ r)
            for (size_t c = 0; c < grid_cols; ++ c) {
                const Slic3r::Point pt = to_Point(Coord(long(r) * gsize, long(c) * gsize));
                const Slic3r::Vec3d p(Slic3r::unscaled(pt.x()), Slic3r::unscaled(pt.y()), z);
                for (size_t zone = 0; zone < zones; ++ zone)
                    if (field.zone_active(zone))
                        distances[(r * grid_cols + c) * zones + zone] = float(field.signed_distance(zone, p));
            }
    }

    size_t samples() const { return grid_rows * grid_cols; }
    long   rows() const { return to_coordr(size.y()); }
    long   cols() const { return to_coordr(size.x()); }

    // Bilinear between the grid values around raster point (row, col).
    float get(const std::vector<float> &grid, long row, long col) const
    {
        const long  r0 = std::min<long>(row / gsize, long(grid_rows) - 2);
        const long  c0 = std::min<long>(col / gsize, long(grid_cols) - 2);
        const float tr = std::clamp(float(row - r0 * gsize) / float(gsize), 0.f, 1.f);
        const float tc = std::clamp(float(col - c0 * gsize) / float(gsize), 0.f, 1.f);
        auto at = [this, &grid](long r, long c) { return grid[size_t(r) * grid_cols + size_t(c)]; };
        const float top    = at(r0, c0) + tc * (at(r0, c0 + 1) - at(r0, c0));
        const float bottom = at(r0 + 1, c0) + tc * (at(r0 + 1, c0 + 1) - at(r0 + 1, c0));
        return top + tr * (bottom - top);
    }

    inline coord_t       to_coord(long x) const { return x * rsize; }
    inline long          to_coordr(coord_t x) const { return long(x / rsize); }
    inline Slic3r::Point to_Point(const Coord &p) const { return Slic3r::Point(to_coord(p.c) + offs.x(), to_coord(p.r) + offs.y()); }
};

// Values on the grid of a ShellRaster, one per grid sample.
struct ShellRasterValue
{
    const ShellRaster        *raster;
    const std::vector<float> *grid;
};

} // namespace

template<> struct _RasterTraits<ShellRasterValue>
{
    using ValueType = float;
    static float  get(const ShellRasterValue &sv, size_t row, size_t col) { return sv.raster->get(*sv.grid, long(row), long(col)); }
    static size_t rows(const ShellRasterValue &sv) { return size_t(sv.raster->rows()); }
    static size_t cols(const ShellRasterValue &sv) { return size_t(sv.raster->cols()); }
};

} // namespace marchsq

namespace Slic3r {

namespace {

// Where the grid values exceed 0.
Polygons contour(const marchsq::ShellRaster &raster, const std::vector<float> &grid)
{
    const marchsq::ShellRasterValue view { &raster, &grid };
    Polygons out;
    for (const marchsq::Ring &ring : marchsq::execute(view, 0.f, { raster.gsize, raster.gsize })) {
        Polygon &polygon = out.emplace_back();
        polygon.points.reserve(ring.size());
        for (const marchsq::Coord &crd : ring)
            polygon.points.emplace_back(raster.to_Point(crd));
    }
    // The rings of a field never cross, and a hole's ring lies inside the ring around it.
    return union_(out, pftEvenOdd);
}

// By how much, in layers, the rule puts the layer k layers from the surface, dz from it, in the shell: positive where
// it does. A layer count exactly k, as inside a zone of k layers, is short of it by a thousandth of a layer, so that
// the rounding of the raster does not put the layer in.
float shell_margin(const ShellRule &rule, size_t k, double dz, double layer_height)
{
    auto margin = [k, dz, layer_height](double layers, double thickness) {
        return std::max(layers - double(k) - 1e-3, layer_height > 0. ? (thickness - dz - EPSILON) / layer_height : -std::numeric_limits<double>::infinity());
    };
    return float(std::max(margin(rule.layers, rule.thickness),
                          std::min(margin(rule.floor_in_layers, rule.floor_in_thickness), margin(rule.floor_base_layers, rule.floor_base_thickness))));
}

ShellLevels threshold(const ShellDepthField &field, const marchsq::ShellRaster &raster, ShellSide side, const std::vector<double> &offsets)
{
    std::vector<const ShellRampProfile*> profiles(raster.zones, nullptr);
    for (size_t zone = 0; zone < raster.zones; ++ zone)
        if (field.zone_active(zone) && field.profile(zone, side).active())
            profiles[zone] = &field.profile(zone, side);
    ShellLevels out;
    if (std::none_of(profiles.begin(), profiles.end(), [](const ShellRampProfile *profile) { return profile != nullptr; }))
        return out;

    // Every grid sample through the combination, giving each zone the value `value(profile, distance)`.
    std::vector<float> grid(raster.samples());
    auto combine = [&raster, &profiles, &grid](auto &&value, auto &&result) {
        for (size_t sample = 0; sample < raster.samples(); ++ sample) {
            SideCombination<float> combination;
            for (size_t zone = 0; zone < raster.zones; ++ zone)
                if (profiles[zone] != nullptr) {
                    const double distance = raster.distances[sample * raster.zones + zone];
                    combination.add(*profiles[zone], distance, value(*profiles[zone], distance));
                }
            grid[sample] = result(combination);
        }
    };

    combine([](const ShellRampProfile &, double) { return 0.f; },
            [](const SideCombination<float> &combination) { return std::isfinite(combination.reach) ? float(combination.reach) : -1.f; });
    out.reach = contour(raster, grid);
    if (out.reach.empty())
        return out;
    for (size_t k = 0; k < offsets.size(); ++ k) {
        // Past where the last ramp ends, the nearest zone's (n0, T0) runs on, so that clipping the level to the reach
        // leaves no gap between it and the stock shell outside.
        combine([k, dz = offsets[k]](const ShellRampProfile &profile, double distance) { return shell_margin(profile.at(distance), k, dz, profile.layer_height); },
                [](const SideCombination<float> &combination) { return combination.decided().value_or(combination.nearest.value_or(-1.f)); });
        Polygons level = intersection(contour(raster, grid), out.reach);
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

    out.top    = threshold(field, raster, ShellSide::Top, top_offsets);
    out.bottom = threshold(field, raster, ShellSide::Bottom, bottom_offsets);
    return out;
}

std::vector<LayerShellLevels> make_shell_levels(const ShellDepthField &field, const PrintObject &object)
{
    const auto                    layers  = object.layers();
    const ShellRule               top     = field.deepest(ShellSide::Top);
    const ShellRule               bottom  = field.deepest(ShellSide::Bottom);
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
            // The Z distances the stock projection compares with the shell thickness, as far as any zone's shell reaches.
            std::vector<double> top_offsets, bottom_offsets;
            for (size_t k = 0; k <= i; ++ k) {
                const double dz = layer.print_z - layers[i - k]->print_z;
                if (! top.solid(k, dz))
                    break;
                top_offsets.emplace_back(dz);
            }
            for (size_t k = 0; i + k < layers.size(); ++ k) {
                const double dz = layers[i + k]->bottom_z() - layer.bottom_z();
                if (! bottom.solid(k, dz))
                    break;
                bottom_offsets.emplace_back(dz);
            }
            out[i] = make_layer_shell_levels(field, extents, layer.slice_z, top_offsets, bottom_offsets);
        }
    });
    return out;
}

std::shared_ptr<const ShellDepthField> build_shell_depth_field(const PrintObject &object, std::vector<ModifierOnPart> *inconsistent)
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
        return a.base_layers == b.base_layers && a.base_thickness == b.base_thickness && a.layers == b.layers && a.thickness == b.thickness &&
            a.taper_ratio == b.taper_ratio;
    };
    struct Candidate
    {
        ShellRampZone zone;
        size_t        ranges { 0 };
        bool          consistent { true };
    };
    std::map<ModifierOnPart, Candidate> candidates;
    const PrintRegion                  *first_part_region = nullptr;
    for (const PrintObjectRegions::LayerRangeRegions &layer_range : regions->layer_ranges) {
        std::map<ModifierOnPart, ShellRampZone> in_range;
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
                zone.top = { key.top_base_layers, key.top_base_thickness, config.top_shell_layers.value, config.top_shell_thickness.value,
                             config.top_shell_ramp_taper_ratio.value, layer_height };
            if (key.bottom_base_layers >= 0)
                zone.bottom = { key.bottom_base_layers, key.bottom_base_thickness, config.bottom_shell_layers.value, config.bottom_shell_thickness.value,
                                config.bottom_shell_ramp_taper_ratio.value, layer_height };
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
            if (inconsistent != nullptr)
                inconsistent->emplace_back(volumes);
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
