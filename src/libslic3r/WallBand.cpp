#include "WallBand.hpp"

#include "BoundingBox.hpp"
#include "ClipperUtils.hpp"
#include "Flow.hpp"
#include "Geometry.hpp"
#include "Layer.hpp"
#include "Model.hpp"
#include "PerimeterGenerator.hpp"
#include "Point.hpp"
#include "Print.hpp"
#include "PrintConfig.hpp"
#include "Surface.hpp"
#include "WallBandShape.hpp"
#include "WallThicknessField.hpp"
#include "libslic3r.h"

#include <admesh/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <utility>
#include <vector>

namespace Slic3r {

std::shared_ptr<const WallThicknessField> build_wall_thickness_field(const PrintObject &object)
{
    const PrintObjectRegions *regions = object.shared_regions();
    if (regions == nullptr)
        return nullptr;

    // One zone per modifier and part. A modifier nested in another one has a region over the part and one over
    // the outer modifier's region; the latter comes later and wins, so its base is the outer modifier's walls.
    std::map<std::pair<const ModelVolume*, const ModelVolume*>, WallRampZone> zones;
    std::vector<const ModelVolume*>                                         parts;
    const PrintRegion                                                      *first_part_region = nullptr;
    for (const PrintObjectRegions::LayerRangeRegions &layer_range : regions->layer_ranges)
        for (const PrintObjectRegions::VolumeRegion &volume_region : layer_range.volume_regions) {
            if (volume_region.parent < 0 || volume_region.region == nullptr)
                continue;
            const PrintRegion &modifier = *volume_region.region;
            const PrintRegion &parent   = *layer_range.volume_regions[volume_region.parent].region;
            // Only a modifier whose region shares the walls of its parent's, see PrintObject::make_perimeters().
            if (! modifier.has_wall_ramp() || modifier.config().wall_loops == parent.config().wall_loops ||
                modifier.gradient_volume_id() != parent.gradient_volume_id() || ! Layer::is_perimeter_compatible(*object.print(), modifier, parent))
                continue;
            int root = volume_region.parent;
            while (layer_range.volume_regions[root].parent >= 0)
                root = layer_range.volume_regions[root].parent;
            const ModelVolume *part = layer_range.volume_regions[root].model_volume;
            if (first_part_region == nullptr)
                first_part_region = layer_range.volume_regions[root].region;
            auto it_part = std::find(parts.begin(), parts.end(), part);
            WallRampZone &zone   = zones[{ volume_region.model_volume, part }];
            zone.part            = size_t(it_part - parts.begin());
            zone.base_wall_loops = modifier.wall_ramp_base_loops();
            zone.wall_loops      = modifier.config().wall_loops.value;
            zone.taper_ratio     = modifier.config().wall_ramp_taper_ratio.value;
            if (it_part == parts.end())
                parts.emplace_back(part);
        }
    if (zones.empty())
        return nullptr;

    // The meshes in the frame the layers were sliced in, see PrintObjectSlice.cpp.
    const Transform3d trafo = object.trafo_sliced();
    auto sliced_mesh = [&trafo](const ModelVolume &volume) {
        indexed_triangle_set its = volume.mesh().its;
        its_transform(its, trafo * volume.get_matrix(), true);
        return its;
    };
    std::vector<indexed_triangle_set> part_meshes;
    part_meshes.reserve(parts.size());
    for (const ModelVolume *part : parts)
        part_meshes.emplace_back(sliced_mesh(*part));
    std::vector<WallRampZone> zone_list;
    zone_list.reserve(zones.size());
    for (auto &[volumes, zone] : zones) {
        zone.mesh = sliced_mesh(*volumes.first);
        zone_list.emplace_back(std::move(zone));
    }

    const PrintObjectConfig &config = object.config();
    double max_layer_height = 0.;
    for (const Layer *layer : object.layers())
        max_layer_height = std::max(max_layer_height, layer->height);
    WallThicknessFieldParams params;
    // The thicknesses are multiples of the spacing of the inner walls, as Arachne lays them.
    params.bead_width = first_part_region->flow(object, frPerimeter, config.layer_height.value).spacing();
    // A contour lies off its part's surface by the largest of: elephant foot compensation and XY compensation,
    // which move it in or out, and half a layer, the slice plane's distance from the layer's top or bottom on
    // a slope. at_contour() looks it up another contour_offset_fraction * w outside on top of that.
    params.surface_tolerance = params.contour_offset_fraction * params.bead_width +
        std::max({ config.elefant_foot_compensation.value, std::abs(config.xy_contour_compensation.value),
                   std::abs(config.xy_hole_compensation.value), 0.5 * max_layer_height });
    return WallThicknessField::build(part_meshes, zone_list, params);
}

std::optional<WallBandPaths> wall_band_paths(const PerimeterGenerator &generator, const Surface &surface, const ExPolygons &island,
                                             const Arachne::WallToolPathsParams &params)
{
    const WallThicknessField *field = generator.wall_thickness_field;
    if (field == nullptr || island.empty())
        return std::nullopt;
    // Wall paths the band does not support yet, see WallBand.hpp.
    const PrintRegionConfig &config = *generator.config;
    if (config.only_one_wall_top || (config.only_one_wall_first_layer && generator.layer_id == generator.object_config->raft_layers) ||
        config.alternate_extra_wall || (config.detect_overhang_wall && config.extra_perimeters_on_overhangs) || surface.extra_perimeters != 0)
        return std::nullopt;

    const BoundingBox extents = get_extents(island);
    // A box needs some height to be defined: the layer's.
    if (! field->may_affect(BoundingBoxf3(Vec3d(unscaled(extents.min.x()), unscaled(extents.min.y()), generator.slice_z - 0.5 * generator.layer_height),
                                          Vec3d(unscaled(extents.max.x()), unscaled(extents.max.y()), generator.slice_z + 0.5 * generator.layer_height))))
        return std::nullopt;

    // The field's thicknesses are multiples of its bead width, the band's of the layer's.
    const double w            = field->bead_width();
    const double layer_w      = generator.perimeter_flow.spacing();
    const double to_layer     = layer_w / w;
    const double default_base = config.wall_loops.value * w;
    // Brick bond: the ramp moves by a quarter of a drop spacing towards the zone on even layers, away on odd ones.
    const double shift = generator.layer_id % 2 == 0 ? 0.25 : -0.25;
    const WallThicknessSampler sampler = [&](const Points &points, const std::vector<Vec2d> &normals) {
        std::vector<Vec2d> unscaled_points;
        unscaled_points.reserve(points.size());
        for (const Point &point : points)
            unscaled_points.emplace_back(unscaled(point));
        std::vector<WallThicknessSample> out;
        out.reserve(points.size());
        for (const WallThickness &t : field->at_contour(unscaled_points, normals, generator.slice_z, shift, default_base))
            out.push_back({ t.thickness * to_layer, t.base * to_layer });
        return out;
    };
    const WallBandParams band_params = WallBandParams::for_bead_width(layer_w, std::max(params.min_bead_width, params.min_feature_size));

    bool       has_band = false;
    ExPolygons band, inner;
    for (const ExPolygon &piece : island)
        if (std::optional<WallBand> piece_band = build_wall_band(piece, sampler, band_params)) {
            has_band = true;
            append(band, std::move(piece_band->band));
            append(inner, std::move(piece_band->inner));
        } else {
            // A piece at its base thickness takes the same walls, so that one wall count fits all pieces.
            ExPolygons piece_inner = offset_ex(piece, - float(scale_(default_base * to_layer)));
            append(band, diff_ex(piece, piece_inner));
            append(inner, std::move(piece_inner));
        }
    if (! has_band)
        return std::nullopt;

    WallBandPaths out;
    out.outline     = to_polygons(band);
    out.loop_number = std::max<int>(int(std::lround(field->max_thickness() / w)), config.wall_loops.value);
    out.params      = params;
    out.params.wall_distribution_count = 1;
    // Arachne suppresses a bead count drop on a wedge narrower than its transition angle, and the ramp's inner
    // edge runs at atan(1 / rho) to its contour, so the angle is raised 5 degrees above that, in both directions.
    out.params.wall_transition_angle = float(std::max<double>(params.wall_transition_angle,
                                                              Geometry::rad2deg(std::atan(1. / field->min_taper_ratio())) + 5.));
    out.infill_contour = union_ex(inner);
    return out;
}

} // namespace Slic3r
