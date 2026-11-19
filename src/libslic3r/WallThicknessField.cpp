#include "WallThicknessField.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <admesh/stl.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "BoundingBox.hpp"
#include "Point.hpp"
#include "SurfaceZoneDistance.hpp"

namespace Slic3r {

double wall_ramp_thickness(double distance, int base_wall_loops, int zone_wall_loops, double taper_ratio, double bead_width)
{
    const int change = zone_wall_loops - base_wall_loops;
    if (change == 0 || taper_ratio <= 0.)
        return base_wall_loops * bead_width;
    return bead_width * (zone_wall_loops - change * std::clamp(distance / (std::abs(change) * taper_ratio * bead_width), 0., 1.));
}

namespace {

// The thickness field of one modifier: the wall counts and ramp of its zone, and the distance from
// the zone's boundary.
struct ZoneField
{
    int    base_wall_loops { 0 };
    int    wall_loops { 0 };
    double taper_ratio { 0. };
    double ramp_length { 0. };
    SurfaceZoneDistance distance;

    bool active() const { return distance.active(); }
};

} // namespace

struct WallThicknessField::Impl
{
    WallThicknessFieldParams params;
    std::vector<ZoneField>   zones;
};

WallThicknessField::WallThicknessField(std::unique_ptr<const Impl> impl) : m_impl(std::move(impl)) {}
WallThicknessField::~WallThicknessField() = default;

std::shared_ptr<const WallThicknessField> WallThicknessField::build(const std::vector<indexed_triangle_set> &parts, const std::vector<WallRampZone> &zones,
                                                                    const WallThicknessFieldParams &params)
{
    if (zones.empty() || params.bead_width <= 0. || params.max_edge_fraction <= 0.)
        return nullptr;

    auto impl = std::make_unique<Impl>();
    impl->params = params;
    impl->params.max_shift_fraction = std::max(0., params.max_shift_fraction);
    impl->params.edge_growth        = std::clamp(params.edge_growth, 0., 1.);
    impl->zones.resize(zones.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, zones.size()), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i < range.end(); ++ i) {
            const WallRampZone &in   = zones[i];
            ZoneField          &zone = impl->zones[i];
            zone.base_wall_loops = in.base_wall_loops;
            zone.wall_loops      = in.wall_loops;
            zone.taper_ratio     = in.taper_ratio;
            if (in.part >= parts.size() || in.base_wall_loops < 0 || in.wall_loops < 0 || in.wall_loops == in.base_wall_loops || ! (in.taper_ratio > 0.))
                continue;
            zone.ramp_length = std::abs(in.wall_loops - in.base_wall_loops) * in.taper_ratio * params.bead_width;
            SurfaceZoneDistanceParams distance_params;
            distance_params.fine_edge         = impl->params.max_edge_fraction * params.bead_width;
            distance_params.coarse_edge       = impl->params.coarse_edge_fraction * params.bead_width;
            distance_params.edge_growth       = impl->params.edge_growth;
            distance_params.surface_tolerance = impl->params.surface_tolerance;
            distance_params.inside_limit      = impl->params.max_shift_fraction * in.taper_ratio * params.bead_width;
            distance_params.outside_limit     = zone.ramp_length + distance_params.inside_limit;
            zone.distance = SurfaceZoneDistance::build(parts[in.part], in.mesh, distance_params);
        }
    });
    if (std::none_of(impl->zones.begin(), impl->zones.end(), [](const ZoneField &zone) { return zone.active(); }))
        return nullptr;
    return std::shared_ptr<const WallThicknessField>(new WallThicknessField(std::move(impl)));
}

size_t WallThicknessField::zone_count() const { return m_impl->zones.size(); }
bool   WallThicknessField::zone_active(size_t zone) const { return m_impl->zones[zone].active(); }
double WallThicknessField::ramp_length(size_t zone) const { return m_impl->zones[zone].ramp_length; }
double WallThicknessField::signed_distance(size_t zone, const Vec3d &p) const { return m_impl->zones[zone].distance.signed_distance(p); }

WallThickness WallThicknessField::at(const Vec3d &p, double shift_fraction, double default_base) const
{
    const WallThicknessFieldParams &params = m_impl->params;
    const double  shift   = std::clamp(shift_fraction, - params.max_shift_fraction, params.max_shift_fraction) * params.bead_width;
    // Thicker wins: the thickest of the zones that thicken the walls here, else the thinnest of
    // those that thin them, each against its own base.
    WallThickness thicker { - std::numeric_limits<double>::infinity(), 0. };
    WallThickness thinner { std::numeric_limits<double>::infinity(), 0. };
    for (const ZoneField &zone : m_impl->zones) {
        if (! zone.active())
            continue;
        const double base      = zone.base_wall_loops * params.bead_width;
        const double thickness = wall_ramp_thickness(zone.distance.signed_distance(p) + shift * zone.taper_ratio, zone.base_wall_loops, zone.wall_loops,
                                                     zone.taper_ratio, params.bead_width);
        if (thickness > base && thickness > thicker.thickness)
            thicker = { thickness, base };
        else if (thickness < base && thickness < thinner.thickness)
            thinner = { thickness, base };
    }
    if (thicker.thickness > - std::numeric_limits<double>::infinity())
        return thicker;
    if (thinner.thickness < std::numeric_limits<double>::infinity())
        return thinner;
    return { default_base, default_base };
}

std::vector<WallThickness> WallThicknessField::at_contour(const std::vector<Vec2d> &points, const std::vector<Vec2d> &outward_normals, double slice_z,
                                                          double shift_fraction, double default_base) const
{
    const double               offset = m_impl->params.contour_offset_fraction * m_impl->params.bead_width;
    std::vector<WallThickness> out;
    out.reserve(points.size());
    for (size_t i = 0; i < points.size(); ++ i) {
        Vec2d point = points[i];
        if (i < outward_normals.size())
            if (const double length = outward_normals[i].norm(); length > 0.)
                point += (offset / length) * outward_normals[i];
        out.emplace_back(this->at(Vec3d(point.x(), point.y(), slice_z), shift_fraction, default_base));
    }
    return out;
}

bool WallThicknessField::may_affect(const BoundingBoxf3 &box) const
{
    if (! box.defined)
        return false;
    const Eigen::AlignedBox3d query(box.min, box.max);
    return std::any_of(m_impl->zones.begin(), m_impl->zones.end(), [&query](const ZoneField &zone) { return zone.active() && zone.distance.reach().intersects(query); });
}

double WallThicknessField::min_thickness() const
{
    double thickness = std::numeric_limits<double>::infinity();
    for (const ZoneField &zone : m_impl->zones)
        if (zone.active())
            thickness = std::min(thickness, std::min(zone.base_wall_loops, zone.wall_loops) * m_impl->params.bead_width);
    return thickness;
}

double WallThicknessField::max_thickness() const
{
    double thickness = 0.;
    for (const ZoneField &zone : m_impl->zones)
        if (zone.active())
            thickness = std::max(thickness, std::max(zone.base_wall_loops, zone.wall_loops) * m_impl->params.bead_width);
    return thickness;
}

double WallThicknessField::bead_width() const { return m_impl->params.bead_width; }

double WallThicknessField::min_taper_ratio() const
{
    double ratio = std::numeric_limits<double>::infinity();
    for (const ZoneField &zone : m_impl->zones)
        if (zone.active())
            ratio = std::min(ratio, zone.taper_ratio);
    return ratio;
}

size_t WallThicknessField::triangle_count() const
{
    size_t count = 0;
    for (const ZoneField &zone : m_impl->zones)
        count += zone.distance.triangle_count();
    return count;
}

} // namespace Slic3r
