#include "WallBandShape.hpp"

#include "ClipperUtils.hpp"
#include "Point.hpp"
#include "ExPolygon.hpp"
#include "Polygon.hpp"
#include "libslic3r.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace Slic3r {

// Unit normals at the vertices of a closed polygon, pointing to the right of its direction: out of
// the material for a CCW contour and for a CW hole. At a vertex, the mean of the two edge normals.
static std::vector<Vec2d> vertex_normals(const Points &pts)
{
    auto edge_normal = [&pts](size_t i) -> Vec2d {
        const Vec2d v = (pts[(i + 1) % pts.size()] - pts[i]).cast<double>();
        const double l = v.norm();
        return l > 0. ? Vec2d(v.y() / l, - v.x() / l) : Vec2d(Vec2d::Zero());
    };
    std::vector<Vec2d> out;
    out.reserve(pts.size());
    Vec2d prev = edge_normal(pts.size() - 1);
    for (size_t i = 0; i < pts.size(); ++ i) {
        const Vec2d next = edge_normal(i);
        const Vec2d n    = prev + next;
        const double l   = n.norm();
        out.emplace_back(l > 0. ? Vec2d(n / l) : next);
        prev = next;
    }
    return out;
}

std::optional<WallBand> build_wall_band(const ExPolygon &island, const WallThicknessSampler &thickness, const WallBandParams &params)
{
    const double spacing = std::max<double>(scale_(params.sample_spacing), SCALED_EPSILON);

    // Simplifying drops vertices too close for the offset. It rarely splits or drops an island; then
    // take the island as it is.
    ExPolygons simplified = island.simplify(SCALED_EPSILON);
    ExPolygon  resampled  = simplified.size() == 1 ? std::move(simplified.front()) : island;

    // One delta per vertex of the contour and of each hole, negative to offset into the material.
    std::vector<std::vector<float>> deltas;
    deltas.reserve(resampled.holes.size() + 1);
    bool all_base = true;
    for (size_t idx = 0; idx <= resampled.holes.size(); ++ idx) {
        Polygon &poly = idx == 0 ? resampled.contour : resampled.holes[idx - 1];
        poly.points   = resample_polygon(poly.points, spacing);
        if (poly.points.size() < 3)
            return std::nullopt;
        const std::vector<WallThicknessSample> samples = thickness(poly.points, vertex_normals(poly.points));
        assert(samples.size() == poly.points.size());
        if (samples.size() != poly.points.size())
            return std::nullopt;
        std::vector<float> &ds = deltas.emplace_back();
        ds.reserve(samples.size());
        for (const WallThicknessSample &sample : samples) {
            double t = std::max(0., sample.thickness);
            if (std::abs(t - sample.base) > params.thickness_tolerance)
                all_base = false;
            // Too thin to print: no walls here, so the band ends where a ramp to no walls gets this thin.
            if (t < params.min_band_width)
                t = 0.;
            ds.emplace_back(- float(scale_(t)));
        }
    }

    // Base thickness everywhere: the stock walls are the right answer.
    if (all_base)
        return std::nullopt;

    // Where the island is narrower than 2t the offset vanishes, so the band takes the whole island
    // there. Where t = 0 the offset runs along the resampled contour, so the band below is empty there.
    WallBand out;
    out.inner = variable_offset_inner_ex(resampled, deltas);
    // Pieces of inner region too small to fill go to the band.
    const double min_area = std::max<double>(sqr(scale_(params.min_band_width)), sqr(double(SCALED_EPSILON)));
    out.inner.erase(std::remove_if(out.inner.begin(), out.inner.end(), [min_area](const ExPolygon &expoly) { return expoly.area() < min_area; }),
                    out.inner.end());
    // The band's outer edge is the resampled contour the offset was built from, so no sliver is left
    // between the two where t = 0.
    out.band = diff_ex(resampled, to_polygons(out.inner));
    return out;
}

} // namespace Slic3r
