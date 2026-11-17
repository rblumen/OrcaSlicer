#include "WallBandShape.hpp"

#include "ClipperUtils.hpp"
#include "Polygon.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace Slic3r {

// Splits every edge of a closed contour into equal pieces no longer than `spacing` (scaled). The
// original vertices are kept, so corners stay sharp.
static Points resample_closed(const Points &contour, double spacing)
{
    Points out;
    out.reserve(contour.size());
    for (size_t i = 0; i < contour.size(); ++ i) {
        const Point &a = contour[i];
        const Point &b = contour[(i + 1) % contour.size()];
        out.emplace_back(a);
        const Vec2d  v = (b - a).cast<double>();
        const size_t pieces = size_t(std::ceil(v.norm() / spacing));
        for (size_t k = 1; k < pieces; ++ k) {
            const Vec2d p = a.cast<double>() + v * (double(k) / double(pieces));
            out.emplace_back(coord_t(std::round(p.x())), coord_t(std::round(p.y())));
        }
    }
    return out;
}

std::optional<WallBand> build_wall_band(const ExPolygon &island, const WallThicknessSampler &thickness, const WallBandParams &params)
{
    assert(params.sample_spacing > 0.);
    const double spacing = scale_(params.sample_spacing);

    ExPolygon resampled = island;
    // One delta per vertex of the contour and of each hole, negative to offset into the material.
    std::vector<std::vector<float>> deltas;
    deltas.reserve(island.holes.size() + 1);
    bool all_base = true;
    for (size_t idx = 0; idx <= island.holes.size(); ++ idx) {
        Polygon &poly = idx == 0 ? resampled.contour : resampled.holes[idx - 1];
        poly.points   = resample_closed(poly.points, spacing);
        std::vector<float> &ds = deltas.emplace_back();
        ds.reserve(poly.points.size());
        for (const Point &pt : poly.points) {
            const WallThicknessSample sample = thickness(pt);
            const double              t      = std::max(0., sample.thickness);
            if (std::abs(t - sample.base) > params.thickness_tolerance)
                all_base = false;
            ds.emplace_back(- float(scale_(t)));
        }
    }

    // Base thickness everywhere: the stock walls are the right answer.
    if (all_base)
        return std::nullopt;

    // Where the island is narrower than 2t the offset vanishes, so the band takes the whole island there.
    ExPolygons band = diff_ex(island, variable_offset_inner_ex(resampled, deltas));
    // Where t is below the minimum width, down to t = 0 where the offset runs along the contour,
    // the difference leaves slivers. Opening drops them and the thin end of a ramp to 0.
    if (params.min_band_width > 0.)
        band = intersection_ex(opening(band, float(scale_(0.5 * params.min_band_width))), to_polygons(island));
    WallBand out;
    out.inner = diff_ex(island, to_polygons(band));
    out.band  = std::move(band);
    return out;
}

} // namespace Slic3r
