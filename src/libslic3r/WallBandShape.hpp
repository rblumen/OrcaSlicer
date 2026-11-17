#pragma once

// Shape of the variable-thickness wall band of the Ramp wall count transition. On one layer, the
// band of an island is the region within the local wall thickness t of the island's boundary,
// measured inward along the contour normals. Its inner edge is the wall/infill boundary: Arachne
// fills the band with walls, and the rest of the island is infill.
//
// Terms: the zone is the part surface inside the Ramp modifier, where t is the modifier's wall count
// times w. The ramp is only the tapered stretch around the zone, where t goes linearly from the
// zone's thickness to the base thickness of the part. The zone may be thicker than the base
// (reinforcement) or thinner, down to t = 0 for a modifier with wall_loops = 0. Where t = 0 the band
// has zero width: there are no walls there, the outer wall ends, and infill reaches the contour, as
// with a stock modifier that has no walls.
//
// The band is built from the island alone and a thickness lookup, so it does not depend on how the
// thickness is computed. The thickness field on the object mesh supplies the lookup.

#include "ExPolygon.hpp"
#include "Point.hpp"

#include <functional>
#include <optional>
#include <vector>

namespace Slic3r {

// Thicknesses in mm at one contour sample.
struct WallThicknessSample
{
    // Wall thickness t there. Negative values are treated as 0.
    double thickness { 0. };
    // Base thickness N0 * w of the part the sample lies on: the thickness its walls have outside the
    // zone and the ramp. Regions are keyed so that parts with different wall counts are in separate
    // wall groups, so the base is the same at every sample of an island. The builder still compares
    // each sample with its own base.
    double base { 0. };
};

// Called once per polygon of the island, the contour first and then each hole, with its resampled
// vertices in scaled layer coordinates and the unit normals at them, pointing out of the material
// (away from the island on the contour, into the hole on a hole). The normals let the lookup pick
// the right face of a thin wall, or find the surface after XY or elephant foot compensation. Returns
// one sample per point, in order.
//
// The caller maps each point to the part surface at the layer's print_z, looks up the distance d
// from the zone there and evaluates t(d). The brick-bond shift belongs to the caller as well: it
// shifts d by +delta on even layers and by -delta on odd layers (delta = rho * w / 4) before
// evaluating t, which moves the whole ramp, and with it the drop stations, by delta along the surface.
using WallThicknessSampler = std::function<std::vector<WallThicknessSample>(const Points &points, const std::vector<Vec2d> &normals)>;

struct WallBandParams
{
    // Largest distance between two contour samples in mm. Use at most w / 4, so the band's inner
    // edge follows the thickness closely enough for Arachne to place the drops where t says.
    double sample_spacing { 0.1 };
    // Thinnest wall in mm the band may have, set from the region's min_bead_width and
    // min_feature_size. Where t is below it, t is taken as 0: no walls there, which is where a ramp
    // to no walls ends.
    double min_band_width { 0. };
    // A thickness within this many mm of its sample's base counts as the base.
    double thickness_tolerance { 0.001 };

    // Samples every w / 4 for bead width w (mm).
    static WallBandParams for_bead_width(double bead_width, double min_band_width)
    {
        WallBandParams out;
        out.sample_spacing = bead_width / 4.;
        out.min_band_width = min_band_width;
        return out;
    }
};

struct WallBand
{
    // The band. Where the island is narrower than 2t, it is the whole island there, as with today's
    // thin features. It is empty where t = 0, and may be empty altogether. Its outer edge is the
    // island's contour, simplified by SCALED_EPSILON.
    ExPolygons band;
    // The island minus the band, before the infill/wall overlap is applied. Where t = 0 it reaches
    // the island's contour.
    ExPolygons inner;
};

// Builds the wall band of one island. The island is simplified by SCALED_EPSILON, its contour and
// holes are resampled so that consecutive vertices are at most params.sample_spacing apart, and t
// is looked up at every vertex. `inner` is the island offset inward by t per vertex with
// variable_offset_inner_ex(), as elephant foot compensation does, without pieces too small to fill;
// the band is the island minus `inner`. Returns no band when t is its sample's base at every sample,
// so the caller can take the stock wall path unchanged.
std::optional<WallBand> build_wall_band(const ExPolygon &island, const WallThicknessSampler &thickness, const WallBandParams &params);

} // namespace Slic3r
