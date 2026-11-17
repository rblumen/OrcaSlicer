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
#include "libslic3r.h"

#include <functional>
#include <optional>

namespace Slic3r {

// Thicknesses in mm at one contour sample.
struct WallThicknessSample
{
    // Wall thickness t there. Negative values are treated as 0.
    double thickness { 0. };
    // Base thickness N0 * w of the part the sample lies on: the thickness its walls have outside the
    // zone and the ramp. It may differ along the contour, where one modifier overlaps parts with
    // different wall counts and each piece ramps to its own part's walls.
    double base { 0. };
};

// The thicknesses at a contour sample `pt`, given in scaled layer coordinates. The caller maps the
// sample to the part surface at the layer's print_z, looks up the distance d from the zone there
// and evaluates t(d). The brick-bond shift belongs to the caller as well: it shifts d by +delta on
// even layers and by -delta on odd layers (delta = rho * w / 4) before evaluating t, which moves the
// whole ramp, and with it the drop stations, by delta along the surface.
using WallThicknessSampler = std::function<WallThicknessSample(const Point &pt)>;

struct WallBandParams
{
    // Largest distance between two contour samples in mm. Use at most w / 4, so the band's inner
    // edge follows the thickness closely enough for Arachne to place the drops where t says.
    double sample_spacing { 0. };
    // Parts of the band narrower than this many mm are dropped and become inner region, so the
    // band never hands Arachne a sliver it cannot print. About w / 2; where a ramp thins the walls
    // to 0, the band ends where t falls below it.
    double min_band_width { 0. };
    // A thickness within this many mm of its sample's base counts as the base.
    double thickness_tolerance { 0.001 };

    // The defaults for bead width w (mm): samples every w / 4, band at least w / 2 wide.
    static WallBandParams for_bead_width(double bead_width)
    {
        WallBandParams out;
        out.sample_spacing = bead_width / 4.;
        out.min_band_width = bead_width / 2.;
        return out;
    }
};

struct WallBand
{
    // The band. Where the island is narrower than 2t, it is the whole island there, as with today's
    // thin features. It is empty where t = 0, and may be empty altogether.
    ExPolygons band;
    // The island minus the band, before the infill/wall overlap is applied. Where t = 0 it reaches
    // the island's contour.
    ExPolygons inner;
};

// Builds the wall band of one island. The contour and the holes are resampled so that consecutive
// vertices are at most params.sample_spacing apart, t is evaluated at every vertex, and the island
// is offset inward by t per vertex with variable_offset_inner_ex(), as elephant foot compensation
// does. The band is the island minus that offset, opened by params.min_band_width to drop slivers.
// Returns no band when t is its sample's base thickness at every sample, so the caller can take the stock
// wall path unchanged.
std::optional<WallBand> build_wall_band(const ExPolygon &island, const WallThicknessSampler &thickness, const WallBandParams &params);

} // namespace Slic3r
