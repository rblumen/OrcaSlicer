#pragma once

// Graded wall reinforcement: the walls of the Ramp wall count transition. The object's thickness field
// (WallThicknessField.hpp) gives the wall thickness along each island's contour, the band
// (WallBandShape.hpp) is the region within that thickness of the contour, and Arachne fills the band
// with walls instead of filling the island with a fixed wall count. The rest of the island is infill.
//
// PerimeterGenerator::process_arachne() only swaps the band in, see wall_band_paths(), so that the
// stock wall path is untouched wherever no Ramp modifier changes the wall count.

#include "Arachne/WallToolPaths.hpp"
#include "ExPolygon.hpp"
#include "Polygon.hpp"

#include <memory>
#include <optional>

namespace Slic3r {

class PerimeterGenerator;
class PrintObject;
class Surface;
class WallThicknessField;

// The thickness field of the object, fed by every modifier with a ramped wall count transition whose region
// shares the walls of the region it modifies. nullptr if there is none, or none changes the wall count on the
// surface of a part.
std::shared_ptr<const WallThicknessField> build_wall_thickness_field(const PrintObject &object);

// What process_arachne() runs Arachne::WallToolPaths with for an island that has a band.
struct WallBandPaths
{
    // The band, in place of the island.
    Polygons                     outline;
    // 0-based wall count: the band's thickest walls plus one, so the count never cuts the band short.
    int                          loop_number { 0 };
    // The island's parameters with the transition angle raised above the ramp's and the distribution count
    // at 1, so that walls drop one at a time along the ramp and only the middle bead flexes.
    Arachne::WallToolPathsParams params;
    // The island minus the band, in place of the inner contour of the walls.
    ExPolygons                   infill_contour;
};

// The band of one island of the generator's layer, `island` being the island as offset for the outer wall. None
// if the generator has no thickness field, the field leaves the island at its base thickness, or the island
// takes a wall path the band does not support yet (one wall on top or on the first layer, alternating extra
// walls, extra walls on overhangs).
std::optional<WallBandPaths> wall_band_paths(const PerimeterGenerator &generator, const Surface &surface, const ExPolygons &island,
                                             const Arachne::WallToolPathsParams &params);

} // namespace Slic3r
