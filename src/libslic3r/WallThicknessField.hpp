#pragma once

// Graded wall reinforcement: the wall thickness field of an object.
//
// A volume modifier set to the Ramp wall count transition changes the wall count where a part's
// surface lies inside it, the zone, from the part's count N0 to the modifier's N1. Outside the zone
// the count returns to N0 over the ramp, a stretch of the surface L_r long. Wall thickness t is a
// function of the geodesic distance d along the part's surface from the zone's boundary, negative
// inside the zone:
//
//     t(d) = w * (N1 - dN * clamp(d / L_r, 0, 1)),    dN = N1 - N0,    L_r = |dN| * rho * w
//
// w is the bead width and rho the taper ratio. A zone thickens the walls when N1 > N0 and thins
// them when N1 < N0, down to none at N1 = 0; either way its ramp lies outside it. A zone with
// N1 = N0 has no effect. A face the modifier shares with the part from inside belongs to the zone;
// one it rests on from outside does not.
//
// Each zone carries its own N0, because one modifier can overlap several parts with different
// wall counts, and its piece on each part tapers to that part's walls: the caller passes one zone
// per piece, on that part's mesh.
//
// Where zones and ramps overlap, thicker wins. If any zone gives a t above its own N0 * w at a
// point, the thickness is the largest such t; otherwise, if any gives a t below its own N0 * w, it
// is the smallest such t; otherwise no zone reaches the point and the caller's default applies.
// The base thickness at the point is the N0 * w of the zone that decided its thickness, or the
// caller's default where none did, so that a point is unchanged exactly where its thickness equals
// its base. Each zone is compared with its own N0, so where pieces on different parts meet, a
// thickening zone wins over a thinning one even when its t is the lower of the two.
//
// The distance comes from the heat method on the part's surface remeshed around each zone, with
// edges of w/2 at its boundary growing to w where the distance is read. On faces and across bends
// it is within 2%, or 0.05 mm near the boundary. Where distances arriving by two routes meet, such
// as behind the corner of a hole or on the narrow side of a plate between its top and bottom faces,
// the heat method smooths the distance, and nearby it is up to about 4% long. Where the solve does not reach, as on a second
// shell with no boundary of its own, the straight line distance to the boundary stands in for it.
//
// The part and modifier meshes have to be closed and consistently oriented: whether a point is in
// the zone is decided by winding numbers. A zone narrower than about w/2 may be missed.
//
// The field is built once per object and only read afterwards.

#include <cstddef>
#include <memory>
#include <vector>

#include <admesh/stl.h>

#include "Point.hpp"

namespace Slic3r {

class BoundingBoxf3;

// One Ramp modifier's piece on one part.
struct WallRampZone
{
    // The modifier's mesh, closed, in the same frame as the part meshes.
    indexed_triangle_set mesh;
    // Index of the part, into the part meshes passed to WallThicknessField::build(), whose surface
    // the zone lies on.
    size_t               part { 0 };
    // N0, the part's wall count, which the ramp returns to.
    int                  base_wall_loops { 2 };
    // N1, the wall count inside the zone. A zone is ignored if this equals N0.
    int                  wall_loops { 0 };
    // rho, mm of ramp per mm of wall thickness changed. A zone is ignored unless this is positive.
    double               taper_ratio { 10. };
};

struct WallThicknessFieldParams
{
    // w, the nominal bead width in mm.
    double bead_width { 0.45 };
    // Largest |shift_fraction| that a query is asked for. Distances are resolved from this far
    // inside a zone's boundary, in units of rho * w, to this far past the end of its ramp.
    double max_shift_fraction { 0.25 };
    // Longest mesh edge at a zone's boundary, in bead widths.
    double max_edge_fraction { 0.5 };
    // Longest mesh edge where the distance is solved, in bead widths.
    double coarse_edge_fraction { 1. };
    // Growth of the edge length away from a zone's boundary, mm per mm, from max_edge_fraction
    // up to coarse_edge_fraction bead widths.
    double edge_growth { 0.1 };
    // How far in mm a query point may lie off a part's surface and still be on it. A contour lies
    // off the surface by the slicing itself (up to half a layer on a slope), elephant foot and XY
    // compensation, and at_contour() moves it out by contour_offset_fraction * w on top. A point
    // further off is out of every zone of that part, and gets the caller's default.
    double surface_tolerance { 0.5 };
    // at_contour() looks the field up this many bead widths outside the contour, along its normal,
    // so that on a wall thinner than the offsets above the face the contour came from is found,
    // not the face behind it.
    double contour_offset_fraction { 0.5 };
};

// Wall thickness at a point, and the base it is compared with, in mm.
struct WallThickness
{
    double thickness { 0. };
    double base { 0. };
};

// Wall thickness over the surface of one object, fed by all its Ramp modifiers.
//
// The part meshes, the zone meshes and every query point share one frame, in unscaled mm. That has
// to be the frame of the layer slices, so that a point of a layer's contour, with the layer's
// slice_z as its Z, lies on a part mesh.
//
// Immutable once built: any number of threads may query it concurrently.
class WallThicknessField
{
public:
    // Returns nullptr if no zone changes the wall count anywhere on the surface of its part: there
    // is no zone, a zone's wall count equals its base count or its taper ratio is not positive, or
    // its modifier does not reach the surface (it lies wholly inside or outside the part).
    static std::shared_ptr<const WallThicknessField> build(const std::vector<indexed_triangle_set> &parts, const std::vector<WallRampZone> &zones,
                                                           const WallThicknessFieldParams &params);

    ~WallThicknessField();
    WallThicknessField(const WallThicknessField &)            = delete;
    WallThicknessField &operator=(const WallThicknessField &) = delete;

    // Number of zones passed to build(). Zone indices below are indices into that vector.
    size_t zone_count() const;
    // Whether the zone changes the wall count anywhere on the surface.
    bool   zone_active(size_t zone) const;
    // L_r of the zone in mm.
    double ramp_length(size_t zone) const;

    // Signed geodesic distance in mm from the zone's boundary to the nearest point of its part's
    // surface to p: negative inside the zone, zero on its boundary. Resolved only where the
    // thickness depends on it, so it saturates at -max_shift_fraction * rho * w inside the zone and
    // at L_r + max_shift_fraction * rho * w outside, which is also what a point further than
    // surface_tolerance off the part gets. Infinite for an inactive zone.
    double signed_distance(size_t zone, const Vec3d &p) const;

    // Wall thickness and base at the surface point nearest to p, combined over the zones as the top
    // of the file describes; default_base where no zone or ramp reaches. shift_fraction slides each
    // zone's ramp along the surface: t is evaluated at d + shift_fraction * rho * w, so a positive
    // value moves the ramp towards the zone (the brick bond passes +1/4 and -1/4 on alternate
    // layers). It is clamped to max_shift_fraction.
    WallThickness at(const Vec3d &p, double shift_fraction, double default_base) const;
    double        thickness(const Vec3d &p, double shift_fraction, double default_base) const { return this->at(p, shift_fraction, default_base).thickness; }

    // at() for each point of a contour of the slice at slice_z, looked up contour_offset_fraction *
    // w outside it along its outward normal (any length; a zero normal looks up the point itself).
    std::vector<WallThickness> at_contour(const std::vector<Vec2d> &points, const std::vector<Vec2d> &outward_normals, double slice_z,
                                          double shift_fraction, double default_base) const;

    // Whether any zone or its ramp may reach into the box. If not, every point in it gets the
    // caller's default, and a layer or contour within it can be skipped.
    bool may_affect(const BoundingBoxf3 &box) const;

    // Bounds of the thickness over the active zones and their ramps, min(N0, N1) * w and
    // max(N0, N1) * w. The caller's default applies elsewhere.
    double min_thickness() const;
    double max_thickness() const;
    // w, the bead width the thicknesses are multiples of, in mm.
    double bead_width() const;
    // The smallest taper ratio rho of the active zones, which gives the steepest ramp.
    double min_taper_ratio() const;
    // Triangles of the remeshed surfaces of all zones, for diagnostics.
    size_t triangle_count() const;

private:
    struct Impl;
    explicit WallThicknessField(std::unique_ptr<const Impl> impl);

    std::unique_ptr<const Impl> m_impl;
};

// t(d) of one zone, see the top of the file. All lengths in mm.
double wall_ramp_thickness(double distance, int base_wall_loops, int zone_wall_loops, double taper_ratio, double bead_width);

} // namespace Slic3r
