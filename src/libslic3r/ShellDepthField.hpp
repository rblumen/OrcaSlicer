#pragma once

// Graded shell reinforcement: the shell depth field of an object.
//
// The stock shell projection puts the layer k layers below a top surface (above a bottom surface), at a Z distance dz
// from it, in the shell where
//
//     k < n  or  dz < T - EPSILON,
//
// n being the shell layer count and T the minimum shell thickness, see ShellRule. A volume modifier set to the Ramp
// top (bottom) shell count transition changes that rule where a part's surface lies inside it, the zone, from the
// part's (n0, T0) to the modifier's (n1, T1). Outside the zone the rule returns to the part's over the ramp, a stretch
// of the surface L_s long, both of its terms linearly in the geodesic distance d along the part's surface from the
// zone's boundary, negative inside the zone:
//
//     n(d) = n1 - (n1 - n0) * t,    T(d) = T1 - (T1 - T0) * t,    t = clamp(d / L_s, 0, 1)
//
// So inside the zone the shell is the modifier's and beyond the ramp the part's, exactly as the stock projection
// builds them whatever the layer heights, and across the ramp the fractional n(d) steps the shell one layer at a time.
// A minimum thickness counts only with shell layers, as in the stock projection, so T is taken as 0 where n is.
//
// The ramp is as long as the change of the shell's depth at the nominal layer height h, see shell_depth(), times the
// shell taper ratio rho_s, in mm of ramp per mm of depth:
//
//     L_s = |D1 - D0| * rho_s,    D0 = shell_depth(n0, T0, h),    D1 = shell_depth(n1, T1, h)
//
// A zone thickens the shell when D1 > D0 and thins it when D1 < D0, down to no shell at n1 = 0; either way its ramp
// lies outside it. A zone with D1 = D0 changes nothing.
//
// Each zone has a top and a bottom profile, which ramp independently: one distance solve serves both. Each carries its
// own (n0, T0), because one modifier can overlap several parts with different shells, and its piece on each part
// tapers to that part's shell: the caller passes one zone per piece, on that part's mesh.
//
// Where zones and ramps overlap, the thicker shell wins. A zone decides at a point within it or its ramp. A layer is in
// the shell if any deciding zone that thickens puts it there; where only zones that thin decide, if all of them put it
// there; where no zone decides, the field has no opinion and the shell is the receiving region's own, see
// PrintRegion::shell_default_counts(). As the layers a zone puts in the shell are the k from 0 up to some count, this
// takes the largest count over the thickening zones, else the smallest over the thinning ones.
//
// The field is built once per object and only queried afterwards, from any number of threads.

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <admesh/stl.h>

#include "BoundingBox.hpp"
#include "Point.hpp"
#include "Polygon.hpp"

namespace Slic3r {

class ModelVolume;
class PrintObject;

enum class ShellSide { Top, Bottom };

// The depth in mm of a shell of `layers` layers and the given minimum thickness, at the nominal layer height: what the
// stock shell projection reaches with uniform layers. No shell when layers is 0, whatever the thickness.
double shell_depth(int layers, double thickness, double layer_height);

// Which layers next to a surface are in its shell, see the top of the file.
struct ShellRule
{
    // n, fractional across a ramp.
    double layers { 0. };
    // T in mm, 0 where n is.
    double thickness { 0. };

    // Whether the layer k layers from the surface, dz from it in Z, is in the shell.
    bool solid(size_t k, double dz) const;
};

// How a zone changes the shell on one side.
struct ShellRampProfile
{
    // n0 and T0, the shell the modifier modifies.
    int    base_layers { 0 };
    double base_thickness { 0. };
    // n1 and T1, the shell inside the zone.
    int    layers { 0 };
    double thickness { 0. };
    // rho_s, mm of ramp per mm of depth changed. 0 keeps the hard step.
    double taper_ratio { 0. };
    // The nominal layer height h of the depths L_s is taken from, in mm.
    double layer_height { 0. };

    // D0 and D1 in mm.
    double    base_depth() const { return shell_depth(base_layers, base_thickness, layer_height); }
    double    depth() const { return shell_depth(layers, thickness, layer_height); }
    // Whether the profile changes the shell anywhere: the taper ratio is positive and D1 differs from D0.
    bool      active() const;
    // Whether it thickens the shell, D1 > D0.
    bool      thickens() const { return this->depth() > this->base_depth(); }
    // L_s in mm.
    double    ramp_length() const;
    // The rule at d, see the top of the file. (n0, T0) for an inactive profile.
    ShellRule at(double distance) const;
};

// One Shell Ramp modifier's piece on one part.
struct ShellRampZone
{
    // The modifier's mesh, closed, in the same frame as the part meshes.
    indexed_triangle_set mesh;
    // Index of the part, into the part meshes passed to ShellDepthField::build(), whose surface the zone lies on.
    size_t               part { 0 };
    ShellRampProfile     top;
    ShellRampProfile     bottom;

    const ShellRampProfile &profile(ShellSide side) const { return side == ShellSide::Top ? top : bottom; }
};

struct ShellDepthFieldParams
{
    // Longest mesh edge at a zone's boundary, in mm.
    double fine_edge { 0.25 };
    // Longest mesh edge where the distance is solved, in mm.
    double coarse_edge { 0.5 };
    // Growth of the edge length away from a zone's boundary, mm per mm, from fine_edge up to coarse_edge.
    double edge_growth { 0.1 };
    // How far in mm a query point may lie off a part's surface and still be on it. A point further off has no opinion.
    double surface_tolerance { 0.5 };
    // The distance is resolved this far in mm past the end of the zone's longer ramp, so that where the ramp ends is
    // found to within a raster sample.
    double outside_margin { 1. };
};

// The shell rule over the surface of one object, fed by all its Shell Ramp modifiers.
//
// The part meshes, the zone meshes and every query point share one frame, in unscaled mm. That has to be the frame of
// the layer slices, so that a point of a layer's slices, with the layer's slice_z as its Z, lies near a part mesh.
class ShellDepthField
{
public:
    // Returns nullptr if no zone changes a shell anywhere on the surface of its part: there is no zone, neither of a
    // zone's profiles is active, or its modifier does not reach the surface (it lies wholly inside or outside the part).
    static std::shared_ptr<const ShellDepthField> build(const std::vector<indexed_triangle_set> &parts, const std::vector<ShellRampZone> &zones,
                                                        const ShellDepthFieldParams &params);

    ~ShellDepthField();
    ShellDepthField(const ShellDepthField &)            = delete;
    ShellDepthField &operator=(const ShellDepthField &) = delete;

    // Number of zones passed to build(). Zone indices below are indices into that vector.
    size_t zone_count() const;
    // Whether the zone changes a shell anywhere on the surface.
    bool   zone_active(size_t zone) const;
    // The profile of the zone on the given side.
    const ShellRampProfile &profile(size_t zone, ShellSide side) const;
    // Signed geodesic distance in mm from the zone's boundary to the nearest point of its part's surface to p,
    // saturated at 0 inside the zone and at the end of its longer ramp plus outside_margin outside it, which is also
    // what a point further than surface_tolerance off the part gets. Infinite for an inactive zone.
    double signed_distance(size_t zone, const Vec3d &p) const;

    // How many layers from a surface on the given side, at the surface point nearest to p, are in the shell, combined
    // over the zones as the top of the file describes; none where no zone decides it. offsets[k] is the Z distance from
    // the surface's layer to the layer k layers away, as make_layer_shell_levels() takes them, so the count is at most
    // offsets.size().
    std::optional<size_t> solid_layers(ShellSide side, const Vec3d &p, const std::vector<double> &offsets) const;

    // Whether any zone or its ramp may reach into the box. If not, no zone decides the shell anywhere in it.
    bool          may_affect(const BoundingBoxf3 &box) const;
    // A box around everything the zones and their ramps may reach.
    BoundingBoxf3 reach() const;
    // A rule holding wherever the rule of any active profile of the side holds: the largest of their layer counts and
    // of their thicknesses. No layers and no thickness if no profile ramps that side.
    ShellRule     deepest(ShellSide side) const;
    // Triangles of the remeshed surfaces of all zones, for diagnostics.
    size_t        triangle_count() const;

private:
    struct Impl;
    explicit ShellDepthField(std::unique_ptr<const Impl> impl);

    std::unique_ptr<const Impl> m_impl;
};

// The field thresholded over one layer, for the surfaces of one side.
struct ShellLevels
{
    // Where some zone decides the shell. Outside it the receiving region's own shell applies.
    Polygons              reach;
    // level[k] is where the shell under (over) a surface of this layer reaches the layer k layers below (above) it,
    // k = 0 being this layer itself: where the rule holds for k and the Z distance between the two. Each lies within
    // reach and within the one before it. Empty from level.size() on.
    std::vector<Polygons> level;

    bool            empty() const { return reach.empty(); }
    const Polygons &at(size_t k) const;
};

struct LayerShellLevels
{
    ShellLevels top;
    ShellLevels bottom;
};

// The levels of a layer within `extents` (scaled), sampled at height z. top_offsets[k] is the Z distance from this
// layer's top to the top of the layer k layers below, bottom_offsets[k] from its bottom to the bottom of the layer k
// layers above, each starting at 0 for k = 0: the dz of the stock projection. There are at most as many levels as
// offsets. The zone distances are sampled once, on a grid of 0.5 mm aligned to a fixed origin, and each level is a
// contour, by marching squares, of by how much the rule holds at those samples.
LayerShellLevels make_layer_shell_levels(const ShellDepthField &field, const BoundingBox &extents, double z, const std::vector<double> &top_offsets,
                                         const std::vector<double> &bottom_offsets);

// The levels of every layer of the object, in parallel. Empty for a layer no zone reaches.
std::vector<LayerShellLevels> make_shell_levels(const ShellDepthField &field, const PrintObject &object);

// A modifier, and the part it modifies.
using ModifierOnPart = std::pair<const ModelVolume*, const ModelVolume*>;

// The shell depth field of the object, fed by every modifier region with a top or bottom Shell Ramp. nullptr if there
// is none, or none changes a shell on the surface of a part. A modifier gets no zone on a part where its shell settings,
// or the part's, differ between the object's height ranges; that pair is added to `inconsistent` if given.
std::shared_ptr<const ShellDepthField> build_shell_depth_field(const PrintObject &object, std::vector<ModifierOnPart> *inconsistent = nullptr);

} // namespace Slic3r
