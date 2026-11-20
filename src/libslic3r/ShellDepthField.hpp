#pragma once

// Graded shell reinforcement: the shell depth field of an object.
//
// A volume modifier set to the Ramp top (bottom) shell count transition changes the depth of the top (bottom) shell
// where a part's surface lies inside it, the zone, from the part's depth D0 to the modifier's D1. Outside the zone the
// depth returns to D0 over the ramp, a stretch of the surface L_s long. The depth D is a function of the geodesic
// distance d along the part's surface from the zone's boundary, negative inside the zone:
//
//     D(d) = D1 - dD * clamp(d / L_s, 0, 1),    dD = D1 - D0,    L_s = |dD| * rho_s
//
// rho_s is the shell taper ratio, mm of ramp per mm of shell depth changed. Depths are in mm, see shell_depth(): a
// layer below a top surface is in the shell where the Z distance between the two is below D, which counts variable
// layer heights and a thicker first layer as they are. A zone deepens the shell when D1 > D0 and thins it when
// D1 < D0, down to no shell at D1 = 0; either way its ramp lies outside it.
//
// Each zone has a top and a bottom profile, which ramp independently: one distance solve serves both. Each carries its
// own D0, because one modifier can overlap several parts with different shells, and its piece on each part tapers to
// that part's shell: the caller passes one zone per piece, on that part's mesh.
//
// Where zones and ramps overlap, the deeper shell wins: if any zone gives a D above its own D0 at a point, the depth is
// the largest such D; otherwise, if any gives a D below its own D0, the smallest such D; otherwise no zone decides the
// depth and the shell is the receiving region's own, see PrintRegion::shell_default_counts().
//
// The field is built once per object and only queried afterwards, from any number of threads.

#include <cstddef>
#include <memory>
#include <optional>
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

// How a zone changes the shell on one side.
struct ShellRampProfile
{
    // D0, the depth of the shell the modifier modifies, in mm.
    double base_depth { 0. };
    // D1, the depth inside the zone, in mm.
    double depth { 0. };
    // rho_s, mm of ramp per mm of depth changed. 0 keeps the hard step.
    double taper_ratio { 0. };

    // Whether the profile changes the depth anywhere: the taper ratio is positive and D1 differs from D0.
    bool   active() const;
    // L_s in mm.
    double ramp_length() const;
    // D(d) in mm, see the top of the file. D0 for an inactive profile.
    double at(double distance) const;
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

// What the field says at one point, for both sides at once.
struct ShellSample
{
    // Positive where some zone decides the depth: the largest L_s - d over the zones, which crosses 0 where the last
    // ramp ends. Negative where none decides, -1 if no zone ramps that side at all.
    float top_reach { -1.f };
    // The depth where a zone decides it. Elsewhere the D0 of the zone closest to deciding it, so that the depth runs
    // on continuously past where the ramp ends.
    float top_depth { 0.f };
    float bottom_reach { -1.f };
    float bottom_depth { 0.f };
};

// Shell depth over the surface of one object, fed by all its Shell Ramp modifiers.
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
    // Signed geodesic distance in mm from the zone's boundary to the nearest point of its part's surface to p,
    // saturated at 0 inside the zone and at the end of its longer ramp plus outside_margin outside it, which is also
    // what a point further than surface_tolerance off the part gets. Infinite for an inactive zone.
    double signed_distance(size_t zone, const Vec3d &p) const;

    // The depth of the shell on the given side at the surface point nearest to p, combined over the zones as the top
    // of the file describes; none where no zone decides it.
    std::optional<double> depth(ShellSide side, const Vec3d &p) const;
    // Both sides at once, with one distance lookup per zone.
    ShellSample           sample(const Vec3d &p) const;

    // Whether any zone or its ramp may reach into the box. If not, no zone decides the depth anywhere in it.
    bool          may_affect(const BoundingBoxf3 &box) const;
    // A box around everything the zones and their ramps may reach.
    BoundingBoxf3 reach() const;
    // The deepest shell any active profile of the side asks for, max(D0, D1), in mm; 0 if none ramps that side.
    double        max_depth(ShellSide side) const;
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
    // Where some zone decides the depth. Outside it the receiving region's own shell applies.
    Polygons              reach;
    // level[k] is where the shell under (over) a surface of this layer reaches the layer k layers below (above) it,
    // k = 0 being this layer itself: where the depth exceeds the Z distance between the two. Each lies within reach
    // and within the one before it. Empty from level.size() on.
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
// layers above; each starts at 0 for k = 0. The field is sampled once, on a grid of 0.5 mm aligned to a fixed origin,
// and each level is a contour of those samples, by marching squares.
LayerShellLevels make_layer_shell_levels(const ShellDepthField &field, const BoundingBox &extents, double z, const std::vector<double> &top_offsets,
                                         const std::vector<double> &bottom_offsets);

// The levels of every layer of the object, in parallel. Empty for a layer no zone reaches.
std::vector<LayerShellLevels> make_shell_levels(const ShellDepthField &field, const PrintObject &object);

// The depth field of the object, fed by every modifier region with a top or bottom Shell Ramp. nullptr if there is
// none, or none changes a shell on the surface of a part. A modifier whose shell settings, or those of the region it
// modifies, differ between the object's height ranges gets no zone, and is added to `inconsistent_modifiers` if given.
std::shared_ptr<const ShellDepthField> build_shell_depth_field(const PrintObject &object, std::vector<const ModelVolume*> *inconsistent_modifiers = nullptr);

} // namespace Slic3r
