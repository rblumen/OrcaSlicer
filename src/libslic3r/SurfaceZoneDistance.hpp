#pragma once

// The signed geodesic distance, over the surface of a part, from the boundary of the zone a closed
// modifier mesh cuts out of it.
//
// A zone is the part of the part's surface that lies inside the modifier. A face the modifier
// shares with the part from inside belongs to the zone; one it rests on from outside does not. The
// distance is negative inside the zone, zero on its boundary and positive outside. It comes from the
// heat method on the part's surface, remeshed around the boundary with short edges there and longer
// ones where the distance is read, and is stored at the vertices of that mesh. It saturates at
// -inside_limit and at outside_limit, and is only resolved that far from the boundary. Where the
// solve does not reach, the straight line distance to the boundary stands in for it.
//
// The part and modifier meshes have to be closed and consistently oriented: whether a point is in
// the zone is decided by winding numbers. A zone narrower than about the fine edge may be missed.
//
// The distance knows nothing about what it is used for: the callers choose the edge sizes, tolerance
// and limits to suit it, and turn the distance into a wall thickness or a shell depth.

#include <cstddef>
#include <vector>

#include <Eigen/Geometry>
#include <admesh/stl.h>

#include "AABBTreeIndirect.hpp"
#include "Point.hpp"

namespace Slic3r {

// A triangle mesh with a tree over it, for distance and containment queries.
struct TreeMesh
{
    indexed_triangle_set     its;
    AABBTreeIndirect::Tree3f tree;

    void build_tree();

    // Distance to the nearest point of the mesh, which lies on `face` at `closest`.
    double distance(const Vec3d &p, size_t &face, Vec3d &closest) const;
    double distance(const Vec3d &p) const;

    // Whether p is enclosed by the mesh, which has to be closed and consistently oriented: the
    // winding number along a ray is not zero for at least two of three rays. Counting signed
    // crossings keeps overlapping shells inside, and the vote outweighs a ray that grazes a face or
    // passes through an edge.
    bool contains(const Vec3d &p) const;
};

struct SurfaceZoneDistanceParams
{
    // Longest mesh edge at the zone's boundary, in mm.
    double fine_edge { 0. };
    // Longest mesh edge where the distance is solved, in mm. Raised to fine_edge if shorter.
    double coarse_edge { 0. };
    // Growth of the edge length away from the boundary, mm per mm, from fine_edge up to coarse_edge.
    double edge_growth { 0.1 };
    // How far in mm a query point may lie off the part's surface and still be on it. A point further
    // off is out of the zone's reach and gets outside_limit.
    double surface_tolerance { 0.5 };
    // The distance saturates at -inside_limit inside the zone and at outside_limit outside it, both
    // in mm and not negative.
    double inside_limit { 0. };
    double outside_limit { 0. };
};

// The distance of one zone: the part's surface around the modifier, refined, with the signed
// geodesic distance from the zone's boundary at its vertices.
class SurfaceZoneDistance
{
public:
    // An inactive zone, which has no distance.
    SurfaceZoneDistance() = default;

    // The zone `modifier` cuts out of `part`. Inactive if the modifier does not reach the part's
    // surface, or the part has no triangles near it.
    static SurfaceZoneDistance build(const indexed_triangle_set &part, const indexed_triangle_set &modifier, const SurfaceZoneDistanceParams &params);

    // Whether the modifier reaches the part's surface.
    bool active() const { return m_active; }

    // Signed distance in mm from the zone's boundary to the nearest point of the part's surface to
    // p, saturated at -inside_limit and outside_limit; outside_limit for a point further than
    // surface_tolerance off the part. Infinite for an inactive zone.
    double signed_distance(const Vec3d &p) const;

    // Everything outside is further than outside_limit from the zone, or off its part's surface.
    const Eigen::AlignedBox3d &reach() const { return m_reach; }

    // Triangles of the remeshed surface, for diagnostics.
    size_t triangle_count() const { return m_surface.its.indices.size(); }

private:
    // The distance saturates at -m_inside_limit and at m_outside_limit.
    double m_inside_limit { 0. };
    double m_outside_limit { 0. };
    bool   m_active { false };

    TreeMesh            m_surface;
    std::vector<float>  m_distance;
    Eigen::AlignedBox3d m_reach;
    // A point further than this from the zone's part is not on it, and out of the zone's reach.
    double m_lookup_tolerance { 0. };
};

} // namespace Slic3r
