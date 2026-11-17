#include "WallThicknessField.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

#include <Eigen/Geometry>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <igl/Hit.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "AABBTreeIndirect.hpp"
#include "BoundingBox.hpp"
#include "KDTreeIndirect.hpp"
#include "Point.hpp"
#include "Triangulation.hpp"

namespace Slic3r {

double wall_ramp_thickness(double distance, int base_wall_loops, int zone_wall_loops, double taper_ratio, double bead_width)
{
    const int change = zone_wall_loops - base_wall_loops;
    if (change == 0 || taper_ratio <= 0.)
        return base_wall_loops * bead_width;
    return bead_width * (zone_wall_loops - change * std::clamp(distance / (std::abs(change) * taper_ratio * bead_width), 0., 1.));
}

namespace {

// The smallest magnitude of the level at a vertex, in mm: a vertex is never on the zone's boundary
// itself, which keeps the sources of the distance on the cut along the boundary.
constexpr double ON_SURFACE_TOLERANCE = 1e-4;
// A modifier triangle whose corners and middle all lie this close (mm) to the part's surface, and
// parallel to it, is a face the two share.
constexpr double SHARED_FACE_TOLERANCE = 1e-3;
// A surface point belongs to a zone if the material this far beneath it is inside the modifier.
// That decides the faces a modifier shares with the part: a modifier flush with the part from
// inside reinforces the face, one resting on the part from outside does not.
constexpr double INSIDE_PROBE_DEPTH = 1e-3;
// Further than this from the modifier's surface, a surface point is inside the modifier exactly
// when the material beneath it is, so it is tested directly.
constexpr double DIRECT_INSIDE_TEST = 10. * INSIDE_PROBE_DEPTH;
// A zone boundary crossing an edge closer than this fraction of its length to one of its ends is
// moved to that end, which keeps slivers out of the mesh.
constexpr double SNAP_FRACTION = 0.1;
// Time step of the heat flow, as a multiple of the square of the coarsest edge length the solved
// band is meshed with. Measured on the plate, the L bracket, a slot and the coupon: 0.5 shortens
// distances around a corner slightly but lengthens them next to a narrow side; 0.25 fails the 2%
// bound on the plate and the bracket.
constexpr double HEAT_TIME_FACTOR = 1.;
// A facet, which is triangulated afresh, gathers neighboring triangles this close to parallel to
// its first triangle (cosine of the angle of their normals) whose corners all lie this close (mm)
// to its plane.
constexpr double FACET_PLANAR_COS      = 1. - 1e-6;
constexpr double FACET_PLANE_TOLERANCE = 1e-4;
// A triangle smaller than this, as twice its area over the square of its longest edge, lies in no
// plane and is dropped.
constexpr double FACET_MIN_AREA = 1e-9;

struct TreeMesh
{
    indexed_triangle_set     its;
    AABBTreeIndirect::Tree3f tree;

    void build_tree() { tree = AABBTreeIndirect::build_aabb_tree_over_indexed_triangle_set(its.vertices, its.indices); }

    // Distance to the nearest point of the mesh, which lies on `face` at `closest`.
    double distance(const Vec3d &p, size_t &face, Vec3d &closest) const
    {
        return std::sqrt(AABBTreeIndirect::squared_distance_to_indexed_triangle_set(its.vertices, its.indices, tree, p, face, closest));
    }
    double distance(const Vec3d &p) const
    {
        size_t face;
        Vec3d  closest;
        return this->distance(p, face, closest);
    }

    // Whether p is enclosed by the mesh, which has to be closed and consistently oriented: the
    // winding number along a ray is not zero for at least two of three rays. Counting signed
    // crossings keeps overlapping shells inside, and the vote outweighs a ray that grazes a face or
    // passes through an edge.
    bool contains(const Vec3d &p) const
    {
        // Skewed, so that a ray from a vertex of an axis aligned part does not graze the faces or
        // pass through the edges of an axis aligned modifier.
        static const std::array<Vec3d, 3> directions { Vec3d(0.4217, 0.5713, 0.7041).normalized(), Vec3d(- 0.6302, 0.2459, 0.7364).normalized(),
                                                       Vec3d(0.1873, - 0.7139, - 0.6747).normalized() };
        std::vector<igl::Hit<float>> hits;
        int                          votes = 0;
        for (size_t ray = 0; ray < directions.size(); ++ ray) {
            // Two rays that agree settle the vote.
            if (ray == 2 && votes != 1)
                break;
            const Vec3d &dir = directions[ray];
            hits.clear();
            AABBTreeIndirect::intersect_ray_all_hits(its.vertices, its.indices, tree, p, dir, hits);
            int   winding   = 0;
            int   last_side = 0;
            float last_t    = 0.f;
            for (const igl::Hit<float> &hit : hits) {
                const stl_triangle_vertex_indices &face = its.indices[hit.id];
                const Vec3d a      = its.vertices[face[0]].cast<double>();
                const Vec3d normal = (its.vertices[face[1]].cast<double>() - a).cross(its.vertices[face[2]].cast<double>() - a);
                const int   side   = normal.dot(dir) > 0. ? 1 : -1;
                // A ray through an edge reports both its triangles.
                if (side == last_side && hit.t - last_t <= 1e-6f * (1.f + hit.t))
                    continue;
                winding  += side;
                last_side = side;
                last_t    = hit.t;
            }
            votes += winding != 0;
        }
        return votes >= 2;
    }
};

// The thickness field of one modifier: the part's surface around the modifier, refined, with the
// signed geodesic distance from the zone's boundary at its vertices.
struct ZoneField
{
    int    base_wall_loops { 0 };
    int    wall_loops { 0 };
    double taper_ratio { 0. };
    double ramp_length { 0. };
    // The distance saturates at -inside_limit and at outside_limit.
    double inside_limit { 0. };
    double outside_limit { 0. };
    bool   active { false };

    TreeMesh           surface;
    std::vector<float> distance;
    // Everything outside is further than outside_limit from the zone, or off its part's surface.
    Eigen::AlignedBox3d reach;
    // A point further than this from the zone's part is not on it, and out of the zone's reach.
    double lookup_tolerance { 0. };

    double signed_distance(const Vec3d &p) const
    {
        if (! active)
            return std::numeric_limits<double>::infinity();
        if (! reach.contains(p))
            return outside_limit;
        size_t face_idx;
        Vec3d  closest;
        if (surface.distance(p, face_idx, closest) > lookup_tolerance)
            return outside_limit;
        const stl_triangle_vertex_indices &face = surface.its.indices[face_idx];
        const Vec3d a  = surface.its.vertices[face[0]].cast<double>();
        const Vec3d ab = surface.its.vertices[face[1]].cast<double>() - a;
        const Vec3d ac = surface.its.vertices[face[2]].cast<double>() - a;
        const Vec3d ap = closest - a;
        // Barycentric coordinates of the closest point.
        const double d00 = ab.squaredNorm(), d01 = ab.dot(ac), d11 = ac.squaredNorm();
        const double d20 = ap.dot(ab), d21 = ap.dot(ac);
        const double denom = d00 * d11 - d01 * d01;
        double v = 1. / 3., w = 1. / 3.;
        if (denom > 0.) {
            v = std::clamp((d11 * d20 - d01 * d21) / denom, 0., 1.);
            w = std::clamp((d00 * d21 - d01 * d20) / denom, 0., 1. - v);
        }
        const double d = (1. - v - w) * distance[face[0]] + v * distance[face[1]] + w * distance[face[2]];
        return std::clamp(d, - inside_limit, outside_limit);
    }
};

uint64_t edge_key(int a, int b)
{
    if (a > b)
        std::swap(a, b);
    return (uint64_t(uint32_t(a)) << 32) | uint64_t(uint32_t(b));
}

// Half edge f * 3 + i of a triangle mesh runs from corner i of face f to the next corner.
inline int next_half_edge(int he) { return he % 3 == 2 ? he - 2 : he + 1; }
inline int prev_half_edge(int he) { return he % 3 == 0 ? he + 2 : he - 1; }

// The half edge on the other side of each half edge, or -1 on an open edge. An edge shared by more
// than two faces, or by two faces oriented apart, is treated as open on all sides.
std::vector<int> opposite_half_edges(const std::vector<Vec3i32> &faces)
{
    std::vector<int> opposite(faces.size() * 3, -1);
    std::unordered_map<uint64_t, int> first;
    first.reserve(faces.size() * 2);
    std::vector<char> shared_badly(faces.size() * 3, 0);
    for (int f = 0; f < int(faces.size()); ++ f)
        for (int i = 0; i < 3; ++ i) {
            const int he = f * 3 + i;
            const int a = faces[f][i], b = faces[f][(i + 1) % 3];
            auto [it, inserted] = first.try_emplace(edge_key(a, b), he);
            if (inserted)
                continue;
            const int other = it->second;
            const int oa = faces[other / 3][other % 3], ob = faces[other / 3][(other % 3 + 1) % 3];
            if (shared_badly[other] || opposite[other] != -1 || oa != b || ob != a) {
                if (opposite[other] >= 0)
                    shared_badly[opposite[other]] = 1;
                shared_badly[other] = shared_badly[he] = 1;
            } else {
                opposite[he]    = other;
                opposite[other] = he;
            }
        }
    for (size_t he = 0; he < opposite.size(); ++ he)
        if (shared_badly[he])
            opposite[he] = -1;
    return opposite;
}

// Graded remeshing of a triangle mesh for the heat method, which needs short edges near the zone's
// boundary and few triangles elsewhere. The flat faces of an STL are often fans of long slivers,
// which subdivision keeps: splitting them conformingly multiplies the triangle count without
// improving them. So the mesh is cut into facets, the connected sets of coplanar triangles, and each
// facet is triangulated afresh in its own plane: a constrained Delaunay triangulation of its outline,
// resampled, and of points spread over its inside, both as densely as the sizing function asks.
// Each edge between two facets is resampled once, so the facets still meet vertex to vertex.
//
// min_size_near(p, r) bounds from below the longest edge wanted anywhere within r of p. It is
// called from several threads at once.
template<typename SizeNearFn>
void remesh_graded(std::vector<Vec3d> &vertices, std::vector<Vec3i32> &faces, const SizeNearFn &min_size_near)
{
    const std::vector<int> opposite  = opposite_half_edges(faces);
    const size_t           num_faces = faces.size();

    // Facets: grown from a seed triangle over its neighbors while every corner of a neighbor lies
    // within FACET_PLANE_TOLERANCE of the seed's plane, so that a finely tessellated curved surface
    // does not merge into one facet through a chain of nearly coplanar neighbors. A degenerate face
    // lies in no plane and is dropped.
    std::vector<Vec3d> normals(num_faces);
    std::vector<char>  degenerate(num_faces, 0);
    for (size_t f = 0; f < num_faces; ++ f) {
        const Vec3d &a = vertices[faces[f][0]], &b = vertices[faces[f][1]], &c = vertices[faces[f][2]];
        const Vec3d  normal      = (b - a).cross(c - a);
        const double longest_sqr = std::max({ (b - a).squaredNorm(), (c - b).squaredNorm(), (a - c).squaredNorm() });
        degenerate[f] = ! (normal.norm() > FACET_MIN_AREA * longest_sqr);
        normals[f]    = degenerate[f] ? Vec3d::Zero() : Vec3d(normal.normalized());
    }
    // Read only once built, also by the threads below.
    std::vector<int> facet_of(num_faces, -1);
    {
        std::vector<int> queue;
        for (size_t seed = 0; seed < num_faces; ++ seed) {
            if (degenerate[seed] || facet_of[seed] != -1)
                continue;
            const Vec3d &normal = normals[seed];
            const double offset = normal.dot(vertices[faces[seed][0]]);
            facet_of[seed]      = int(seed);
            queue.assign(1, int(seed));
            while (! queue.empty()) {
                const int f = queue.back();
                queue.pop_back();
                for (int i = 0; i < 3; ++ i) {
                    const int o = opposite[f * 3 + i];
                    if (o < 0)
                        continue;
                    const int g = o / 3;
                    if (degenerate[g] || facet_of[g] != -1 || normals[g].dot(normal) < FACET_PLANAR_COS)
                        continue;
                    bool in_plane = true;
                    for (int k = 0; k < 3; ++ k)
                        in_plane &= std::abs(normal.dot(vertices[faces[g][k]]) - offset) <= FACET_PLANE_TOLERANCE;
                    if (in_plane) {
                        facet_of[g] = int(seed);
                        queue.emplace_back(g);
                    }
                }
            }
        }
    }
    auto on_boundary = [&](int he) {
        const int o = opposite[he];
        return o < 0 || degenerate[o / 3] || facet_of[o / 3] != facet_of[he / 3];
    };

    // Resample the edges of the facets' outlines. Each edge is split in halves while a piece is
    // longer than the size near it, and its pieces are listed from its lower vertex to its higher.
    std::unordered_map<uint64_t, std::vector<int>> outline_edges;
    std::function<void(int, int, std::vector<int> &)> resample = [&](int a, int b, std::vector<int> &out) {
        const double length = (vertices[b] - vertices[a]).norm();
        const Vec3d  middle = 0.5 * (vertices[a] + vertices[b]);
        // Written so that a length or size that is not a number ends the recursion too.
        if (! (length > min_size_near(middle, 0.5 * length))) {
            out.emplace_back(b);
            return;
        }
        const int m = int(vertices.size());
        vertices.emplace_back(middle);
        resample(a, m, out);
        resample(m, b, out);
    };
    for (size_t f = 0; f < num_faces; ++ f)
        if (! degenerate[f])
            for (int i = 0; i < 3; ++ i)
                if (on_boundary(int(f * 3 + i))) {
                    int a = faces[f][i], b = faces[f][(i + 1) % 3];
                    if (a > b)
                        std::swap(a, b);
                    auto [it, inserted] = outline_edges.try_emplace(edge_key(a, b));
                    if (inserted) {
                        it->second.emplace_back(a);
                        resample(a, b, it->second);
                    }
                }

    // The faces of each facet.
    std::unordered_map<int, std::vector<int>> facet_faces;
    for (size_t f = 0; f < num_faces; ++ f)
        if (! degenerate[f])
            facet_faces[facet_of[f]].emplace_back(int(f));
    std::vector<int> roots;
    roots.reserve(facet_faces.size());
    for (const auto &[root, members] : facet_faces)
        roots.emplace_back(root);
    std::sort(roots.begin(), roots.end());

    // Triangulate the facets in parallel. A facet only reads the shared vertices; the points it adds
    // inside, and its triangles, are collected and merged afterwards. A triangle's index refers to
    // an existing vertex if it is not negative, and to the facet's own new point -1 - index otherwise.
    struct FacetMesh
    {
        std::vector<Vec3d>   new_points;
        std::vector<Vec3i32> triangles;
    };
    std::vector<std::vector<int>> facet_members(roots.size());
    for (size_t i = 0; i < roots.size(); ++ i)
        facet_members[i] = std::move(facet_faces[roots[i]]);
    std::vector<FacetMesh> facet_meshes(roots.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, roots.size(), 1), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t facet_idx = range.begin(); facet_idx < range.end(); ++ facet_idx) {
            const std::vector<int> &members = facet_members[facet_idx];
            FacetMesh              &out     = facet_meshes[facet_idx];
            auto keep_original = [&]() {
                out.new_points.clear();
                out.triangles.clear();
                for (int f : members)
                    out.triangles.emplace_back(faces[f]);
            };
            // A lone triangle no larger than asked, as a finely tessellated surface is made of,
            // stays as it is: its edges were not resampled, and nothing would go inside.
            if (members.size() == 1) {
                bool small = true;
                for (int i = 0; i < 3; ++ i)
                    if (on_boundary(members.front() * 3 + i))
                        small &= outline_edges.at(edge_key(faces[members.front()][i], faces[members.front()][(i + 1) % 3])).size() == 2;
                if (small) {
                    keep_original();
                    continue;
                }
            }
            // A frame of the facet's plane.
            const Vec3d origin = vertices[faces[members.front()][0]];
            const Vec3d normal = normals[members.front()];
            const Vec3d u      = (vertices[faces[members.front()][1]] - origin).normalized();
            const Vec3d v      = normal.cross(u);
            auto to_plane = [&](const Vec3d &p) { return Vec2d((p - origin).dot(u), (p - origin).dot(v)); };

            // Outline: the resampled boundary half edges, interior on their left.
            std::unordered_map<int, uint32_t> local;
            std::vector<int>                  global;
            Points                            points;
            std::vector<Vec2d>                points_2d;
            auto add_point = [&](int vertex, const Vec2d &position) {
                auto [it, inserted] = local.try_emplace(vertex, uint32_t(global.size()));
                if (inserted) {
                    global.emplace_back(vertex);
                    points_2d.emplace_back(position);
                    points.emplace_back(Point::new_scale(position.x(), position.y()));
                }
                return it->second;
            };
            Triangulation::HalfEdges constraints;
            for (int f : members)
                for (int i = 0; i < 3; ++ i)
                    if (on_boundary(f * 3 + i)) {
                        const int a = faces[f][i], b = faces[f][(i + 1) % 3];
                        const std::vector<int> &samples = outline_edges.at(edge_key(a, b));
                        for (size_t k = 0; k + 1 < samples.size(); ++ k) {
                            const int from = a < b ? samples[k] : samples[samples.size() - 1 - k];
                            const int to   = a < b ? samples[k + 1] : samples[samples.size() - 2 - k];
                            constraints.emplace_back(add_point(from, to_plane(vertices[from])), add_point(to, to_plane(vertices[to])));
                        }
                    }
            const size_t num_outline = global.size();
            if (constraints.empty()) {
                keep_original();
                continue;
            }

            // Inside: a quadtree over the facet, split while a cell is larger than the size near it,
            // with a point at the middle of each leaf inside the facet and clear of the outline.
            std::vector<std::array<Vec2d, 3>> triangles;
            std::vector<std::array<bool, 3>>  outline_sides;
            std::vector<Eigen::AlignedBox2d>  triangle_boxes;
            Eigen::AlignedBox2d               box;
            for (int f : members) {
                std::array<Vec2d, 3> t;
                std::array<bool, 3>  on_outline;
                Eigen::AlignedBox2d  triangle_box;
                for (int i = 0; i < 3; ++ i) {
                    t[i]          = to_plane(vertices[faces[f][i]]);
                    on_outline[i] = on_boundary(f * 3 + i);
                    triangle_box.extend(t[i]);
                }
                box.extend(triangle_box);
                triangles.emplace_back(t);
                outline_sides.emplace_back(on_outline);
                triangle_boxes.emplace_back(triangle_box);
            }
            // On an edge between two of the facet's triangles counts as inside, on its outline not.
            auto inside = [&](const Vec2d &p, const std::vector<int> &candidates) {
                for (int idx : candidates) {
                    const std::array<Vec2d, 3> &t      = triangles[idx];
                    bool                        within = true;
                    for (int i = 0; i < 3 && within; ++ i) {
                        const Vec2d &a = t[i], &b = t[(i + 1) % 3];
                        const double side = (b - a).x() * (p - a).y() - (b - a).y() * (p - a).x();
                        within = side > 0. || (side == 0. && ! outline_sides[idx][i]);
                    }
                    if (within)
                        return true;
                }
                return false;
            };
            auto outline_coordinate = [&points_2d](size_t idx, size_t dimension) { return points_2d[idx][dimension]; };
            const KDTreeIndirect<2, double, decltype(outline_coordinate)> outline_tree(outline_coordinate, num_outline);
            // Each cell carries the facet's triangles that overlap it, so that a cell off the facet
            // ends at once, and a point is tested against a few triangles only.
            std::function<void(const Vec2d &, double, const std::vector<int> &)> fill = [&](const Vec2d &center, double half,
                                                                                           const std::vector<int> &overlapping) {
                const Eigen::AlignedBox2d cell(center - Vec2d(half, half), center + Vec2d(half, half));
                std::vector<int>          candidates;
                for (int idx : overlapping)
                    if (triangle_boxes[idx].intersects(cell))
                        candidates.emplace_back(idx);
                if (candidates.empty())
                    return;
                const Vec3d p = origin + center.x() * u + center.y() * v;
                if (2. * half > min_size_near(p, half * 1.4142136)) {
                    for (double dx : { -0.5, 0.5 })
                        for (double dy : { -0.5, 0.5 })
                            fill(center + Vec2d(dx * half, dy * half), 0.5 * half, candidates);
                    return;
                }
                if (! inside(center, candidates))
                    return;
                // Clear of the outline by most of a cell.
                const size_t nearest = find_closest_point(outline_tree, center);
                if (nearest < num_outline && (points_2d[nearest] - center).norm() < 1.4 * half)
                    return;
                out.new_points.emplace_back(p);
                add_point(- int(out.new_points.size()), center);
            };
            if (const double size = box.sizes().maxCoeff(); size > 0.) {
                std::vector<int> all(triangles.size());
                std::iota(all.begin(), all.end(), 0);
                fill(box.center(), 0.5 * size, all);
            }

            // The triangulation only asserts its preconditions. Two points that round to one would
            // be merged into one vertex and break the triangles' indices, so they are checked here,
            // as are the indices that come back; the facet keeps its own triangles if either fails.
            Points sorted = points;
            std::sort(sorted.begin(), sorted.end(), [](const Point &a, const Point &b) { return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y()); });
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
                keep_original();
                continue;
            }
            std::sort(constraints.begin(), constraints.end());
            const Triangulation::Indices triangulated = Triangulation::triangulate(points, constraints);
            const bool valid = ! triangulated.empty() && std::all_of(triangulated.begin(), triangulated.end(), [&points](const Vec3i32 &t) {
                return t.minCoeff() >= 0 && size_t(t.maxCoeff()) < points.size();
            });
            if (! valid) {
                keep_original();
                continue;
            }
            for (const Vec3i32 &t : triangulated)
                out.triangles.emplace_back(global[t[0]], global[t[1]], global[t[2]]);
        }
    });

    std::vector<Vec3i32> out_faces;
    out_faces.reserve(num_faces * 4);
    for (FacetMesh &facet_mesh : facet_meshes) {
        const int first_new = int(vertices.size());
        vertices.insert(vertices.end(), facet_mesh.new_points.begin(), facet_mesh.new_points.end());
        for (const Vec3i32 &t : facet_mesh.triangles) {
            Vec3i32 face;
            for (int i = 0; i < 3; ++ i)
                face[i] = t[i] >= 0 ? t[i] : first_new - 1 - t[i];
            out_faces.emplace_back(face);
        }
    }
    faces = std::move(out_faces);
}

// Area of a triangle from the lengths of its sides (Heron's formula in Kahan's stable form), zero
// if they violate the triangle inequality.
double triangle_area(double a, double b, double c)
{
    if (a < b)
        std::swap(a, b);
    if (a < c)
        std::swap(a, c);
    if (b < c)
        std::swap(b, c);
    const double product = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c));
    return product > 0. ? 0.25 * std::sqrt(product) : 0.;
}

// The intrinsic Delaunay triangulation of a triangle mesh (Bobenko and Springborn 2007): edges are
// flipped while the angles facing them sum to more than pi, but a flipped edge is a geodesic over
// the two triangles it crosses, known by its length only. The surface stays exactly what it was,
// so creases and flat faces need no special care, and the cotangent Laplacian of the result has
// no negative weights however poor the triangles of the mesh are (Sharp, Soliman, Crane 2019).
struct IntrinsicTriangulation
{
    std::vector<Vec3i32> faces;
    // Length of each half edge.
    std::vector<double> length;

    IntrinsicTriangulation(const std::vector<Vec3d> &vertices, std::vector<Vec3i32> faces_in)
        : faces(std::move(faces_in)), length(faces.size() * 3), m_opposite(opposite_half_edges(faces))
    {
        for (size_t f = 0; f < faces.size(); ++ f)
            for (int i = 0; i < 3; ++ i)
                length[f * 3 + i] = (vertices[faces[f][(i + 1) % 3]] - vertices[faces[f][i]]).norm();
        std::vector<int> stack(m_opposite.size());
        std::iota(stack.begin(), stack.end(), 0);
        // Each flip lowers the Dirichlet energy, so this ends; the bound only guards against
        // rounding making two configurations look worse than each other.
        size_t flips_left = 20 * faces.size() + 1000;
        while (! stack.empty() && flips_left > 0) {
            const int he = stack.back();
            stack.pop_back();
            if (this->flip_if_not_delaunay(he, stack))
                -- flips_left;
        }
    }

private:
    std::vector<int> m_opposite;

    void link(int he, int opposite, double edge_length)
    {
        m_opposite[he] = opposite;
        length[he]     = edge_length;
        if (opposite >= 0) {
            m_opposite[opposite] = he;
            length[opposite]     = edge_length;
        }
    }

    bool flip_if_not_delaunay(int he, std::vector<int> &stack)
    {
        const int other = m_opposite[he];
        if (other < 0)
            return false;
        const int f = he / 3, g = other / 3;
        const int a = faces[f][he % 3], b = faces[f][(he % 3 + 1) % 3], c = faces[f][(he % 3 + 2) % 3], d = faces[g][(other % 3 + 2) % 3];
        if (c == d)
            return false;
        const double ab = length[he], bc = length[next_half_edge(he)], ca = length[prev_half_edge(he)];
        const double ad = length[next_half_edge(other)], db = length[prev_half_edge(other)];
        const double area_f = triangle_area(ab, bc, ca), area_g = triangle_area(ab, ad, db);
        if (! (area_f > 0.) || ! (area_g > 0.))
            return false;
        // Cotangents of the angles at c and d.
        const double cot_c = (bc * bc + ca * ca - ab * ab) / (4. * area_f);
        const double cot_d = (ad * ad + db * db - ab * ab) / (4. * area_g);
        if (cot_c + cot_d >= - 1e-12)
            return false;
        // Lay the two triangles out in the plane on either side of the edge to measure the new one.
        const double cx = (ab * ab + ca * ca - bc * bc) / (2. * ab), cy = 2. * area_f / ab;
        const double dx = (ab * ab + ad * ad - db * db) / (2. * ab), dy = - 2. * area_g / ab;
        const double cd = std::hypot(cx - dx, cy - dy);
        if (! (triangle_area(ca, ad, cd) > 0.) || ! (triangle_area(db, bc, cd) > 0.))
            return false;
        const int o_bc = m_opposite[next_half_edge(he)], o_ca = m_opposite[prev_half_edge(he)];
        const int o_ad = m_opposite[next_half_edge(other)], o_db = m_opposite[prev_half_edge(other)];
        faces[f] = Vec3i32(c, a, d);
        faces[g] = Vec3i32(d, b, c);
        link(f * 3, o_ca, ca);
        link(f * 3 + 1, o_ad, ad);
        link(g * 3, o_db, db);
        link(g * 3 + 1, o_bc, bc);
        link(f * 3 + 2, g * 3 + 2, cd);
        stack.insert(stack.end(), { f * 3, f * 3 + 1, g * 3, g * 3 + 1 });
        return true;
    }
};

// Geodesic distance from the source vertices to all vertices of a triangle mesh by the heat
// method (Crane, Weischedel, Wardetzky 2013): let heat flow from the sources for a short time,
// normalize its gradient, and find the function whose gradient that is. It works on the intrinsic
// Delaunay triangulation of the mesh, each triangle laid out in its own plane.
//
// The sources are the vertices of a curve, so both steps hold them fixed, the heat at 1 and the
// distance at 0. The distance is then exactly zero on the zone's boundary, and the Poisson problem
// is positive definite without the extra constraint the free-floating formulation needs.
//
// libigl's heat_geodesics, which can also use the intrinsic Delaunay triangulation, is not used:
// it fixes the distance only up to a constant, shifting it so that the sources average zero rather
// than holding each at zero, and it needs an edge-manifold mesh, which a part's surface cut out
// around a zone, with overlapping shells or open edges, need not be.
//
// Vertices not connected to a source get infinity. Returns an empty vector if a factorization fails.
std::vector<double> heat_distances(const std::vector<Vec3d> &vertices, const std::vector<Vec3i32> &mesh_faces, const std::vector<char> &is_source,
                                   double time)
{
    const IntrinsicTriangulation triangulation(vertices, mesh_faces);
    const std::vector<Vec3i32>  &faces        = triangulation.faces;
    const size_t                 num_vertices = vertices.size();
    const size_t                 num_faces    = faces.size();
    const double                 infinity     = std::numeric_limits<double>::infinity();

    // Per face, the gradients of its three hat functions scaled by twice its area, in a frame of
    // the face's own plane, and that area. A degenerate face has a zero area and is skipped throughout.
    std::vector<std::array<Vec2d, 3>> hat_gradients(num_faces);
    std::vector<double>               double_areas(num_faces, 0.);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, num_faces), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t f = range.begin(); f < range.end(); ++ f) {
            const double l01 = triangulation.length[f * 3], l12 = triangulation.length[f * 3 + 1], l20 = triangulation.length[f * 3 + 2];
            const double double_area = 2. * triangle_area(l01, l12, l20);
            if (double_area <= 1e-9 * std::max({ l01 * l01, l12 * l12, l20 * l20 }))
                continue;
            const Vec2d a(0., 0.), b(l01, 0.), c((l01 * l01 + l20 * l20 - l12 * l12) / (2. * l01), double_area / l01);
            auto rotate = [](const Vec2d &v) { return Vec2d(- v.y(), v.x()); };
            hat_gradients[f] = { rotate(c - b), rotate(a - c), rotate(b - a) };
            double_areas[f]  = double_area;
        }
    });

    // Connected components over the valid faces, to leave out those without a source.
    std::vector<int> parent(num_vertices);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&parent](int v) {
        while (parent[v] != v)
            v = parent[v] = parent[parent[v]];
        return v;
    };
    for (size_t f = 0; f < num_faces; ++ f)
        if (double_areas[f] > 0.) {
            const int root = find(faces[f][0]);
            parent[find(faces[f][1])] = root;
            parent[find(faces[f][2])] = root;
        }
    std::vector<char> has_source(num_vertices, 0);
    for (size_t v = 0; v < num_vertices; ++ v)
        if (is_source[v])
            has_source[find(int(v))] = 1;

    // Unknowns: the vertices connected to a source.
    std::vector<int> unknown(num_vertices, -1);
    int              num_unknowns = 0;
    for (size_t f = 0; f < num_faces; ++ f)
        if (double_areas[f] > 0.)
            for (int i = 0; i < 3; ++ i) {
                const int v = faces[f][i];
                if (unknown[v] == -1 && ! is_source[v] && has_source[find(v)])
                    unknown[v] = num_unknowns ++;
            }
    std::vector<double> result(num_vertices, infinity);
    for (size_t v = 0; v < num_vertices; ++ v)
        if (is_source[v])
            result[v] = 0.;
    if (num_unknowns == 0)
        return result;

    // Stiffness matrix K (lower triangle) and lumped mass matrix M over the unknowns, and the
    // right hand side of the heat step (M + t K) u = 0 with u = 1 at the sources.
    using Triplet = Eigen::Triplet<double>;
    std::vector<Triplet> stiffness_entries;
    stiffness_entries.reserve(num_faces * 6);
    Eigen::VectorXd mass     = Eigen::VectorXd::Zero(num_unknowns);
    Eigen::VectorXd heat_rhs = Eigen::VectorXd::Zero(num_unknowns);
    for (size_t f = 0; f < num_faces; ++ f) {
        if (double_areas[f] == 0.)
            continue;
        const double scale = 0.5 / double_areas[f];
        for (int i = 0; i < 3; ++ i) {
            const int row = unknown[faces[f][i]];
            if (row == -1)
                continue;
            mass[row] += double_areas[f] / 6.;
            stiffness_entries.emplace_back(row, row, scale * hat_gradients[f][i].squaredNorm());
            for (int j = 0; j < 3; ++ j) {
                if (j == i)
                    continue;
                const double entry = scale * hat_gradients[f][i].dot(hat_gradients[f][j]);
                const int    col   = unknown[faces[f][j]];
                if (col == -1) {
                    if (is_source[faces[f][j]])
                        heat_rhs[row] -= time * entry;
                } else if (col < row)
                    stiffness_entries.emplace_back(row, col, entry);
            }
        }
    }
    Eigen::SparseMatrix<double> stiffness(num_unknowns, num_unknowns);
    stiffness.setFromTriplets(stiffness_entries.begin(), stiffness_entries.end());
    Eigen::SparseMatrix<double> heat_matrix = time * stiffness;
    for (int i = 0; i < num_unknowns; ++ i)
        heat_matrix.coeffRef(i, i) += mass[i];

    // Both matrices share one sparsity pattern, so one symbolic analysis serves both.
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
    solver.analyzePattern(stiffness);
    solver.factorize(heat_matrix);
    if (solver.info() != Eigen::Success)
        return {};
    const Eigen::VectorXd heat = solver.solve(heat_rhs);

    // Right hand side of the Poisson step K phi = integral of X . grad(hat), with X the unit
    // vector field pointing away from the sources, down the heat gradient.
    Eigen::VectorXd poisson_rhs = Eigen::VectorXd::Zero(num_unknowns);
    for (size_t f = 0; f < num_faces; ++ f) {
        if (double_areas[f] == 0.)
            continue;
        double values[3];
        double largest = 0.;
        bool   solved  = true;
        for (int i = 0; i < 3; ++ i) {
            const int v = faces[f][i];
            if (is_source[v])
                values[i] = 1.;
            else if (unknown[v] != -1)
                values[i] = heat[unknown[v]];
            else
                solved = false;
            if (solved)
                largest = std::max(largest, std::abs(values[i]));
        }
        // Far from the sources the heat is tiny, down to denormals: scale it before taking the norm.
        if (! solved || ! (largest > 0.))
            continue;
        Vec2d gradient = Vec2d::Zero();
        for (int i = 0; i < 3; ++ i)
            gradient += (values[i] / largest) * hat_gradients[f][i];
        const double norm = gradient.norm();
        if (! (norm > 0.))
            continue;
        for (int i = 0; i < 3; ++ i)
            if (const int row = unknown[faces[f][i]]; row != -1)
                poisson_rhs[row] -= 0.5 * gradient.dot(hat_gradients[f][i]) / norm;
    }
    solver.factorize(stiffness);
    if (solver.info() != Eigen::Success)
        return {};
    const Eigen::VectorXd phi = solver.solve(poisson_rhs);
    for (size_t v = 0; v < num_vertices; ++ v)
        if (unknown[v] != -1)
            result[v] = std::max(0., phi[unknown[v]]);
    return result;
}

// Split the triangles that the zero level of `level` crosses, so that it runs along edges. `level`
// is linear along an edge; the new vertices get zero.
void cut_at_zero_level(std::vector<Vec3d> &vertices, std::vector<double> &level, std::vector<Vec3i32> &faces)
{
    auto crosses = [&level](int a, int b) { return (level[a] < 0. && level[b] > 0.) || (level[a] > 0. && level[b] < 0.); };

    std::vector<char> snapped(vertices.size(), 0);
    bool              any_crossing = false;
    for (const Vec3i32 &face : faces)
        for (int i = 0; i < 3; ++ i) {
            const int a = face[i], b = face[(i + 1) % 3];
            if (crosses(a, b)) {
                any_crossing = true;
                const double fraction = level[a] / (level[a] - level[b]);
                if (fraction < SNAP_FRACTION)
                    snapped[a] = 1;
                else if (fraction > 1. - SNAP_FRACTION)
                    snapped[b] = 1;
            }
        }
    if (! any_crossing)
        return;
    for (size_t v = 0; v < snapped.size(); ++ v)
        if (snapped[v])
            level[v] = 0.;

    std::unordered_map<uint64_t, int> cut_vertices;
    auto cut = [&](int a, int b) {
        if (a > b)
            std::swap(a, b);
        auto [it, inserted] = cut_vertices.try_emplace(edge_key(a, b), int(vertices.size()));
        if (inserted) {
            const double fraction = level[a] / (level[a] - level[b]);
            // Evaluated before emplace_back(), which may move the vertices the expression reads.
            const Vec3d  crossing = vertices[a] + fraction * (vertices[b] - vertices[a]);
            vertices.emplace_back(crossing);
            level.emplace_back(0.);
        }
        return it->second;
    };

    std::vector<Vec3i32> out;
    out.reserve(faces.size() + faces.size() / 8);
    for (const Vec3i32 &face : faces) {
        const bool crossed[3] = { crosses(face[0], face[1]), crosses(face[1], face[2]), crosses(face[2], face[0]) };
        const int  count      = int(crossed[0]) + int(crossed[1]) + int(crossed[2]);
        if (count == 0) {
            out.emplace_back(face);
        } else if (count == 1) {
            // The level crosses the edge opposite the vertex it passes through.
            const int i = crossed[0] ? 2 : crossed[1] ? 0 : 1;
            const int a = face[i], b = face[(i + 1) % 3], c = face[(i + 2) % 3];
            const int m = cut(b, c);
            out.emplace_back(a, b, m);
            out.emplace_back(a, m, c);
        } else {
            // One corner is cut off; the remaining quadrilateral splits along its shorter diagonal.
            const int i = ! crossed[1] ? 0 : ! crossed[2] ? 1 : 2;
            const int a = face[i], b = face[(i + 1) % 3], c = face[(i + 2) % 3];
            const int m_ab = cut(a, b), m_ca = cut(c, a);
            out.emplace_back(a, m_ab, m_ca);
            if ((vertices[m_ab] - vertices[c]).squaredNorm() <= (vertices[b] - vertices[m_ca]).squaredNorm()) {
                out.emplace_back(m_ab, b, c);
                out.emplace_back(m_ab, c, m_ca);
            } else {
                out.emplace_back(m_ab, b, m_ca);
                out.emplace_back(b, c, m_ca);
            }
        }
    }
    faces = std::move(out);
}

Eigen::AlignedBox3d bounding_box_of(const std::vector<stl_vertex> &vertices)
{
    Eigen::AlignedBox3d box;
    for (const stl_vertex &v : vertices)
        box.extend(v.cast<double>());
    return box;
}

void build_zone(ZoneField &zone, const indexed_triangle_set &part, const indexed_triangle_set &modifier_mesh, const WallThicknessFieldParams &params)
{
    const double bead_width  = params.bead_width;
    const double fine_edge   = params.max_edge_fraction * bead_width;
    const double coarse_edge = std::max(fine_edge, params.coarse_edge_fraction * bead_width);
    const double tolerance   = std::max(0., params.surface_tolerance);
    // The distance is solved this far past the limits it saturates at, which keeps what the
    // open edge of the solved band does to the solution out of the range that is read.
    const double margin      = tolerance + 2. * coarse_edge;
    const double infinity    = std::numeric_limits<double>::infinity();

    TreeMesh modifier;
    modifier.its = modifier_mesh;
    modifier.build_tree();
    if (modifier.tree.empty() || part.indices.empty())
        return;
    const Eigen::AlignedBox3d modifier_box = bounding_box_of(modifier.its.vertices);
    if (! modifier_box.intersects(bounding_box_of(part.vertices)))
        return;
    zone.lookup_tolerance = tolerance;
    zone.reach            = modifier_box;
    zone.reach.extend(modifier_box.min() - Vec3d::Constant(zone.outside_limit + tolerance));
    zone.reach.extend(modifier_box.max() + Vec3d::Constant(zone.outside_limit + tolerance));

    // The part's triangles within the zone's reach.
    std::vector<Vec3d>   vertices;
    std::vector<Vec3i32> faces;
    {
        std::vector<int> index(part.vertices.size(), -1);
        for (const stl_triangle_vertex_indices &face : part.indices) {
            Eigen::AlignedBox3d box;
            for (int i = 0; i < 3; ++ i)
                box.extend(part.vertices[face[i]].cast<double>());
            if (! box.intersects(zone.reach) || face[0] == face[1] || face[1] == face[2] || face[2] == face[0])
                continue;
            Vec3i32 out;
            for (int i = 0; i < 3; ++ i) {
                int &idx = index[face[i]];
                if (idx == -1) {
                    idx = int(vertices.size());
                    vertices.emplace_back(part.vertices[face[i]].cast<double>());
                }
                out[i] = idx;
            }
            faces.emplace_back(out);
        }
    }
    if (faces.empty())
        return;

    // The zone's boundary is where the part's surface passes through the modifier's, or where a face
    // the two share ends. It lies on the modifier's triangles that do not lie on the part's surface,
    // so the distance to those bounds the distance to the boundary from below, also across a shared
    // face, where the distance to the whole modifier is zero.
    TreeMesh boundary_surface;
    {
        TreeMesh part_surface;
        for (const Vec3d &v : vertices)
            part_surface.its.vertices.emplace_back(v.cast<float>());
        for (const Vec3i32 &f : faces)
            part_surface.its.indices.emplace_back(f);
        part_surface.build_tree();
        for (const stl_triangle_vertex_indices &face : modifier.its.indices) {
            const Vec3d a = modifier.its.vertices[face[0]].cast<double>(), b = modifier.its.vertices[face[1]].cast<double>(),
                        c = modifier.its.vertices[face[2]].cast<double>();
            const Vec3d normal = (b - a).cross(c - a);
            bool on_part = normal.norm() > 0.;
            for (const Vec3d &corner : { a, b, c, Vec3d((a + b + c) / 3.) }) {
                if (! on_part)
                    break;
                size_t part_face;
                Vec3d  closest;
                if (part_surface.distance(corner, part_face, closest) > SHARED_FACE_TOLERANCE) {
                    on_part = false;
                    break;
                }
                const stl_triangle_vertex_indices &pf = part_surface.its.indices[part_face];
                const Vec3d pa = part_surface.its.vertices[pf[0]].cast<double>();
                const Vec3d part_normal = (part_surface.its.vertices[pf[1]].cast<double>() - pa).cross(part_surface.its.vertices[pf[2]].cast<double>() - pa);
                on_part = std::abs(part_normal.normalized().dot(normal.normalized())) >= FACET_PLANAR_COS;
            }
            if (! on_part) {
                const int first = int(boundary_surface.its.vertices.size());
                for (int i = 0; i < 3; ++ i)
                    boundary_surface.its.vertices.emplace_back(modifier.its.vertices[face[i]]);
                boundary_surface.its.indices.emplace_back(first, first + 1, first + 2);
            }
        }
        boundary_surface.build_tree();
    }
    const double infinite_distance = std::numeric_limits<double>::infinity();
    auto distance_to_boundary = [&](const Vec3d &p) { return boundary_surface.tree.empty() ? infinite_distance : boundary_surface.distance(p); };

    // Remesh: edges of fine_edge at the boundary, growing with the distance from it by edge_growth
    // up to coarse_edge while the distance is solved, and coarsening fast beyond. Inside the zone
    // the distance is needed only to inside_limit, so the mesh coarsens from there.
    auto size_at_distance = [&](double distance, bool inside) {
        const double limit  = (inside ? zone.inside_limit : zone.outside_limit) + margin;
        const double graded = std::min(coarse_edge, fine_edge + params.edge_growth * std::min(distance, limit));
        return distance <= limit ? graded : graded + distance - limit;
    };
    remesh_graded(vertices, faces, [&](const Vec3d &p, double radius) {
        const double distance = std::max(0., distance_to_boundary(p) - radius);
        // A ball clear of the boundary lies wholly inside or wholly outside the zone, which matters
        // only where it is far enough inside.
        const bool inside = distance > zone.inside_limit + margin && modifier_box.contains(p) && modifier.contains(p);
        return size_at_distance(distance, inside);
    });

    // Level at the vertices: the distance to the boundary surface, negative where the material
    // beneath the surface is inside the modifier. It is never zero at a vertex, so that the zone's
    // boundary, and with it the sources of the distance, come only from where the level crosses zero.
    std::vector<Vec3d> normals(vertices.size(), Vec3d::Zero());
    for (const Vec3i32 &face : faces) {
        const Vec3d normal = (vertices[face[1]] - vertices[face[0]]).cross(vertices[face[2]] - vertices[face[0]]);
        for (int i = 0; i < 3; ++ i)
            normals[face[i]] += normal;
    }
    std::vector<double> level(vertices.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vertices.size()), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t v = range.begin(); v < range.end(); ++ v) {
            const double length = normals[v].norm();
            // Away from the modifier's surface the point itself is as good as the probe beneath it.
            const Vec3d  probe  = modifier.distance(vertices[v]) > DIRECT_INSIDE_TEST || ! (length > 0.) ?
                                      vertices[v] : Vec3d(vertices[v] - (INSIDE_PROBE_DEPTH / length) * normals[v]);
            const bool   inside = modifier_box.contains(probe) && modifier.contains(probe);
            const double distance = std::max(ON_SURFACE_TOLERANCE, std::min(distance_to_boundary(vertices[v]), zone.outside_limit + 2. * margin));
            level[v] = inside ? - distance : distance;
        }
    });
    normals = {};
    if (std::none_of(level.begin(), level.end(), [](double l) { return l < 0.; }))
        // The modifier does not reach the surface.
        return;

    cut_at_zero_level(vertices, level, faces);

    // The straight line distance from the boundary bounds the geodesic one from below.
    std::vector<size_t> boundary;
    for (size_t v = 0; v < vertices.size(); ++ v)
        if (level[v] == 0.)
            boundary.emplace_back(v);
    double longest_boundary_edge = 0.;
    for (const Vec3i32 &face : faces)
        for (int i = 0; i < 3; ++ i)
            if (level[face[i]] == 0. && level[face[(i + 1) % 3]] == 0.)
                longest_boundary_edge = std::max(longest_boundary_edge, (vertices[face[i]] - vertices[face[(i + 1) % 3]]).norm());
    auto boundary_coordinate = [&vertices, &boundary](size_t idx, size_t dimension) { return vertices[boundary[idx]][dimension]; };
    const KDTreeIndirect<3, double, decltype(boundary_coordinate)> boundary_tree(boundary_coordinate, boundary.size());
    std::vector<double> lower_bound(vertices.size(), 0.);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, vertices.size()), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t v = range.begin(); v < range.end(); ++ v)
            if (level[v] != 0.) {
                lower_bound[v] = std::abs(level[v]);
                if (! boundary.empty()) {
                    const double to_boundary = (vertices[boundary[find_closest_point(boundary_tree, vertices[v])]] - vertices[v]).norm();
                    lower_bound[v] = std::max(lower_bound[v], to_boundary - 0.5 * longest_boundary_edge);
                }
            }
    });

    // Solve the distance on the band around the boundary where it is read.
    auto limit_at = [&](size_t v) { return level[v] < 0. ? zone.inside_limit : zone.outside_limit; };
    std::vector<double> solved(vertices.size(), infinity);
    bool                heat_failed = false;
    {
        std::vector<int>     band_index(vertices.size(), -1);
        std::vector<int>     band_vertices;
        std::vector<Vec3i32> band_faces;
        double               longest_edge = 0.;
        for (const Vec3i32 &face : faces) {
            bool in_band = false;
            for (int i = 0; i < 3; ++ i)
                in_band |= lower_bound[face[i]] < limit_at(face[i]) + margin;
            if (! in_band)
                continue;
            Vec3i32 band_face;
            for (int i = 0; i < 3; ++ i) {
                int &idx = band_index[face[i]];
                if (idx == -1) {
                    idx = int(band_vertices.size());
                    band_vertices.emplace_back(face[i]);
                }
                band_face[i] = idx;
                longest_edge = std::max(longest_edge, (vertices[face[i]] - vertices[face[(i + 1) % 3]]).norm());
            }
            band_faces.emplace_back(band_face);
        }
        std::vector<Vec3d> band_positions(band_vertices.size());
        std::vector<char>  band_sources(band_vertices.size());
        bool               any_source = false;
        for (size_t i = 0; i < band_vertices.size(); ++ i) {
            band_positions[i] = vertices[band_vertices[i]];
            band_sources[i]   = level[band_vertices[i]] == 0.;
            any_source       |= band_sources[i] != 0;
        }
        // The heat has to spread over the coarsest triangles of the band in one step.
        const double time_edge = std::min(longest_edge, coarse_edge);
        if (any_source) {
            const std::vector<double> band_distances = heat_distances(band_positions, band_faces, band_sources, HEAT_TIME_FACTOR * time_edge * time_edge);
            heat_failed = band_distances.empty();
            for (size_t i = 0; i < band_distances.size(); ++ i)
                solved[band_vertices[i]] = band_distances[i];
        }
    }

    zone.distance.resize(vertices.size());
    zone.surface.its.vertices.resize(vertices.size());
    for (size_t v = 0; v < vertices.size(); ++ v) {
        const double limit = limit_at(v);
        double       d     = 0.;
        if (level[v] != 0.) {
            // The straight line distance bounds the geodesic one from below. It stands in for it
            // where the solve did not reach, on a shell or a piece of the band with no boundary of
            // its own, or failed, and catches what the heat flow does not resolve far away.
            d = heat_failed || ! std::isfinite(solved[v]) ? lower_bound[v] : std::max(solved[v], lower_bound[v]);
            d = level[v] < 0. ? - std::min(d, limit) : std::min(d, limit);
        }
        zone.distance[v]             = float(d);
        zone.surface.its.vertices[v] = vertices[v].cast<float>();
    }
    zone.surface.its.indices.reserve(faces.size());
    for (const Vec3i32 &face : faces)
        zone.surface.its.indices.emplace_back(face);
    zone.surface.build_tree();
    zone.active = true;
}

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
            zone.ramp_length   = std::abs(in.wall_loops - in.base_wall_loops) * in.taper_ratio * params.bead_width;
            zone.inside_limit  = impl->params.max_shift_fraction * in.taper_ratio * params.bead_width;
            zone.outside_limit = zone.ramp_length + zone.inside_limit;
            build_zone(zone, parts[in.part], in.mesh, impl->params);
        }
    });
    if (std::none_of(impl->zones.begin(), impl->zones.end(), [](const ZoneField &zone) { return zone.active; }))
        return nullptr;
    return std::shared_ptr<const WallThicknessField>(new WallThicknessField(std::move(impl)));
}

size_t WallThicknessField::zone_count() const { return m_impl->zones.size(); }
bool   WallThicknessField::zone_active(size_t zone) const { return m_impl->zones[zone].active; }
double WallThicknessField::ramp_length(size_t zone) const { return m_impl->zones[zone].ramp_length; }
double WallThicknessField::signed_distance(size_t zone, const Vec3d &p) const { return m_impl->zones[zone].signed_distance(p); }

WallThickness WallThicknessField::at(const Vec3d &p, double shift_fraction, double default_base) const
{
    const WallThicknessFieldParams &params = m_impl->params;
    const double  shift   = std::clamp(shift_fraction, - params.max_shift_fraction, params.max_shift_fraction) * params.bead_width;
    // Thicker wins: the thickest of the zones that thicken the walls here, else the thinnest of
    // those that thin them, each against its own base.
    WallThickness thicker { - std::numeric_limits<double>::infinity(), 0. };
    WallThickness thinner { std::numeric_limits<double>::infinity(), 0. };
    for (const ZoneField &zone : m_impl->zones) {
        if (! zone.active)
            continue;
        const double base      = zone.base_wall_loops * params.bead_width;
        const double thickness = wall_ramp_thickness(zone.signed_distance(p) + shift * zone.taper_ratio, zone.base_wall_loops, zone.wall_loops,
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
    return std::any_of(m_impl->zones.begin(), m_impl->zones.end(), [&query](const ZoneField &zone) { return zone.active && zone.reach.intersects(query); });
}

double WallThicknessField::min_thickness() const
{
    double thickness = std::numeric_limits<double>::infinity();
    for (const ZoneField &zone : m_impl->zones)
        if (zone.active)
            thickness = std::min(thickness, std::min(zone.base_wall_loops, zone.wall_loops) * m_impl->params.bead_width);
    return thickness;
}

double WallThicknessField::max_thickness() const
{
    double thickness = 0.;
    for (const ZoneField &zone : m_impl->zones)
        if (zone.active)
            thickness = std::max(thickness, std::max(zone.base_wall_loops, zone.wall_loops) * m_impl->params.bead_width);
    return thickness;
}

size_t WallThicknessField::triangle_count() const
{
    size_t count = 0;
    for (const ZoneField &zone : m_impl->zones)
        count += zone.surface.its.indices.size();
    return count;
}

} // namespace Slic3r
