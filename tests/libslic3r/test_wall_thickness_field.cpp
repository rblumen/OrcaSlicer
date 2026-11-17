#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/Tesselate.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/WallThicknessField.hpp"
#include "libslic3r/libslic3r.h"

using namespace Slic3r;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// The plan asks for distances within 2%. Close to the boundary a relative bound means nothing,
// so there the error may be a fraction of the w/2 the boundary is resolved to.
constexpr double DISTANCE_TOLERANCE = 0.02;
constexpr double DISTANCE_FLOOR     = 0.05;

// The bead width of every test.
constexpr double W = 0.45;

auto IsDistance(double expected) { return WithinAbs(expected, std::max(DISTANCE_FLOOR, DISTANCE_TOLERANCE * std::abs(expected))); }

indexed_triangle_set translated(indexed_triangle_set its, const Vec3f &offset)
{
    for (stl_vertex &v : its.vertices)
        v += offset;
    return its;
}

// A cylinder along Z of the given radius, centered at (x, y), spanning z_min to z_max.
indexed_triangle_set cylinder_at(double x, double y, double radius, double z_min, double z_max)
{
    return translated(its_make_cylinder(radius, z_max - z_min), Vec3f(float(x), float(y), float(z_min)));
}

// A box spanning min to max.
indexed_triangle_set box_between(const Vec3d &min, const Vec3d &max)
{
    const Vec3d size = max - min;
    return translated(its_make_cube(size.x(), size.y(), size.z()), min.cast<float>());
}

WallRampZone ramp_zone(indexed_triangle_set mesh, int base_wall_loops, int wall_loops, double taper_ratio, size_t part = 0)
{
    WallRampZone zone;
    zone.mesh            = std::move(mesh);
    zone.part            = part;
    zone.base_wall_loops = base_wall_loops;
    zone.wall_loops      = wall_loops;
    zone.taper_ratio     = taper_ratio;
    return zone;
}

// Every parameter the assertions depend on, set.
WallThicknessFieldParams test_params()
{
    WallThicknessFieldParams params;
    params.bead_width              = W;
    params.max_shift_fraction      = 0.25;
    params.max_edge_fraction       = 0.5;
    params.coarse_edge_fraction    = 1.;
    params.edge_growth             = 0.1;
    params.surface_tolerance       = 0.5;
    params.contour_offset_fraction = 0.5;
    return params;
}

std::shared_ptr<const WallThicknessField> build_on(const indexed_triangle_set &part, const std::vector<WallRampZone> &zones)
{
    return WallThicknessField::build({ part }, zones, test_params());
}

// A prism of the given section, from z = 0 to height.
indexed_triangle_set make_prism(const ExPolygon &section, double height)
{
    indexed_triangle_set its;
    its_merge(its, triangulate_expolygon_3d(section, height, NORMALS_UP));
    its_merge(its, triangulate_expolygon_3d(section, 0., NORMALS_DOWN));
    auto add_walls = [&its, height](const Polygon &polygon) {
        Pointf3s triangles;
        for (size_t i = 0; i < polygon.points.size(); ++ i) {
            const Vec2d p = unscaled(polygon.points[i]);
            const Vec2d q = unscaled(polygon.points[(i + 1) % polygon.points.size()]);
            const Vec3d p0(p.x(), p.y(), 0.), p1(p.x(), p.y(), height), q0(q.x(), q.y(), 0.), q1(q.x(), q.y(), height);
            for (const Vec3d &v : { p0, q0, q1, p0, q1, p1 })
                triangles.emplace_back(v);
        }
        its_merge(its, triangles);
    };
    add_walls(section.contour);
    for (const Polygon &hole : section.holes)
        add_walls(hole);
    its_merge_vertices(its);
    return its;
}

Polygon rectangle(double x_min, double y_min, double x_max, double y_max)
{
    Polygon polygon;
    polygon.points = { Point::new_scale(x_min, y_min), Point::new_scale(x_max, y_min), Point::new_scale(x_max, y_max), Point::new_scale(x_min, y_max) };
    return polygon;
}

// An L bracket: its section in XZ is a leg of 40 x 5 along X and a leg of 5 x 30 along Z, sharing
// the corner at the origin; it spans 0 to 20 in Y. Twenty triangles, as coarse as an STL gets.
indexed_triangle_set make_l_bracket()
{
    const double section[6][2] = { { 0, 0 }, { 40, 0 }, { 40, 5 }, { 5, 5 }, { 5, 30 }, { 0, 30 } };
    indexed_triangle_set its;
    for (double y : { 0., 20. })
        for (const auto &p : section)
            its.vertices.emplace_back(float(p[0]), float(y), float(p[1]));
    for (int i = 0; i < 6; ++ i) {
        const int j = (i + 1) % 6;
        its.indices.emplace_back(i, i + 6, j);
        its.indices.emplace_back(j, i + 6, j + 6);
    }
    // The section as a fan from its corner at the origin, facing -Y at the front and +Y at the back.
    for (int i = 1; i < 5; ++ i) {
        its.indices.emplace_back(0, i, i + 1);
        its.indices.emplace_back(6, i + 7, i + 6);
    }
    return its;
}

// The test coupon of the plan: a 190 x 35 x 4 mm strap with a 6.5 mm hole 15 mm from each end. Its
// faces are fans of long slivers, as an STL's often are.
indexed_triangle_set make_coupon()
{
    const double length = 190., width = 35., hole_radius = 3.25, edge_distance = 15.;
    ExPolygon section;
    section.contour = rectangle(0., 0., length, width);
    for (double x : { edge_distance, length - edge_distance }) {
        Polygon hole;
        const int steps = 64;
        for (int i = 0; i < steps; ++ i) {
            const double angle = - 2. * PI * i / steps;
            hole.points.emplace_back(Point::new_scale(x + hole_radius * std::cos(angle), width / 2. + hole_radius * std::sin(angle)));
        }
        section.holes.emplace_back(std::move(hole));
    }
    return make_prism(section, 4.);
}

// The coupon's two zones, around its holes.
std::vector<WallRampZone> coupon_zones(double taper_ratio)
{
    std::vector<WallRampZone> zones;
    for (double x : { 15., 175. })
        zones.push_back(ramp_zone(cylinder_at(x, 17.5, 20., - 3., 7.), 2, 4, taper_ratio));
    return zones;
}

// A 60 x 40 x 4 mm plate, with the axis of the modifiers of the tests below at its middle.
const double PLATE_HEIGHT = 4.;
const Vec2d  PLATE_CENTER(30., 20.);

indexed_triangle_set make_plate() { return its_make_cube(2. * PLATE_CENTER.x(), 2. * PLATE_CENTER.y(), PLATE_HEIGHT); }

// A cylinder modifier through the plate, centered offset mm along X from its middle.
indexed_triangle_set plate_cylinder(double radius, double offset = 0.)
{
    return cylinder_at(PLATE_CENTER.x() + offset, PLATE_CENTER.y(), radius, - 3., PLATE_HEIGHT + 3.);
}

// A point of the plate's top (or bottom) face at the given distance and direction from its middle.
Vec3d plate_point(double radius, double angle_deg, double z = PLATE_HEIGHT)
{
    const double angle = angle_deg * PI / 180.;
    return Vec3d(PLATE_CENTER.x() + radius * std::cos(angle), PLATE_CENTER.y() + radius * std::sin(angle), z);
}

} // namespace

TEST_CASE("Wall thickness ramps linearly from the zone's wall count to the base count", "[WallBand]")
{
    // Two beads over a ramp of 2 * 10 * 0.5 = 10 mm, both up and down.
    const double taper_ratio = 10., bead_width = 0.5, ramp_length = 10.;
    auto thickening = [&](double distance) { return wall_ramp_thickness(distance, 2, 4, taper_ratio, bead_width); };
    auto thinning   = [&](double distance) { return wall_ramp_thickness(distance, 4, 2, taper_ratio, bead_width); };

    CHECK_THAT(thickening(- 3.), WithinAbs(2., 1e-12));
    CHECK_THAT(thickening(0.), WithinAbs(2., 1e-12));
    CHECK_THAT(thickening(ramp_length / 4.), WithinAbs(1.75, 1e-12));
    CHECK_THAT(thickening(ramp_length / 2.), WithinAbs(1.5, 1e-12));
    CHECK_THAT(thickening(ramp_length), WithinAbs(1., 1e-12));
    CHECK_THAT(thickening(3. * ramp_length), WithinAbs(1., 1e-12));

    CHECK_THAT(thinning(- 3.), WithinAbs(1., 1e-12));
    CHECK_THAT(thinning(0.), WithinAbs(1., 1e-12));
    CHECK_THAT(thinning(ramp_length / 2.), WithinAbs(1.5, 1e-12));
    CHECK_THAT(thinning(ramp_length), WithinAbs(2., 1e-12));
    CHECK_THAT(thinning(3. * ramp_length), WithinAbs(2., 1e-12));

    // No walls in the zone: three beads over 3 * 10 * 0.5 = 15 mm.
    CHECK_THAT(wall_ramp_thickness(- 1., 3, 0, taper_ratio, bead_width), WithinAbs(0., 1e-12));
    CHECK_THAT(wall_ramp_thickness(5., 3, 0, taper_ratio, bead_width), WithinAbs(0.5, 1e-12));
    CHECK_THAT(wall_ramp_thickness(15., 3, 0, taper_ratio, bead_width), WithinAbs(1.5, 1e-12));
    // A zone that keeps the count changes nothing.
    CHECK_THAT(wall_ramp_thickness(- 1., 3, 3, taper_ratio, bead_width), WithinAbs(1.5, 1e-12));
}

TEST_CASE("Distance on a plate from a cylinder modifier through it is the distance to the circle", "[WallBand]")
{
    const double radius = 8.;
    // Ramp length 2 * 10 * 0.45 = 9 mm, resolved from 1.125 mm inside the circle to 10.125 mm outside.
    const auto field = build_on(make_plate(), { ramp_zone(plate_cylinder(radius), 2, 4, 10.) });
    REQUIRE(field);
    REQUIRE(field->zone_active(0));
    REQUIRE_THAT(field->ramp_length(0), WithinAbs(9., 1e-9));

    for (double z : { 0., PLATE_HEIGHT })
        for (double angle : { 0., 33., 90., 147., 200., 285. })
            for (double distance : { - 1., - 0.4, 0., 0.3, 1., 2.5, 4.5, 7., 9., 10. }) {
                CAPTURE(z, angle, distance);
                CHECK_THAT(field->signed_distance(0, plate_point(radius + distance, angle, z)), IsDistance(distance));
            }
}

TEST_CASE("Distance saturates beyond the range the thickness depends on", "[WallBand]")
{
    const double radius = 8.;
    // Ramp length 2 * 4 * 0.45 = 3.6 mm and the largest shift 0.25 * 4 * 0.45 = 0.45 mm.
    const auto field = build_on(make_plate(), { ramp_zone(plate_cylinder(radius), 2, 4, 4.) });
    REQUIRE(field);

    CHECK_THAT(field->signed_distance(0, plate_point(0., 0.)), WithinAbs(- 0.45, 1e-6));
    CHECK_THAT(field->signed_distance(0, plate_point(radius - 3., 70.)), WithinAbs(- 0.45, 1e-6));
    CHECK_THAT(field->signed_distance(0, plate_point(radius + 8., 70.)), WithinAbs(4.05, 1e-6));
    CHECK_THAT(field->signed_distance(0, Vec3d(1., 1., PLATE_HEIGHT)), WithinAbs(4.05, 1e-6));
}

TEST_CASE("Distance across the bend of an L bracket is the unfolded distance", "[WallBand]")
{
    const indexed_triangle_set bracket = make_l_bracket();
    REQUIRE_THAT(its_volume(bracket), WithinRel(6500., 1e-4));

    // The zone is the end of the horizontal leg beyond x = 12. Ramp length 2 * 20 * 0.45 = 18 mm.
    const double boundary_x = 12.;
    const auto   field      = build_on(bracket, { ramp_zone(box_between(Vec3d(boundary_x, - 5., - 5.), Vec3d(50., 25., 10.)), 2, 4, 20.) });
    REQUIRE(field);

    auto check = [&field](const Vec3d &point, double expected) {
        CAPTURE(point.x(), point.y(), point.z(), expected);
        CHECK_THAT(field->signed_distance(0, point), IsDistance(expected));
    };
    for (double y : { 4., 10., 15. }) {
        // Inside the zone, and towards the bend along the bottom face.
        check(Vec3d(boundary_x + 2., y, 0.), - 2.);
        for (double x : { 10., 6., 1. })
            check(Vec3d(x, y, 0.), boundary_x - x);
        // Around the outer corner and up the outer face of the vertical leg. Higher up, or closer to
        // the bracket's ends, the shortest path runs over an end face instead (12 + the distance to
        // that end, or around the inner corner), so the points stay below the nearest end's distance.
        for (double z : { 1., 3. })
            check(Vec3d(0., y, z), boundary_x + z);
        // Along the top face of the horizontal leg to the inner corner at x = 5, z = 5.
        for (double x : { 10., 7. })
            check(Vec3d(x, y, 5.), boundary_x - x);
        // Around the inner corner and up the inner face of the vertical leg.
        for (double z : { 6., 10., 15. })
            check(Vec3d(5., y, z), boundary_x - 5. + z - 5.);
    }
}

TEST_CASE("Distance around a slot is longer than the straight line through it", "[WallBand]")
{
    // A 60 x 40 x 4 mm plate with a slot through it, x 28 to 32 and y 5 to 35, and a zone of the
    // plate beyond x = 20 on the other side. From (40, 30) the way round the slot's near end is
    // |(40, 30) - (32, 35)| + 12 = 21.43 mm; the straight line is 20 mm. Paths that bend round a
    // corner meet behind it, where the heat method smooths the distance: it comes out a few
    // percent long, so the bound here is 5%, which still tells it from the straight line.
    ExPolygon section;
    section.contour = rectangle(0., 0., 60., 40.);
    Polygon slot    = rectangle(28., 5., 32., 35.);
    slot.reverse();
    section.holes.emplace_back(std::move(slot));
    // Ramp length 2 * 40 * 0.45 = 36 mm.
    const auto field = build_on(make_prism(section, 4.), { ramp_zone(box_between(Vec3d(- 5., - 5., - 5.), Vec3d(20., 45., 10.)), 2, 4, 40.) });
    REQUIRE(field);

    const double geodesic = std::hypot(8., 5.) + 12.;
    const double distance = field->signed_distance(0, Vec3d(40., 30., 4.));
    CHECK_THAT(distance, WithinRel(geodesic, 0.05));
    CHECK(distance > 20.8);
}

TEST_CASE("Thickness on a plate takes the ramp's values at its start, middle and end", "[WallBand]")
{
    const double radius = 8.;
    // Ramp length 2 * 10 * 0.45 = 9 mm; a 2% error of the distance is 0.018 mm of thickness there.
    const double ramp_length = 9.;
    const auto   field       = build_on(make_plate(), { ramp_zone(plate_cylinder(radius), 2, 4, 10.) });
    REQUIRE(field);

    CHECK_THAT(field->min_thickness(), WithinAbs(2 * W, 1e-12));
    CHECK_THAT(field->max_thickness(), WithinAbs(4 * W, 1e-12));
    const double base = 2 * W;
    for (double angle : { 10., 120., 250. }) {
        CAPTURE(angle);
        CHECK_THAT(field->thickness(plate_point(0., angle), 0., base), WithinAbs(4 * W, 1e-9));
        CHECK_THAT(field->thickness(plate_point(radius - 2., angle), 0., base), WithinAbs(4 * W, 1e-9));
        CHECK_THAT(field->thickness(plate_point(radius, angle), 0., base), WithinAbs(4 * W, 0.01));
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length / 2., angle), 0., base), WithinAbs(3 * W, 0.01));
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length, angle), 0., base), WithinAbs(2 * W, 0.02));
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length + 2., angle), 0., base), WithinAbs(2 * W, 1e-9));
        // The base is the zone's N0 in the zone and on its ramp, and the caller's default elsewhere.
        CHECK_THAT(field->at(plate_point(radius + ramp_length / 2., angle), 0., 7.).base, WithinAbs(2 * W, 1e-12));
        CHECK_THAT(field->at(plate_point(radius + ramp_length + 2., angle), 0., 7.).base, WithinAbs(7., 1e-12));
    }
}

TEST_CASE("A shift slides the ramp along the surface by that fraction of the taper ratio times the bead width", "[WallBand]")
{
    const double radius = 8.;
    // Ramp length 9 mm. A shift of a quarter is 0.25 * 10 * 0.45 = 1.125 mm of distance, which
    // is an eighth of the ramp, or a quarter of a bead width.
    const double ramp_length = 9., shift = 1.125, base = 2 * W;
    const auto   field       = build_on(make_plate(), { ramp_zone(plate_cylinder(radius), 2, 4, 10.) });
    REQUIRE(field);

    for (double angle : { 45., 190. }) {
        CAPTURE(angle);
        // The ramp starts and ends that much earlier with a positive shift, later with a negative one.
        CHECK_THAT(field->thickness(plate_point(radius, angle), 0.25, base), WithinAbs(4 * W - 0.1125, 0.01));
        CHECK_THAT(field->thickness(plate_point(radius - shift, angle), 0.25, base), WithinAbs(4 * W, 0.01));
        CHECK_THAT(field->thickness(plate_point(radius + shift, angle), - 0.25, base), WithinAbs(4 * W, 0.01));
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length, angle), - 0.25, base), WithinAbs(2 * W + 0.1125, 0.02));
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length - shift, angle), 0.25, base), WithinAbs(2 * W, 0.02));
        // A shift beyond the largest one the field was built for is clamped to it.
        CHECK_THAT(field->thickness(plate_point(radius, angle), 1., base), WithinAbs(4 * W - 0.1125, 0.01));
    }
}

TEST_CASE("A modifier that does not reach the part's surface gives no field", "[WallBand]")
{
    const indexed_triangle_set part = its_make_cube(20., 20., 20.);

    SECTION("wholly inside the part") {
        CHECK_FALSE(build_on(part, { ramp_zone(box_between(Vec3d(7., 7., 7.), Vec3d(12., 12., 12.)), 2, 4, 10.) }));
    }
    SECTION("wholly outside the part") {
        CHECK_FALSE(build_on(part, { ramp_zone(box_between(Vec3d(30., 0., 0.), Vec3d(40., 10., 10.)), 2, 4, 10.) }));
    }
    SECTION("resting on a face of the part from outside") {
        CHECK_FALSE(build_on(part, { ramp_zone(box_between(Vec3d(5., 5., 20.), Vec3d(15., 15., 30.)), 2, 4, 10.) }));
    }
}

TEST_CASE("A zone that does not change the wall count gives no field", "[WallBand]")
{
    const indexed_triangle_set modifier = plate_cylinder(8.);

    CHECK_FALSE(build_on(make_plate(), {}));
    CHECK_FALSE(build_on(make_plate(), { ramp_zone(modifier, 3, 3, 10.) }));
    CHECK_FALSE(build_on(make_plate(), { ramp_zone(modifier, 3, 5, 0.) }));
    CHECK(build_on(make_plate(), { ramp_zone(modifier, 3, 5, 10.) }));
    CHECK(build_on(make_plate(), { ramp_zone(modifier, 3, 2, 10.) }));
}

TEST_CASE("Only the zones that reach the surface are active", "[WallBand]")
{
    const auto field = build_on(make_plate(), {
        ramp_zone(box_between(Vec3d(20., 15., 1.), Vec3d(25., 20., 3.)), 2, 6, 10.), // buried in the plate
        ramp_zone(plate_cylinder(8.), 2, 4, 10.),
    });
    REQUIRE(field);

    CHECK(field->zone_count() == 2);
    CHECK_FALSE(field->zone_active(0));
    CHECK(field->zone_active(1));
    CHECK_THAT(field->max_thickness(), WithinAbs(4 * W, 1e-12));
    CHECK_THAT(field->thickness(plate_point(0., 0.), 0., 2 * W), WithinAbs(4 * W, 1e-9));
}

TEST_CASE("Where zones overlap the thickness is that of the larger wall count", "[WallBand]")
{
    // A wide zone of 4 walls with a 9 mm ramp, and a narrow one of 6 walls with a ramp of
    // 4 * 2 * 0.45 = 3.6 mm, centered 10 mm off the first so that it straddles its boundary.
    const double wide_radius = 14., narrow_radius = 5., narrow_offset = 10., base = 2 * W;
    const auto   field = build_on(make_plate(), { ramp_zone(plate_cylinder(wide_radius), 2, 4, 10.),
                                                  ramp_zone(plate_cylinder(narrow_radius, narrow_offset), 2, 6, 2.) });
    REQUIRE(field);

    CHECK_THAT(field->max_thickness(), WithinAbs(6 * W, 1e-12));
    // Inside both zones.
    CHECK_THAT(field->thickness(plate_point(narrow_offset, 0.), 0., base), WithinAbs(6 * W, 1e-9));
    // Inside the narrow zone, 0.5 mm outside the wide one.
    CHECK_THAT(field->thickness(plate_point(wide_radius + 0.5, 0.), 0., base), WithinAbs(6 * W, 1e-9));
    // Inside the wide zone only, beyond the ramp of the narrow one.
    CHECK_THAT(field->thickness(plate_point(8., 180.), 0., base), WithinAbs(4 * W, 1e-9));
    // Inside the wide zone, 1.8 mm outside the narrow one, half way down its ramp from 6 to 2 walls.
    CHECK_THAT(field->thickness(plate_point(narrow_offset - narrow_radius - 1.8, 0.), 0., base), WithinAbs(4 * W, 0.03));
    // 0.9 mm outside the narrow zone its ramp has dropped one bead of four; the wide zone's 4 walls are fewer.
    CHECK_THAT(field->thickness(plate_point(narrow_offset - narrow_radius - 0.9, 0.), 0., base), WithinAbs(5 * W, 0.03));
}

TEST_CASE("A thinning zone has its wall count inside and ramps back up to the base count", "[WallBand]")
{
    const double radius = 8.;
    // From 4 walls to 2: ramp length 2 * 10 * 0.45 = 9 mm.
    const double ramp_length = 9., base = 4 * W;
    const auto   field       = build_on(make_plate(), { ramp_zone(plate_cylinder(radius), 4, 2, 10.) });
    REQUIRE(field);
    REQUIRE_THAT(field->ramp_length(0), WithinAbs(ramp_length, 1e-9));

    CHECK_THAT(field->min_thickness(), WithinAbs(2 * W, 1e-12));
    CHECK_THAT(field->max_thickness(), WithinAbs(4 * W, 1e-12));
    for (double angle : { 30., 160., 300. }) {
        CAPTURE(angle);
        CHECK_THAT(field->thickness(plate_point(radius - 3., angle), 0., base), WithinAbs(2 * W, 1e-9));
        CHECK_THAT(field->at(plate_point(radius - 3., angle), 0., base).base, WithinAbs(4 * W, 1e-12));
        for (double distance : { 0., ramp_length / 4., ramp_length / 2., 3. * ramp_length / 4. }) {
            CAPTURE(distance);
            CHECK_THAT(field->thickness(plate_point(radius + distance, angle), 0., base), WithinAbs(W * (2. + 2. * distance / ramp_length), 0.02));
        }
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length + 2., angle), 0., base), WithinAbs(4 * W, 1e-9));
    }
}

TEST_CASE("A zone without walls has none inside and ramps up to the base count", "[WallBand]")
{
    const double radius = 8.;
    // From 2 walls to none: ramp length 2 * 10 * 0.45 = 9 mm.
    const double ramp_length = 9., base = 2 * W;
    const auto   field       = build_on(make_plate(), { ramp_zone(plate_cylinder(radius), 2, 0, 10.) });
    REQUIRE(field);

    CHECK_THAT(field->min_thickness(), WithinAbs(0., 1e-12));
    for (double angle : { 0., 135., 270. }) {
        CAPTURE(angle);
        CHECK_THAT(field->thickness(plate_point(0., angle), 0., base), WithinAbs(0., 1e-9));
        CHECK_THAT(field->thickness(plate_point(radius - 2., angle), 0., base), WithinAbs(0., 1e-9));
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length / 2., angle), 0., base), WithinAbs(W, 0.02));
        CHECK_THAT(field->thickness(plate_point(radius + ramp_length + 2., angle), 0., base), WithinAbs(2 * W, 1e-9));
    }
}

TEST_CASE("Where a thickening zone overlaps a thinning one the thicker wins", "[WallBand]")
{
    // A wide zone without walls and a 9 mm ramp, and a narrow zone of 4 walls with a ramp of
    // 2 * 2 * 0.45 = 1.8 mm, centered 10 mm off the first so that it straddles its boundary.
    const double wide_radius = 14., narrow_radius = 5., narrow_offset = 10., base = 2 * W;
    const auto   field = build_on(make_plate(), { ramp_zone(plate_cylinder(wide_radius), 2, 0, 10.),
                                                  ramp_zone(plate_cylinder(narrow_radius, narrow_offset), 2, 4, 2.) });
    REQUIRE(field);

    CHECK_THAT(field->min_thickness(), WithinAbs(0., 1e-12));
    CHECK_THAT(field->max_thickness(), WithinAbs(4 * W, 1e-12));
    // Inside both zones, and inside the narrow one just outside the wide one.
    CHECK_THAT(field->thickness(plate_point(narrow_offset, 0.), 0., base), WithinAbs(4 * W, 1e-9));
    CHECK_THAT(field->thickness(plate_point(wide_radius + 0.5, 0.), 0., base), WithinAbs(4 * W, 1e-9));
    // Inside the wide zone, half way down the narrow zone's ramp: one bead above the base.
    CHECK_THAT(field->thickness(plate_point(narrow_offset - narrow_radius - 0.9, 0.), 0., base), WithinAbs(3 * W, 0.03));
    CHECK_THAT(field->at(plate_point(narrow_offset - narrow_radius - 0.9, 0.), 0., base).base, WithinAbs(2 * W, 1e-12));
    // Inside the wide zone only, beyond the narrow zone's ramp.
    CHECK_THAT(field->thickness(plate_point(8., 180.), 0., base), WithinAbs(0., 1e-9));
}

TEST_CASE("Where two thinning zones overlap the thinner wins", "[WallBand]")
{
    // From 4 walls, a wide zone of 2 with a ramp of 2 * 10 * 0.45 = 9 mm, and a narrow zone of 1
    // with a ramp of 3 * 2 * 0.45 = 2.7 mm, centered 10 mm off the first.
    const double wide_radius = 14., narrow_radius = 5., narrow_offset = 10., base = 4 * W;
    const auto   field = build_on(make_plate(), { ramp_zone(plate_cylinder(wide_radius), 4, 2, 10.),
                                                  ramp_zone(plate_cylinder(narrow_radius, narrow_offset), 4, 1, 2.) });
    REQUIRE(field);

    CHECK_THAT(field->min_thickness(), WithinAbs(W, 1e-12));
    // Inside both zones, and inside the narrow one 0.5 mm outside the wide one.
    CHECK_THAT(field->thickness(plate_point(narrow_offset, 0.), 0., base), WithinAbs(W, 1e-9));
    CHECK_THAT(field->thickness(plate_point(wide_radius + 0.5, 0.), 0., base), WithinAbs(W, 1e-9));
    // Inside the wide zone, half way up the narrow zone's ramp, where it is thicker than the wide zone.
    CHECK_THAT(field->thickness(plate_point(narrow_offset - narrow_radius - 1.35, 0.), 0., base), WithinAbs(2 * W, 1e-9));
    // A sixth of the way up the narrow zone's ramp it is thinner than the wide zone's 2 walls.
    CHECK_THAT(field->thickness(plate_point(narrow_offset - narrow_radius - 0.45, 0.), 0., base), WithinAbs(1.5 * W, 0.03));
    // Inside the wide zone only.
    CHECK_THAT(field->thickness(plate_point(8., 180.), 0., base), WithinAbs(2 * W, 1e-9));
}

TEST_CASE("Pieces of one modifier on parts with different wall counts each ramp to their own part's count", "[WallBand]")
{
    // Two 40 x 40 x 4 plates 1 mm apart along X, of 2 and 3 walls, and one cylinder of 5 walls
    // across the gap. Ramps of 3 * 10 * 0.45 = 13.5 mm on the first, 2 * 10 * 0.45 = 9 mm on the
    // second, saturating 0.25 * 10 * 0.45 = 1.125 mm beyond.
    const indexed_triangle_set first = its_make_cube(40., 40., 4.), second = translated(its_make_cube(40., 40., 4.), Vec3f(41.f, 0.f, 0.f));
    const double               center_x = 40.5, radius = 8.;
    const indexed_triangle_set modifier = cylinder_at(center_x, 20., radius, - 3., 7.);
    const auto field = WallThicknessField::build({ first, second }, { ramp_zone(modifier, 2, 5, 10., 0), ramp_zone(modifier, 3, 5, 10., 1) }, test_params());
    REQUIRE(field);
    REQUIRE_THAT(field->ramp_length(0), WithinAbs(13.5, 1e-9));
    REQUIRE_THAT(field->ramp_length(1), WithinAbs(9., 1e-9));

    const double default_base = 9.;
    auto on_first  = [&](double distance) { return Vec3d(center_x - radius - distance, 20., 4.); };
    auto on_second = [&](double distance) { return Vec3d(center_x + radius + distance, 20., 4.); };
    for (double fraction : { 0., 0.25, 0.5, 0.75 }) {
        CAPTURE(fraction);
        const WallThickness a = field->at(on_first(fraction * 13.5), 0., default_base);
        const WallThickness b = field->at(on_second(fraction * 9.), 0., default_base);
        CHECK_THAT(a.thickness, WithinAbs(W * (5. - 3. * fraction), 0.02));
        CHECK_THAT(a.base, WithinAbs(2 * W, 1e-12));
        CHECK_THAT(b.thickness, WithinAbs(W * (5. - 2. * fraction), 0.02));
        CHECK_THAT(b.base, WithinAbs(3 * W, 1e-12));
    }
    // Beyond each ramp no zone reaches, and the caller's default applies.
    CHECK_THAT(field->thickness(on_first(15.), 0., default_base), WithinAbs(default_base, 1e-12));
    CHECK_THAT(field->thickness(on_second(10.5), 0., default_base), WithinAbs(default_base, 1e-12));
}

TEST_CASE("A part wholly inside the modifier is reinforced everywhere", "[WallBand]")
{
    const auto field = build_on(its_make_cube(10., 10., 10.), { ramp_zone(box_between(Vec3d(- 5., - 5., - 5.), Vec3d(15., 15., 15.)), 2, 4, 10.) });
    REQUIRE(field);

    CHECK_THAT(field->thickness(Vec3d(5., 5., 10.), 0., 2 * W), WithinAbs(4 * W, 1e-9));
    CHECK_THAT(field->thickness(Vec3d(0., 3., 2.), 0., 2 * W), WithinAbs(4 * W, 1e-9));
    CHECK_THAT(field->thickness(Vec3d(10., 10., 0.), 0., 2 * W), WithinAbs(4 * W, 1e-9));
}

TEST_CASE("A face the modifier shares with the part from inside belongs to the zone", "[WallBand]")
{
    const double radius = 8.;
    // The cylinder is exactly as tall as the plate, so its caps lie in the plate's faces.
    const indexed_triangle_set flush = cylinder_at(PLATE_CENTER.x(), PLATE_CENTER.y(), radius, 0., PLATE_HEIGHT);
    const auto                 field = build_on(make_plate(), { ramp_zone(flush, 2, 4, 10.) });
    REQUIRE(field);

    for (double z : { 0., PLATE_HEIGHT })
        for (double angle : { 20., 110., 260. }) {
            CAPTURE(z, angle);
            // Inside the zone the distance is negative, so a shift towards the zone keeps all its walls.
            CHECK(field->signed_distance(0, plate_point(radius - 3., angle, z)) < - 1.);
            CHECK_THAT(field->thickness(plate_point(0., angle, z), 0., 2 * W), WithinAbs(4 * W, 1e-9));
            CHECK_THAT(field->thickness(plate_point(radius - 3., angle, z), 0.25, 2 * W), WithinAbs(4 * W, 1e-9));
            CHECK_THAT(field->signed_distance(0, plate_point(radius - 0.5, angle, z)), IsDistance(- 0.5));
            CHECK_THAT(field->signed_distance(0, plate_point(radius + 3., angle, z)), IsDistance(3.));
            CHECK_THAT(field->thickness(plate_point(radius + 12., angle, z), 0., 2 * W), WithinAbs(2 * W, 1e-9));
        }
    // The shared faces are meshed finely only near the zone's boundary: no more triangles than for
    // a cylinder through the plate, give or take.
    const auto through = build_on(make_plate(), { ramp_zone(plate_cylinder(radius), 2, 4, 10.) });
    REQUIRE(through);
    CHECK(field->triangle_count() < 3 * through->triangle_count() / 2);

    // A flush zone without walls keeps none with the shift either.
    const auto empty = build_on(make_plate(), { ramp_zone(flush, 2, 0, 10.) });
    REQUIRE(empty);
    CHECK_THAT(empty->thickness(plate_point(radius - 3., 45.), 0.25, 2 * W), WithinAbs(0., 1e-9));
}

TEST_CASE("Thickness is continuous onto a second shell the zone does not reach", "[WallBand]")
{
    // Two overlapping boxes as separate shells, the zone on the first. The second shell has no zone
    // boundary of its own, so the straight line distance stands in there. Ramp 2 * 20 * 0.45 = 18 mm.
    indexed_triangle_set part = box_between(Vec3d(0., 0., 0.), Vec3d(20., 20., 4.));
    its_merge(part, box_between(Vec3d(17., 0., 0.), Vec3d(40., 20., 4.)));
    const auto field = build_on(part, { ramp_zone(cylinder_at(8., 10., 4., - 3., 7.), 2, 4, 20.) });
    REQUIRE(field);

    double previous = field->thickness(Vec3d(12.5, 10., 4.), 0., 2 * W);
    for (double x = 12.75; x <= 35.; x += 0.25) {
        const double thickness = field->thickness(Vec3d(x, 10., 4.), 0., 2 * W);
        CAPTURE(x, previous, thickness);
        // The ramp drops 2 * 0.45 / 18 = 0.025 mm of thickness per 0.25 mm.
        CHECK(std::abs(thickness - previous) < 0.06);
        previous = thickness;
    }
}

TEST_CASE("A point further off the surface than the tolerance is out of the zone", "[WallBand]")
{
    const auto field = build_on(make_plate(), { ramp_zone(plate_cylinder(8.), 2, 4, 10.) });
    REQUIRE(field);
    const double tolerance = test_params().surface_tolerance;

    for (double side : { 1., - 1. }) {
        CAPTURE(side);
        const double z_face = side > 0. ? PLATE_HEIGHT : 0.;
        CHECK_THAT(field->thickness(plate_point(3., 30., z_face + side * (tolerance - 0.05)), 0., 2 * W), WithinAbs(4 * W, 1e-9));
        CHECK_THAT(field->thickness(plate_point(3., 30., z_face + side * (tolerance + 0.05)), 0., 2 * W), WithinAbs(2 * W, 1e-12));
    }
    // Contours are looked up just outside: inside the plate's material, between its faces, too.
    CHECK_THAT(field->thickness(plate_point(3., 30., PLATE_HEIGHT - 0.3), 0., 2 * W), WithinAbs(4 * W, 1e-9));
}

TEST_CASE("A contour of a thin wall is looked up on the face it comes from", "[WallBand]")
{
    // A fin 0.6 mm thick, its face y = 0 in a zone that ends at y = 0.3, its face y = 0.6 not. From
    // (2, 0.6, 5) the way to the zone runs around the end at x = 0: 2 + 0.3 mm, past the ramp of
    // 2 * 2 * 0.45 = 1.8 mm.
    const indexed_triangle_set fin   = box_between(Vec3d(0., 0., 0.), Vec3d(20., 0.6, 10.));
    const auto                 field = build_on(fin, { ramp_zone(box_between(Vec3d(- 5., - 5., - 5.), Vec3d(5., 0.3, 15.)), 2, 4, 2.) });
    REQUIRE(field);

    // A contour point of the face y = 0.6, moved 0.32 mm into the material by compensation, is
    // closer to the face y = 0.
    const Vec2d point(2., 0.28), outward(0., 1.);
    REQUIRE_THAT(field->thickness(Vec3d(point.x(), point.y(), 5.), 0., 2 * W), WithinAbs(4 * W, 1e-9));
    const std::vector<WallThickness> contour = field->at_contour({ point, Vec2d(2., 0.32) }, { outward, Vec2d(0., 2.) }, 5., 0., 2 * W);
    REQUIRE(contour.size() == 2);
    CHECK_THAT(contour[0].thickness, WithinAbs(2 * W, 1e-12));
    CHECK_THAT(contour[1].thickness, WithinAbs(2 * W, 1e-12));
    // A contour point of the face y = 0 stays in the zone.
    const std::vector<WallThickness> other = field->at_contour({ Vec2d(2., 0.05) }, { Vec2d(0., - 1.) }, 5., 0., 2 * W);
    CHECK_THAT(other.front().thickness, WithinAbs(4 * W, 1e-9));
}

TEST_CASE("Only boxes near a zone may be affected", "[WallBand]")
{
    // Ramp 9 mm and its saturation 1.125 mm beyond, around a circle of radius 8 at (30, 20).
    const auto field = build_on(make_plate(), { ramp_zone(plate_cylinder(8.), 2, 4, 10.) });
    REQUIRE(field);

    CHECK(field->may_affect(BoundingBoxf3(Vec3d(25., 15., 0.), Vec3d(35., 25., 4.))));
    CHECK(field->may_affect(BoundingBoxf3(Vec3d(45., 15., 0.), Vec3d(50., 25., 4.))));
    CHECK_FALSE(field->may_affect(BoundingBoxf3(Vec3d(55., 15., 0.), Vec3d(60., 25., 4.))));
    CHECK_FALSE(field->may_affect(BoundingBoxf3(Vec3d(25., 15., 20.), Vec3d(35., 25., 24.))));
    CHECK_FALSE(field->may_affect(BoundingBoxf3()));
}

TEST_CASE("Concurrent queries give the same results as serial ones", "[WallBand]")
{
    const auto field = build_on(make_plate(), { ramp_zone(plate_cylinder(8.), 2, 4, 10.), ramp_zone(plate_cylinder(5., 10.), 2, 0, 3.) });
    REQUIRE(field);

    std::vector<Vec3d> points;
    for (int i = 0; i < 60; ++ i)
        for (int j = 0; j < 40; ++ j)
            points.emplace_back(0.5 + i, 0.5 + j, (i + j) % 2 ? PLATE_HEIGHT : 0.);
    std::vector<WallThickness> serial, parallel(points.size());
    for (const Vec3d &p : points)
        serial.emplace_back(field->at(p, 0.25, 2 * W));
    tbb::parallel_for(tbb::blocked_range<size_t>(0, points.size(), 16), [&](const tbb::blocked_range<size_t> &range) {
        for (size_t i = range.begin(); i < range.end(); ++ i)
            parallel[i] = field->at(points[i], 0.25, 2 * W);
    });
    size_t mismatches = 0;
    for (size_t i = 0; i < points.size(); ++ i)
        mismatches += serial[i].thickness != parallel[i].thickness || serial[i].base != parallel[i].base;
    CHECK(mismatches == 0);
}

TEST_CASE("A part of long slivers is remeshed into a bounded number of triangles", "[WallBand]")
{
    // The coupon's faces are fans of slivers up to 190 mm long. Subdividing them would make millions
    // of triangles; remeshed, the two zones of 9 mm long ramps take about 100 thousand.
    const auto field = build_on(make_coupon(), coupon_zones(5.));
    REQUIRE(field);
    CHECK(field->triangle_count() < 400000);
}

TEST_CASE("An open or degenerate part does not stop the build", "[WallBand]")
{
    // A box without its top, a degenerate triangle and a lone open triangle, with a zone across all.
    indexed_triangle_set part = its_make_cube(20., 20., 4.);
    part.indices.erase(std::remove_if(part.indices.begin(), part.indices.end(), [&part](const stl_triangle_vertex_indices &f) {
        return part.vertices[f[0]].z() > 3.f && part.vertices[f[1]].z() > 3.f && part.vertices[f[2]].z() > 3.f;
    }), part.indices.end());
    const int base = int(part.vertices.size());
    for (const Vec3f &v : { Vec3f(5.f, 5.f, 2.f), Vec3f(6.f, 5.f, 2.f), Vec3f(7.f, 5.f, 2.f), Vec3f(30.f, 0.f, 0.f), Vec3f(40.f, 0.f, 0.f), Vec3f(30.f, 10.f, 0.f) })
        part.vertices.emplace_back(v);
    part.indices.emplace_back(base, base + 1, base + 2);
    part.indices.emplace_back(base + 3, base + 4, base + 5);

    std::shared_ptr<const WallThicknessField> field;
    CHECK_NOTHROW(field = build_on(part, { ramp_zone(box_between(Vec3d(10., - 5., - 5.), Vec3d(45., 25., 10.)), 2, 4, 10.) }));
    if (field) {
        const double thickness = field->thickness(Vec3d(5., 0., 2.), 0., 2 * W);
        CHECK(std::isfinite(thickness));
        CHECK(thickness >= 2 * W);
        CHECK(thickness <= 4 * W);
    }
}

// Hidden: it measures rather than tests, and its times depend on the machine. Run it with
// "[WallBandTiming]" to print how long the field of the plan's test coupon takes to build at the
// three taper ratios of the test groups, and how far its distances on the top face are from the
// exact ones. Within a few mm of the strap's long edges the heat method smooths the distance by a
// few percent (see the field's documentation of accuracy); the 2% bound is checked on the plate and
// the L bracket above.
TEST_CASE("The thickness field of the test coupon is timed and measured", "[.][WallBand][WallBandTiming]")
{
    const indexed_triangle_set coupon = make_coupon();
    REQUIRE_THAT(its_volume(coupon), WithinRel((190. * 35. - 2. * PI * 3.25 * 3.25) * 4., 1e-2));

    const double zone_radius = 20., hole_x = 15., center_y = 17.5;
    for (double taper_ratio : { 5., 10., 20. }) {
        double                                    best_ms = 1e9;
        std::shared_ptr<const WallThicknessField> field;
        for (int run = 0; run < 3; ++ run) {
            const auto start = std::chrono::steady_clock::now();
            field = build_on(coupon, coupon_zones(taper_ratio));
            best_ms = std::min(best_ms, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        REQUIRE(field);

        // On the top face towards the shank the distance is that to the zone's circle.
        const double ramp_length = field->ramp_length(0);
        // Error over the tolerance of the tests above: 2% of the distance, at least DISTANCE_FLOOR.
        double max_relative = 0., max_absolute = 0., max_over_tolerance = 0., worst_distance = 0.;
        Vec3d  worst        = Vec3d::Zero();
        size_t queries      = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int a = - 50; a <= 50; a += 2)
            for (int i = 1; i <= 100; ++ i) {
                const double distance = ramp_length * i / 100.;
                const double angle    = a * PI / 180.;
                const Vec3d  point(hole_x + (zone_radius + distance) * std::cos(angle), center_y + (zone_radius + distance) * std::sin(angle), 4.);
                if (point.y() < 0. || point.y() > 35.)
                    continue;
                const double error = std::abs(field->signed_distance(0, point) - distance);
                max_absolute       = std::max(max_absolute, error);
                max_relative       = std::max(max_relative, error / distance);
                const double over  = error / std::max(DISTANCE_FLOOR, DISTANCE_TOLERANCE * distance);
                if (over > max_over_tolerance) {
                    max_over_tolerance = over;
                    worst              = point;
                    worst_distance     = distance;
                }
                ++ queries;
            }
        const double query_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / double(queries);

        std::ostringstream report;
        report << "taper ratio " << taper_ratio << ": build " << best_ms << " ms, " << field->triangle_count() << " triangles, max error "
               << max_absolute << " mm, " << 100. * max_relative << "%, " << max_over_tolerance << " of the tolerance (at " << worst.x() << ", "
               << worst.y() << ": " << field->signed_distance(0, worst) << " for " << worst_distance << "), " << query_us << " us per query";
        // Printed directly, as Catch2 would wrap it.
        std::cout << report.str() << std::endl;
        CHECK(queries > 0);
    }
}
