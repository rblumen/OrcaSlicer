#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/ShellDepthField.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/libslic3r.h"

using namespace Slic3r;
using Catch::Matchers::WithinAbs;

namespace {

// The layer height of every test, and the depths of 0, 3 and 7 layers of it.
constexpr double H  = 0.2;
constexpr double D0 = 3 * H;
constexpr double D1 = 7 * H;
// The heat method distance is within about 0.05 mm, which the depth's slope of at most 0.25 mm per mm makes 0.0125.
constexpr double DEPTH_TOLERANCE = 0.02;
// A level's edge is found to within a grid cell of 0.5 mm, plus the distance's own error.
constexpr double LEVEL_TOLERANCE = 0.55;

indexed_triangle_set box_between(const Vec3d &min, const Vec3d &max)
{
    const Vec3d          size = max - min;
    indexed_triangle_set its  = its_make_cube(size.x(), size.y(), size.z());
    for (stl_vertex &v : its.vertices)
        v += min.cast<float>();
    return its;
}

// The part of every test: a slab over x = 0..40, y = 0..20, z = 0..4.
indexed_triangle_set slab() { return box_between({ 0., 0., 0. }, { 40., 20., 4. }); }
// A z just below the slab's top face and one just above its bottom face, where the top and bottom layers are sampled.
constexpr double TOP_Z    = 4. - 0.5 * H;
constexpr double BOTTOM_Z = 0.5 * H;

// A box modifier over x = x0..x1 sticking out of the slab in Y, and in Z over z0..z1.
indexed_triangle_set modifier(double x0, double x1, double z0, double z1) { return box_between({ x0, -5., z0 }, { x1, 25., z1 }); }

ShellRampProfile profile(double base_depth, double depth, double taper_ratio) { return { base_depth, depth, taper_ratio }; }

ShellRampZone zone(indexed_triangle_set mesh, const ShellRampProfile &top, const ShellRampProfile &bottom = {})
{
    ShellRampZone out;
    out.mesh   = std::move(mesh);
    out.top    = top;
    out.bottom = bottom;
    return out;
}

// Every parameter the assertions depend on, set.
ShellDepthFieldParams test_params()
{
    ShellDepthFieldParams params;
    params.fine_edge         = 0.25;
    params.coarse_edge       = 0.5;
    params.edge_growth       = 0.1;
    params.surface_tolerance = 0.5;
    params.outside_margin    = 1.;
    return params;
}

std::shared_ptr<const ShellDepthField> build_on_slab(const std::vector<ShellRampZone> &zones) { return ShellDepthField::build({ slab() }, zones, test_params()); }

std::optional<double> top_depth(const ShellDepthField &field, double x) { return field.depth(ShellSide::Top, Vec3d(x, 10., TOP_Z)); }

// The x extent of polygons along the line y = 10, empty if they do not cross it.
std::optional<std::pair<double, double>> x_extent_at_middle(const Polygons &polygons)
{
    Polygon line;
    line.points = { Point::new_scale(-100., 9.9), Point::new_scale(100., 9.9), Point::new_scale(100., 10.1), Point::new_scale(-100., 10.1) };
    const Polygons crossing = intersection(polygons, Polygons{ line });
    if (crossing.empty())
        return std::nullopt;
    const BoundingBox box = get_extents(crossing);
    return std::make_pair(unscaled(box.min.x()), unscaled(box.max.x()));
}

} // namespace

// Top shell 3 layers deep, 7 in the zone, taper ratio 5: the ramp is 0.8 mm * 5 = 4 mm long, across the top face from
// where the modifier's sides cut it at x = 15 and 25.
TEST_CASE("The shell depth is the zone's inside it and falls linearly to the base depth over the ramp", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(D0, D1, 5.)) });
    REQUIRE(field != nullptr);
    const double ramp = (D1 - D0) * 5.;
    REQUIRE_THAT(ramp, WithinAbs(4., 1e-9));

    for (const double x : { 20., 16., 24. }) {
        CAPTURE(x);
        REQUIRE(top_depth(*field, x).has_value());
        CHECK_THAT(*top_depth(*field, x), WithinAbs(D1, DEPTH_TOLERANCE));
    }
    for (const double d : { 1., 2., 3. }) {
        CAPTURE(d);
        const double expected = D1 - (D1 - D0) * d / ramp;
        for (const double x : { 15. - d, 25. + d }) {
            CAPTURE(x);
            REQUIRE(top_depth(*field, x).has_value());
            CHECK_THAT(*top_depth(*field, x), WithinAbs(expected, DEPTH_TOLERANCE));
        }
    }
    // Beyond the ramp the zone has no opinion, and the sample runs on at the base depth.
    for (const double x : { 5., 35. }) {
        CAPTURE(x);
        CHECK_FALSE(top_depth(*field, x).has_value());
        const ShellSample sample = field->sample(Vec3d(x, 10., TOP_Z));
        CHECK(sample.top_reach < 0.f);
        CHECK_THAT(sample.top_depth, WithinAbs(D0, 1e-6));
    }
    // The modifier does not reach the bottom face, and the bottom shell is not on the Ramp anyway.
    CHECK_FALSE(field->depth(ShellSide::Bottom, Vec3d(20., 10., BOTTOM_Z)).has_value());
}

// The zone takes the top shell away; its ramp of 0.6 mm * 5 = 3 mm brings it back outside.
TEST_CASE("A thinning zone has no shell inside and ramps up to the base depth outside", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(D0, 0., 5.)) });
    REQUIRE(field != nullptr);
    REQUIRE(top_depth(*field, 20.).has_value());
    CHECK_THAT(*top_depth(*field, 20.), WithinAbs(0., DEPTH_TOLERANCE));
    REQUIRE(top_depth(*field, 13.5).has_value());
    CHECK_THAT(*top_depth(*field, 13.5), WithinAbs(0.5 * D0, DEPTH_TOLERANCE));
    REQUIRE(top_depth(*field, 26.5).has_value());
    CHECK_THAT(*top_depth(*field, 26.5), WithinAbs(0.5 * D0, DEPTH_TOLERANCE));
    CHECK_FALSE(top_depth(*field, 5.).has_value());
}

// A deepening zone over x = 15..25 with a 4 mm ramp, and a thinning one over x = 27..35 with a 3 mm ramp. At x = 26
// the ramps overlap; at x = 28 the deepening ramp reaches into the thinning zone. Each zone is compared with its own
// base, and the deeper shell wins.
TEST_CASE("Where a deepening and a thinning zone overlap the deeper shell wins", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(D0, D1, 5.)), zone(modifier(27., 35., 2., 6.), profile(D0, 0., 5.)) });
    REQUIRE(field != nullptr);
    auto deepening = [](double d) { return D1 - (D1 - D0) * d / 4.; };
    REQUIRE(top_depth(*field, 26.).has_value());
    CHECK_THAT(*top_depth(*field, 26.), WithinAbs(deepening(1.), DEPTH_TOLERANCE));
    REQUIRE(top_depth(*field, 28.).has_value());
    CHECK_THAT(*top_depth(*field, 28.), WithinAbs(deepening(3.), DEPTH_TOLERANCE));
    // Past the deepening ramp, the thinning zone decides.
    REQUIRE(top_depth(*field, 31.).has_value());
    CHECK_THAT(*top_depth(*field, 31.), WithinAbs(0., DEPTH_TOLERANCE));
}

// A modifier through the whole slab puts both faces in the zone. Only one side is on the Ramp; the other has a taper
// ratio of 0, the hard step, and the field has no opinion on it.
TEST_CASE("The top and bottom profiles of a zone ramp independently", "[ShellDepthField]")
{
    const bool top_ramps = GENERATE(true, false);
    CAPTURE(top_ramps);
    const ShellRampProfile ramp = profile(D0, D1, 5.);
    const ShellRampProfile hard = profile(D0, D1, 0.);
    const auto field = build_on_slab({ zone(modifier(15., 25., -1., 5.), top_ramps ? ramp : hard, top_ramps ? hard : ramp) });
    REQUIRE(field != nullptr);

    const ShellSide ramped = top_ramps ? ShellSide::Top : ShellSide::Bottom;
    const ShellSide fixed  = top_ramps ? ShellSide::Bottom : ShellSide::Top;
    const double    z_ramped = top_ramps ? TOP_Z : BOTTOM_Z;
    const double    z_fixed  = top_ramps ? BOTTOM_Z : TOP_Z;
    for (const double x : { 20., 13. }) {
        CAPTURE(x);
        const std::optional<double> depth = field->depth(ramped, Vec3d(x, 10., z_ramped));
        REQUIRE(depth.has_value());
        CHECK_THAT(*depth, WithinAbs(x == 20. ? D1 : D1 - (D1 - D0) * 2. / 4., DEPTH_TOLERANCE));
        CHECK_FALSE(field->depth(fixed, Vec3d(x, 10., z_fixed)).has_value());
    }
    CHECK_THAT(field->max_depth(ramped), WithinAbs(D1, 1e-9));
    CHECK_THAT(field->max_depth(fixed), WithinAbs(0., 1e-9));
}

TEST_CASE("A point further than the surface tolerance from the part has no shell depth", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., -1., 6.), profile(D0, D1, 5.), profile(D0, D1, 5.)) });
    REQUIRE(field != nullptr);
    REQUIRE(field->depth(ShellSide::Top, Vec3d(20., 10., TOP_Z)).has_value());
    // The slab's middle is 2 mm from its faces, and the point above it 1 mm off the top face, both inside the
    // modifier; the tolerance is 0.5 mm.
    for (const Vec3d &p : { Vec3d(20., 10., 2.), Vec3d(20., 10., 5.) }) {
        CAPTURE(p.z());
        CHECK_FALSE(field->depth(ShellSide::Top, p).has_value());
        CHECK_FALSE(field->depth(ShellSide::Bottom, p).has_value());
        CHECK(field->sample(p).top_reach < 0.f);
    }
}

// The top layer, with layers of H below it: layer k is in the shell where the depth exceeds k * H, so the edge of
// level k lies (D1 - k H) / (D1 - D0) * L from the zone's edge, capped at the end of the ramp, L = 4 mm. Levels 0 to 6
// exist, as 7 H is the deepest shell.
TEST_CASE("The shell levels of a layer nest and their edges follow the ramp", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(D0, D1, 5.)) });
    REQUIRE(field != nullptr);
    std::vector<double> offsets;
    for (int k = 0; k <= 10; ++ k)
        offsets.emplace_back(k * H);
    const BoundingBox      extents(Point::new_scale(0., 0.), Point::new_scale(40., 20.));
    const LayerShellLevels levels = make_layer_shell_levels(*field, extents, TOP_Z, offsets, offsets);
    const ShellLevels     &top    = levels.top;
    // The bottom shell is not on the Ramp.
    CHECK(levels.bottom.empty());
    CHECK(levels.bottom.level.empty());

    REQUIRE_FALSE(top.empty());
    const auto reach = x_extent_at_middle(top.reach);
    REQUIRE(reach.has_value());
    CHECK_THAT(reach->first, WithinAbs(11., LEVEL_TOLERANCE));
    CHECK_THAT(reach->second, WithinAbs(29., LEVEL_TOLERANCE));

    REQUIRE(top.level.size() == 7);
    for (size_t k = 0; k < top.level.size(); ++ k) {
        CAPTURE(k);
        const double past_edge = std::min(4., (D1 - double(k) * H) / (D1 - D0) * 4.);
        const auto   extent    = x_extent_at_middle(top.level[k]);
        REQUIRE(extent.has_value());
        CHECK_THAT(extent->first, WithinAbs(15. - past_edge, LEVEL_TOLERANCE));
        CHECK_THAT(extent->second, WithinAbs(25. + past_edge, LEVEL_TOLERANCE));
        // Within the reach, and within the level before.
        CHECK(area(diff(top.level[k], top.reach)) < 1.);
        if (k > 0)
            CHECK(area(diff(top.level[k], top.level[k - 1])) < 1.);
    }
    CHECK(top.at(7).empty());
}

TEST_CASE("A modifier that does not reach the part's surface gives no shell depth field", "[ShellDepthField]")
{
    // Wholly inside the slab, and wholly above it.
    for (const indexed_triangle_set &mesh : { box_between({ 15., 5., 1. }, { 25., 15., 3. }), box_between({ 15., 5., 6. }, { 25., 15., 8. }) })
        CHECK(build_on_slab({ zone(mesh, profile(D0, D1, 5.)) }) == nullptr);
    // A zone whose profiles change nothing.
    CHECK(build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(D0, D0, 5.), profile(D0, D1, 0.)) }) == nullptr);
}

TEST_CASE("The shell depth is the layer count's depth, at least the minimum thickness, and none without layers", "[ShellDepthField]")
{
    CHECK_THAT(shell_depth(3, 0.4, 0.2), WithinAbs(0.6, 1e-9));
    CHECK_THAT(shell_depth(3, 1., 0.2), WithinAbs(1., 1e-9));
    CHECK_THAT(shell_depth(0, 1., 0.2), WithinAbs(0., 1e-9));
}
