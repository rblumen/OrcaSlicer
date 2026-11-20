#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <tuple>
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

// The nominal layer height of every test.
constexpr double H = 0.2;
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

// From n0 layers to n1, both without a minimum thickness, at the nominal layer height h.
ShellRampProfile profile(int base_layers, int layers, double taper_ratio, double h = H) { return { base_layers, 0., layers, 0., taper_ratio, h }; }

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

// The Z distances of the k-th layer from a surface, k = 0..count-1, with every layer h thick.
std::vector<double> uniform_offsets(size_t count, double h = H)
{
    std::vector<double> out;
    for (size_t k = 0; k < count; ++ k)
        out.emplace_back(double(k) * h);
    return out;
}

std::optional<size_t> top_layers(const ShellDepthField &field, double x) { return field.solid_layers(ShellSide::Top, Vec3d(x, 10., TOP_Z), uniform_offsets(12)); }

// The x extent along the line y = 10 of the part of polygons over x = x0..x1, empty if none.
std::optional<std::pair<double, double>> x_extent_at_middle(const Polygons &polygons, double x0 = -100., double x1 = 100.)
{
    Polygon line;
    line.points = { Point::new_scale(x0, 9.9), Point::new_scale(x1, 9.9), Point::new_scale(x1, 10.1), Point::new_scale(x0, 10.1) };
    const Polygons crossing = intersection(polygons, Polygons{ line });
    if (crossing.empty())
        return std::nullopt;
    const BoundingBox box = get_extents(crossing);
    return std::make_pair(unscaled(box.min.x()), unscaled(box.max.x()));
}

double area_mm2(const Polygons &polygons) { return area(polygons) * SCALING_FACTOR * SCALING_FACTOR; }

} // namespace

TEST_CASE("The rule of a shell ramp profile runs linearly from the zone's to the base's over the ramp", "[ShellDepthField]")
{
    // 3 layers with a 0.4 mm minimum, 0.6 mm deep, to 2 layers with a 1.2 mm minimum: L_s = 0.6 mm * 5 = 3 mm.
    const ShellRampProfile thicker { 3, 0.4, 2, 1.2, 5., H };
    REQUIRE(thicker.active());
    CHECK(thicker.thickens());
    CHECK_THAT(thicker.ramp_length(), WithinAbs(3., 1e-9));
    for (const auto &[distance, layers, thickness] : { std::tuple{ -1., 2., 1.2 }, std::tuple{ 0., 2., 1.2 }, std::tuple{ 1.5, 2.5, 0.8 },
                                                       std::tuple{ 3., 3., 0.4 }, std::tuple{ 9., 3., 0.4 } }) {
        CAPTURE(distance);
        const ShellRule rule = thicker.at(distance);
        CHECK_THAT(rule.layers, WithinAbs(layers, 1e-9));
        CHECK_THAT(rule.thickness, WithinAbs(thickness, 1e-9));
    }

    // To no shell layers: the minimum thickness counts only with layers, so it runs to 0 with them.
    const ShellRampProfile none { 3, 0.4, 0, 1.2, 5., H };
    REQUIRE(none.active());
    CHECK_FALSE(none.thickens());
    CHECK_THAT(none.at(0.).thickness, WithinAbs(0., 1e-9));
    CHECK_THAT(none.at(1.5).layers, WithinAbs(1.5, 1e-9));
    CHECK_THAT(none.at(1.5).thickness, WithinAbs(0.2, 1e-9));

    // The same depth through other settings changes nothing: the base's rule everywhere.
    const ShellRampProfile same { 3, 0.4, 2, 0.6, 5., H };
    CHECK_FALSE(same.active());
    CHECK_THAT(same.at(-1.).layers, WithinAbs(3., 1e-9));
    CHECK_THAT(same.at(-1.).thickness, WithinAbs(0.4, 1e-9));
}

TEST_CASE("A layer is in the shell under the stock rule", "[ShellDepthField]")
{
    const ShellRule rule { 2.5, 0.7 };
    CHECK(rule.solid(2, 2.));
    CHECK(rule.solid(3, 0.6));
    CHECK_FALSE(rule.solid(3, 0.7));
    CHECK_FALSE(ShellRule{ 2., 0. }.solid(2, 0.));
}

// Top shell 3 layers, 7 in the zone, taper ratio 5: the ramp is 0.8 mm * 5 = 4 mm long, across the top face from where
// the modifier's sides cut it at x = 15 and 25. n(d) = 7 - d, and the shell has ceil(n(d)) layers.
TEST_CASE("The shell is the zone's inside it and steps down a layer at a time to the base's over the ramp", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(3, 7, 5.)) });
    REQUIRE(field != nullptr);

    for (const double x : { 20., 16., 24. }) {
        CAPTURE(x);
        CHECK(top_layers(*field, x) == std::optional<size_t>(7));
    }
    // Midway between the steps, clear of the distance's error.
    for (const double d : { 0.5, 1.5, 2.5, 3.5 }) {
        CAPTURE(d);
        for (const double x : { 15. - d, 25. + d }) {
            CAPTURE(x);
            CHECK(top_layers(*field, x) == std::optional<size_t>(size_t(std::ceil(7. - d))));
        }
    }
    // Beyond the ramp the zone has no opinion.
    for (const double x : { 5., 35. }) {
        CAPTURE(x);
        CHECK_FALSE(top_layers(*field, x).has_value());
    }
    // The modifier does not reach the bottom face, and the bottom shell is not on the Ramp anyway.
    CHECK_FALSE(field->solid_layers(ShellSide::Bottom, Vec3d(20., 10., BOTTOM_Z), uniform_offsets(12)).has_value());
}

// A shell thickened through its minimum thickness alone: 3 layers to 3 layers of at least 1.4 mm, a 4 mm ramp.
// T(d) = 1.4 - 0.35 d, and the shell has the layers k with k H < T(d), at least 3.
TEST_CASE("A shell thickened through its minimum thickness ramps through it", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), { 3, 0., 3, 1.4, 5., H }) });
    REQUIRE(field != nullptr);
    CHECK(top_layers(*field, 20.) == std::optional<size_t>(7));
    // T = 1.31 and 0.875.
    CHECK(top_layers(*field, 14.75) == std::optional<size_t>(7));
    CHECK(top_layers(*field, 26.5) == std::optional<size_t>(5));
    // T = 0.525: still the 3 layers.
    CHECK(top_layers(*field, 12.5) == std::optional<size_t>(3));
}

// The zone takes the top shell away; its ramp of 0.6 mm * 5 = 3 mm brings it back outside, n(d) = d.
TEST_CASE("A thinning zone has no shell inside and ramps up to the base's outside", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(3, 0, 5.)) });
    REQUIRE(field != nullptr);
    CHECK(top_layers(*field, 20.) == std::optional<size_t>(0));
    CHECK(top_layers(*field, 13.5) == std::optional<size_t>(2));
    CHECK(top_layers(*field, 26.5) == std::optional<size_t>(2));
    CHECK(top_layers(*field, 12.5) == std::optional<size_t>(3));
    CHECK_FALSE(top_layers(*field, 5.).has_value());
}

// A thickening zone over x = 15..25 with a 4 mm ramp, n(d) = 7 - d, and a thinning one over x = 27..35 with a 3 mm ramp.
// At x = 25.5 the ramps overlap; at x = 28.5 the thickening ramp reaches into the thinning zone. Each zone is compared
// with its own base, and the thicker shell wins.
TEST_CASE("Where a thickening and a thinning zone overlap the thicker shell wins", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(3, 7, 5.)), zone(modifier(27., 35., 2., 6.), profile(3, 0, 5.)) });
    REQUIRE(field != nullptr);
    CHECK(top_layers(*field, 25.5) == std::optional<size_t>(7));
    CHECK(top_layers(*field, 28.5) == std::optional<size_t>(4));
    // Past the thickening ramp, the thinning zone decides.
    CHECK(top_layers(*field, 31.) == std::optional<size_t>(0));
}

// Two zones taking the shell away, over x = 10..20 and x = 23..30, with ramps of 3 mm, n(d) = d, which overlap between
// them. A layer stays in the shell only if both zones put it there.
TEST_CASE("Where thinning zones overlap the shell is the thinnest", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(10., 20., 2., 6.), profile(3, 0, 5.)), zone(modifier(23., 30., 2., 6.), profile(3, 0, 5.)) });
    REQUIRE(field != nullptr);
    // n = 0.5 and 2.5, then 2.5 and 0.5.
    CHECK(top_layers(*field, 20.5) == std::optional<size_t>(1));
    CHECK(top_layers(*field, 22.5) == std::optional<size_t>(1));
}

// A modifier through the whole slab puts both faces in the zone. Only one side is on the Ramp; the other has a taper
// ratio of 0, the hard step, and the field has no opinion on it.
TEST_CASE("The top and bottom profiles of a zone ramp independently", "[ShellDepthField]")
{
    const bool top_ramps = GENERATE(true, false);
    CAPTURE(top_ramps);
    const ShellRampProfile ramp = profile(3, 7, 5.);
    const ShellRampProfile hard = profile(3, 7, 0.);
    const auto field = build_on_slab({ zone(modifier(15., 25., -1., 5.), top_ramps ? ramp : hard, top_ramps ? hard : ramp) });
    REQUIRE(field != nullptr);

    const ShellSide ramped   = top_ramps ? ShellSide::Top : ShellSide::Bottom;
    const ShellSide fixed    = top_ramps ? ShellSide::Bottom : ShellSide::Top;
    const double    z_ramped = top_ramps ? TOP_Z : BOTTOM_Z;
    const double    z_fixed  = top_ramps ? BOTTOM_Z : TOP_Z;
    for (const double x : { 20., 12.5 }) {
        CAPTURE(x);
        CHECK(field->solid_layers(ramped, Vec3d(x, 10., z_ramped), uniform_offsets(12)) == std::optional<size_t>(x == 20. ? 7 : 5));
        CHECK_FALSE(field->solid_layers(fixed, Vec3d(x, 10., z_fixed), uniform_offsets(12)).has_value());
    }
    CHECK_THAT(field->deepest(ramped).layers, WithinAbs(7., 1e-9));
    CHECK_THAT(field->deepest(fixed).layers, WithinAbs(0., 1e-9));
}

TEST_CASE("A point further than the surface tolerance from the part has no shell", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., -1., 6.), profile(3, 7, 5.), profile(3, 7, 5.)) });
    REQUIRE(field != nullptr);
    REQUIRE(field->solid_layers(ShellSide::Top, Vec3d(20., 10., TOP_Z), uniform_offsets(12)).has_value());
    // The slab's middle is 2 mm from its faces, and the point above it 1 mm off the top face, both inside the
    // modifier; the tolerance is 0.5 mm.
    for (const Vec3d &p : { Vec3d(20., 10., 2.), Vec3d(20., 10., 5.) }) {
        CAPTURE(p.z());
        CHECK_FALSE(field->solid_layers(ShellSide::Top, p, uniform_offsets(12)).has_value());
        CHECK_FALSE(field->solid_layers(ShellSide::Bottom, p, uniform_offsets(12)).has_value());
    }
}

// The top layer, with layers of H below it: layer k is in the shell where n(d) = 7 - d > k, so the edge of level k lies
// 7 - k from the zone's edge, capped at the end of the ramp, L = 4 mm. Levels 0 to 6 exist, as 7 layers is the most.
TEST_CASE("The shell levels of a layer nest and their edges follow the ramp", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(3, 7, 5.)) });
    REQUIRE(field != nullptr);
    const BoundingBox      extents(Point::new_scale(0., 0.), Point::new_scale(40., 20.));
    const LayerShellLevels levels = make_layer_shell_levels(*field, extents, TOP_Z, uniform_offsets(11), uniform_offsets(11));
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
        const double past_edge = std::min(4., 7. - double(k));
        const auto   extent    = x_extent_at_middle(top.level[k]);
        REQUIRE(extent.has_value());
        CHECK_THAT(extent->first, WithinAbs(15. - past_edge, LEVEL_TOLERANCE));
        CHECK_THAT(extent->second, WithinAbs(25. + past_edge, LEVEL_TOLERANCE));
        // Within the reach, and within the level before.
        CHECK(area_mm2(diff(top.level[k], top.reach)) < 1e-3);
        if (k > 0)
            CHECK(area_mm2(diff(top.level[k], top.level[k - 1])) < 1e-3);
        // Within the base's 3 layers, the whole reach: no gap against the stock shell outside it.
        if (k < 3)
            CHECK(area_mm2(diff(top.reach, top.level[k])) < 1e-3);
    }
    CHECK(top.at(7).empty());
}

// A thinning zone, 3 layers to none, under a bottom surface whose first layer is thinner than the rest, as a first
// layer of 0.2 mm under layers of 0.28 mm: L_s = 0.84 mm * 5 = 4.2 mm, n(d) = 3 d / L_s. Layer k is in the shell
// where n(d) > k, out to the end of the ramp, d > 1.4 k, and never at k = 3, as stock outside the reach, though the
// 4th layer is 0.76 mm from the surface, within the base's depth of 0.84 mm.
TEST_CASE("The shell levels of a thinning zone follow its ramp and match the base's at its end", "[ShellDepthField]")
{
    const auto field = build_on_slab({ zone(modifier(15., 25., -1., 2.), {}, profile(3, 0, 5., 0.28)) });
    REQUIRE(field != nullptr);
    const BoundingBox         extents(Point::new_scale(0., 0.), Point::new_scale(40., 20.));
    const std::vector<double> offsets { 0., 0.2, 0.48, 0.76, 1.04, 1.32 };
    const LayerShellLevels    levels = make_layer_shell_levels(*field, extents, BOTTOM_Z, offsets, offsets);
    const ShellLevels        &bottom = levels.bottom;
    CHECK(levels.top.empty());

    REQUIRE_FALSE(bottom.empty());
    const auto reach = x_extent_at_middle(bottom.reach);
    REQUIRE(reach.has_value());
    CHECK_THAT(reach->first, WithinAbs(15. - 4.2, LEVEL_TOLERANCE));
    CHECK_THAT(reach->second, WithinAbs(25. + 4.2, LEVEL_TOLERANCE));

    REQUIRE(bottom.level.size() == 3);
    for (size_t k = 0; k < bottom.level.size(); ++ k) {
        CAPTURE(k);
        // The levels lie on either side of the zone, out to the end of the ramp.
        const double past_edge = 1.4 * double(k);
        const auto   left      = x_extent_at_middle(bottom.level[k], -100., 20.);
        const auto   right     = x_extent_at_middle(bottom.level[k], 20., 100.);
        REQUIRE(left.has_value());
        REQUIRE(right.has_value());
        CHECK_THAT(left->first, WithinAbs(reach->first, LEVEL_TOLERANCE));
        CHECK_THAT(left->second, WithinAbs(15. - past_edge, LEVEL_TOLERANCE));
        CHECK_THAT(right->first, WithinAbs(25. + past_edge, LEVEL_TOLERANCE));
        CHECK_THAT(right->second, WithinAbs(reach->second, LEVEL_TOLERANCE));
        CHECK(area_mm2(diff(bottom.level[k], bottom.reach)) < 1e-3);
        if (k > 0)
            CHECK(area_mm2(diff(bottom.level[k], bottom.level[k - 1])) < 1e-3);
    }
    CHECK(bottom.at(3).empty());
}

TEST_CASE("A modifier that does not reach the part's surface gives no shell depth field", "[ShellDepthField]")
{
    // Wholly inside the slab, and wholly above it.
    for (const indexed_triangle_set &mesh : { box_between({ 15., 5., 1. }, { 25., 15., 3. }), box_between({ 15., 5., 6. }, { 25., 15., 8. }) })
        CHECK(build_on_slab({ zone(mesh, profile(3, 7, 5.)) }) == nullptr);
    // A zone whose profiles change nothing.
    CHECK(build_on_slab({ zone(modifier(15., 25., 2., 6.), profile(3, 3, 5.), profile(3, 7, 0.)) }) == nullptr);
}

TEST_CASE("The shell depth is the layer count's depth, at least the minimum thickness, and none without layers", "[ShellDepthField]")
{
    CHECK_THAT(shell_depth(3, 0.4, 0.2), WithinAbs(0.6, 1e-9));
    CHECK_THAT(shell_depth(3, 1., 0.2), WithinAbs(1., 1e-9));
    CHECK_THAT(shell_depth(0, 1., 0.2), WithinAbs(0., 1e-9));
}
