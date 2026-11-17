#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/SVG.hpp"
#include "libslic3r/WallBandShape.hpp"
#include "libslic3r/libslic3r.h"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/nowide/cstdlib.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>

using namespace Slic3r;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Bead width w in mm.
constexpr double W = 0.45;

// Samples every w / 4, band at least w / 2 wide, base thickness of `base_walls` beads.
WallBandParams band_params() { return WallBandParams::for_bead_width(W); }

ExPolygon rectangle_mm(double x0, double y0, double x1, double y1)
{
    ExPolygon out;
    out.contour.points = { Point::new_scale(x0, y0), Point::new_scale(x1, y0), Point::new_scale(x1, y1), Point::new_scale(x0, y1) };
    return out;
}

// A clockwise circle, to be used as a hole.
Polygon hole_mm(const Vec2d &center, double radius)
{
    constexpr size_t segments = 64;
    Polygon out;
    for (size_t i = 0; i < segments; ++ i) {
        const double a = - 2. * PI * double(i) / double(segments);
        out.points.emplace_back(Point::new_scale(center.x() + radius * std::cos(a), center.y() + radius * std::sin(a)));
    }
    return out;
}

// The plan's linear t(d): zone_t inside the zone (d <= 0), base_t past the ramp (d >= ramp_length).
double ramp_thickness(double d, double ramp_length, double zone_t, double base_t)
{
    return zone_t + (base_t - zone_t) * std::clamp(d / ramp_length, 0., 1.);
}

// Distance in mm from `origin` along the unit vector `dir` to the first crossing of the boundary of `regions`.
double first_hit_mm(const ExPolygons &regions, const Vec2d &origin, const Vec2d &dir)
{
    double best = std::numeric_limits<double>::max();
    for (const Polygon &poly : to_polygons(regions))
        for (size_t i = 0; i < poly.points.size(); ++ i) {
            const Vec2d  a = unscaled(poly.points[i]);
            const Vec2d  e = unscaled(poly.points[(i + 1) % poly.points.size()]) - a;
            const double denom = dir.x() * e.y() - dir.y() * e.x();
            if (std::abs(denom) < 1e-12)
                continue;
            const Vec2d  ao = a - origin;
            const double s  = (ao.x() * e.y() - ao.y() * e.x()) / denom;
            const double u  = (ao.x() * dir.y() - ao.y() * dir.x()) / denom;
            if (s >= 0. && u >= 0. && u <= 1.)
                best = std::min(best, s);
        }
    return best;
}

bool strictly_inside(const ExPolygons &regions, const Point &pt)
{
    return std::any_of(regions.begin(), regions.end(), [&pt](const ExPolygon &expoly) { return expoly.contains(pt, false); });
}

// Band width along the top edge of a rectangle whose top is at y = top, measured downward at x.
double top_band_width(const WallBand &band, double x, double top) { return first_hit_mm(band.inner, Vec2d(x, top), Vec2d(0., -1.)); }

// First x in [x_from, x_to] where the band width along the top edge falls below `width`, interpolated between scan steps.
double top_width_crossing(const WallBand &band, double top, double width, double x_from, double x_to)
{
    constexpr double step = 0.05;
    double x_prev = x_from;
    double w_prev = top_band_width(band, x_prev, top);
    for (double x = x_from + step; x <= x_to; x += step) {
        const double w = top_band_width(band, x, top);
        if (w < width)
            return x_prev + (w_prev - width) / (w_prev - w) * (x - x_prev);
        x_prev = x;
        w_prev = w;
    }
    return std::numeric_limits<double>::max();
}

} // namespace

TEST_CASE("A constant thickness gives the band of a uniform inward offset", "[WallBand][WallBandShape]")
{
    ExPolygon plate_with_hole = rectangle_mm(0., 0., 60., 30.);
    plate_with_hole.holes.emplace_back(hole_mm(Vec2d(15., 15.), 3.25));
    ExPolygon l_shape;
    l_shape.contour.points = { Point::new_scale(0., 0.), Point::new_scale(40., 0.), Point::new_scale(40., 10.),
                               Point::new_scale(10., 10.), Point::new_scale(10., 30.), Point::new_scale(0., 30.) };
    const ExPolygon island = GENERATE_COPY(rectangle_mm(0., 0., 40., 20.), plate_with_hole, l_shape);

    // Anything but the base thickness of 2w, so the band is built.
    const double t = 3. * W;
    const std::optional<WallBand> band = build_wall_band(island, [t](const Point &) { return WallThicknessSample{ t, 2. * W }; }, band_params());
    REQUIRE(band.has_value());

    const ExPolygons stock_band = diff_ex(island, offset_ex(island, - float(scale_(t))));
    CHECK_THAT(area(band->band), WithinRel(area(stock_band), 0.01));
    CHECK_THAT(area(band->band) + area(band->inner), WithinRel(island.area(), 1e-6));
}

TEST_CASE("The band width follows a thickness that ramps along the contour", "[WallBand][WallBandShape]")
{
    // A 60 x 20 mm rectangle, zone at x <= 20 mm, ramp over 9 mm. Thickening: 4w zone on 2w walls.
    // Thinning: 2w zone on 4w walls.
    const auto [zone_walls, base_walls] = GENERATE(std::pair<double, double>{ 4., 2. }, std::pair<double, double>{ 2., 4. });
    const double zone_end = 20.;
    const double ramp_length = 9.;
    const double height = 20.;
    const ExPolygon island = rectangle_mm(0., 0., 60., height);
    const auto t_of_x = [&](double x) { return ramp_thickness(x - zone_end, ramp_length, zone_walls * W, base_walls * W); };

    const std::optional<WallBand> band = build_wall_band(island, [&](const Point &pt) { return WallThicknessSample{ t_of_x(unscale<double>(pt.x())), base_walls * W }; }, band_params());
    REQUIRE(band.has_value());

    // Away from the left and right sides, the contour normal on the top and bottom edges is vertical.
    for (double x = 5.; x <= 55.; x += 0.25) {
        INFO("x = " << x);
        CHECK_THAT(top_band_width(*band, x, height), WithinAbs(t_of_x(x), W / 4.));
        CHECK_THAT(first_hit_mm(band->inner, Vec2d(x, 0.), Vec2d(0., 1.)), WithinAbs(t_of_x(x), W / 4.));
    }
}

TEST_CASE("A ramp down to no walls ends the band without slivers", "[WallBand][WallBandShape]")
{
    // A 60 x 20 mm rectangle with 2w walls, and a zone with no walls at x >= 29 mm reached over a 9 mm ramp.
    const double ramp_start = 20.;
    const double ramp_length = 9.;
    const double height = 20.;
    const ExPolygon island = rectangle_mm(0., 0., 60., height);
    const WallBandParams params = band_params();
    const auto t_of_x = [&](double x) { return ramp_thickness(ramp_start + ramp_length - x, ramp_length, 0., 2. * W); };

    const std::optional<WallBand> band = build_wall_band(island, [&](const Point &pt) { return WallThicknessSample{ t_of_x(unscale<double>(pt.x())), 2. * W }; }, params);
    REQUIRE(band.has_value());

    // The band tapers while it is at least the minimum width wide.
    for (double x = 5.; t_of_x(x) >= params.min_band_width + W / 4.; x += 0.25) {
        INFO("x = " << x);
        CHECK_THAT(top_band_width(*band, x, height), WithinAbs(t_of_x(x), W / 4.));
    }
    // Past the point where t falls below the minimum width, there is no band at all: it ends in one
    // piece, and the inner region covers the rest of the island up to its contour.
    const double band_end = ramp_start + ramp_length * (1. - params.min_band_width / (2. * W));
    CHECK(band->band.size() == 1);
    const ExPolygons past_end = intersection_ex(band->band, ExPolygons{ rectangle_mm(band_end + W, -1., 61., height + 1.) });
    CHECK_THAT(area(past_end) * SCALING_FACTOR * SCALING_FACTOR, WithinAbs(0., 1e-6));
    for (const double x : { 35., 45., 55. }) {
        INFO("x = " << x);
        CHECK(strictly_inside(band->inner, Point::new_scale(x, height - 0.01)));
        CHECK(strictly_inside(band->inner, Point::new_scale(x, 0.01)));
    }
    CHECK(strictly_inside(band->inner, Point::new_scale(60. - 0.01, height / 2.)));
    CHECK_THAT(area(band->band) + area(band->inner), WithinRel(island.area(), 1e-6));
}

TEST_CASE("An island with no walls anywhere has an empty band", "[WallBand][WallBandShape]")
{
    ExPolygon island = rectangle_mm(0., 0., 60., 30.);
    island.holes.emplace_back(hole_mm(Vec2d(15., 15.), 3.25));

    const std::optional<WallBand> band = build_wall_band(island, [](const Point &) { return WallThicknessSample{ 0., 2. * W }; }, band_params());
    REQUIRE(band.has_value());
    CHECK(band->band.empty());
    REQUIRE(band->inner.size() == 1);
    CHECK_THAT(area(band->inner), WithinRel(island.area(), 1e-6));
    CHECK(band->inner.front().holes.size() == island.holes.size());
}

TEST_CASE("The band around a hole has the hole's thickness", "[WallBand][WallBandShape]")
{
    // A plate with a 6.5 mm hole and 2w walls, with a 4w zone within 10 mm of the hole's center and a 9 mm ramp.
    const Vec2d  center(15., 15.);
    const double radius = 3.25;
    ExPolygon    island = rectangle_mm(0., 0., 60., 30.);
    island.holes.emplace_back(hole_mm(center, radius));
    const auto sampler = [&](const Point &pt) { return WallThicknessSample{ ramp_thickness((unscaled(pt) - center).norm() - 10., 9., 4. * W, 2. * W), 2. * W }; };

    const std::optional<WallBand> band = build_wall_band(island, sampler, band_params());
    REQUIRE(band.has_value());

    // Measured along the hole's normals, which point away from its center.
    for (int deg = 0; deg < 360; deg += 15) {
        INFO("angle = " << deg);
        const double a = double(deg) * PI / 180.;
        CHECK_THAT(first_hit_mm(band->inner, center, Vec2d(std::cos(a), std::sin(a))) - radius, WithinAbs(4. * W, W / 4.));
    }
    // The right side is past the end of the ramp.
    CHECK_THAT(first_hit_mm(band->inner, Vec2d(60., 15.), Vec2d(-1., 0.)), WithinAbs(2. * W, W / 4.));
}

TEST_CASE("Shifting the thickness lookup moves the ramp by the shift", "[WallBand][WallBandShape]")
{
    // Brick bond: the caller shifts d by +delta or -delta, delta = rho * w / 4 with rho = 10.
    const double delta = GENERATE(10. * W / 4., - 10. * W / 4.);
    const double zone_end = 20.;
    const double ramp_length = 9.;
    const double height = 20.;
    const ExPolygon island = rectangle_mm(0., 0., 60., height);
    const auto band_for_shift = [&](double shift) {
        return build_wall_band(
            island, [&](const Point &pt) {
                return WallThicknessSample{ ramp_thickness(unscale<double>(pt.x()) - zone_end - shift, ramp_length, 4. * W, 2. * W), 2. * W };
            },
            band_params());
    };

    const std::optional<WallBand> unshifted = band_for_shift(0.);
    const std::optional<WallBand> shifted = band_for_shift(delta);
    REQUIRE(unshifted.has_value());
    REQUIRE(shifted.has_value());

    // Where the band is 3 beads wide, halfway down the ramp.
    const double crossing = top_width_crossing(*unshifted, height, 3. * W, 5., 55.);
    const double crossing_shifted = top_width_crossing(*shifted, height, 3. * W, 5., 55.);
    CHECK_THAT(crossing, WithinAbs(zone_end + ramp_length / 2., W / 4.));
    CHECK_THAT(crossing_shifted - crossing, WithinAbs(delta, W / 4.));
}

TEST_CASE("No band is built where the thickness is the base thickness everywhere", "[WallBand][WallBandShape]")
{
    ExPolygon island = rectangle_mm(0., 0., 60., 30.);
    island.holes.emplace_back(hole_mm(Vec2d(15., 15.), 3.25));
    WallBandParams params = band_params();
    params.thickness_tolerance = 0.001;
    // Off the base thickness of 2w by less than the tolerance, in either direction.
    const double base = 2. * W;
    const double t = base + GENERATE(0.5, -0.5) * params.thickness_tolerance;

    CHECK_FALSE(build_wall_band(island, [t, base](const Point &) { return WallThicknessSample{ t, base }; }, params).has_value());
}

TEST_CASE("No band is built where parts with different wall counts meet and there is no zone", "[WallBand][WallBandShape]")
{
    // Two adjacent stretches of contour: 2w walls at x < 30 mm, 3w walls from there on.
    const ExPolygon island = rectangle_mm(0., 0., 60., 30.);
    const auto      sampler = [](const Point &pt) {
        const double base = unscale<double>(pt.x()) < 30. ? 2. * W : 3. * W;
        return WallThicknessSample{ base, base };
    };

    CHECK_FALSE(build_wall_band(island, sampler, band_params()).has_value());
}

TEST_CASE("An island narrower than twice the thickness is all band", "[WallBand][WallBandShape]")
{
    // 1.5 mm wide, under 2 * 4w = 3.6 mm.
    const ExPolygon island = rectangle_mm(0., 0., 20., 1.5);

    const std::optional<WallBand> band = build_wall_band(island, [](const Point &) { return WallThicknessSample{ 4. * W, 2. * W }; }, band_params());
    REQUIRE(band.has_value());
    CHECK(band->inner.empty());
    CHECK_THAT(area(band->band), WithinRel(island.area(), 1e-6));
}

// Hidden because it is there to write SVGs of a coupon layer for eyeballing the ramp. They are
// written only when ORCA_TEST_SVG_DIR names the directory to put them in.
TEST_CASE("A coupon layer band ramps away from the zone around the hole", "[WallBand][WallBandShape][.]")
{
    // 190 x 35 mm coupon with 2w walls and a 6.5 mm hole at (15, 17.5). The zone is within R 20 of
    // the hole's center, and the ramp to the base runs over L_r = |N_zone - N0| * rho * w = 9 mm. The
    // 2D distance from the R 20 circle stands in for the geodesic distance d.
    // Reinforced: a 4w zone, drawn for the even and the odd layer of the brick bond.
    // Thinned: a zone with no walls.
    struct CouponLayer { std::string name; double zone_walls; double shift; };
    const double rho = 10.;
    const double ramp_length = 2. * rho * W;
    const double delta = rho * W / 4.;
    const CouponLayer layers[] = { { "reinforced_even", 4., delta }, { "reinforced_odd", 4., - delta }, { "no_walls", 0., 0. } };
    const Vec2d  center(15., 17.5);
    const double zone_radius = 20.;
    ExPolygon    island = rectangle_mm(0., 0., 190., 35.);
    island.holes.emplace_back(hole_mm(center, 3.25));

    const char *svg_dir = boost::nowide::getenv("ORCA_TEST_SVG_DIR");
    if (svg_dir != nullptr)
        boost::filesystem::create_directories(svg_dir);

    for (const CouponLayer &layer : layers) {
        const auto sampler = [&](const Point &pt) {
            return WallThicknessSample{ ramp_thickness((unscaled(pt) - center).norm() - zone_radius + layer.shift, ramp_length, layer.zone_walls * W, 2. * W),
                                        2. * W };
        };
        const std::optional<WallBand> band = build_wall_band(island, sampler, band_params());
        REQUIRE(band.has_value());
        // The zone's thickness next to the hole, the base thickness at the far end.
        CHECK_THAT(first_hit_mm(band->inner, Vec2d(150., 35.), Vec2d(0., -1.)), WithinAbs(2. * W, W / 4.));
        if (layer.zone_walls > 0.)
            CHECK_THAT(first_hit_mm(band->inner, Vec2d(15., 35.), Vec2d(0., -1.)), WithinAbs(layer.zone_walls * W, W / 4.));
        else
            CHECK(strictly_inside(band->inner, Point::new_scale(15., 35. - 0.01)));

        if (svg_dir == nullptr)
            continue;
        const std::string path = (boost::filesystem::path(svg_dir) / ("wall_band_coupon_" + layer.name + ".svg")).string();
        SVG svg(path, get_extents(island));
        svg.draw(band->inner, "#e8e8e8");
        svg.draw(band->band, "#f08c00");
        svg.draw_outline(island, "black", "black", scale_(0.05));
        svg.draw_outline(band->band, "#1c7ed6", "#1c7ed6", scale_(0.03));
        // The zone's edge and the end of the ramp, before the brick-bond shift.
        svg.draw_outline(hole_mm(center, zone_radius), "#2f9e44", scale_(0.03));
        svg.draw_outline(hole_mm(center, zone_radius + ramp_length), "#e03131", scale_(0.03));
        svg.Close();
        INFO("SVG written to " << path);
        CHECK(boost::filesystem::exists(path));
    }
}
