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
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace Slic3r;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace {

// Bead width w in mm.
constexpr double W = 0.45;
// Thinnest band in mm, standing in for the region's min_bead_width.
constexpr double MIN_BAND_WIDTH = W / 2.;
// Tolerance in mm for band widths measured on straight edges.
constexpr double WIDTH_TOLERANCE = 0.01;

// Samples every w / 4, band at least MIN_BAND_WIDTH wide.
WallBandParams band_params() { return WallBandParams::for_bead_width(W, MIN_BAND_WIDTH); }

using FieldMm = std::function<double(const Vec2d &pt_mm)>;

// Looks up t and the base per point, from functions of the point in mm.
WallThicknessSampler sampler(FieldMm thickness, FieldMm base)
{
    return [thickness = std::move(thickness), base = std::move(base)](const Points &pts, const std::vector<Vec2d> &) {
        std::vector<WallThicknessSample> out;
        out.reserve(pts.size());
        for (const Point &pt : pts)
            out.push_back({ thickness(unscaled(pt)), base(unscaled(pt)) });
        return out;
    };
}

WallThicknessSampler sampler(FieldMm thickness, double base)
{
    return sampler(std::move(thickness), [base](const Vec2d &) { return base; });
}

ExPolygon rectangle_mm(double x0, double y0, double x1, double y1)
{
    ExPolygon out;
    out.contour.points = { Point::new_scale(x0, y0), Point::new_scale(x1, y0), Point::new_scale(x1, y1), Point::new_scale(x0, y1) };
    return out;
}

// A circle with 64 vertices, counter-clockwise as a contour or clockwise as a hole.
Polygon circle_mm(const Vec2d &center, double radius, bool hole)
{
    constexpr size_t segments = 64;
    Polygon out;
    for (size_t i = 0; i < segments; ++ i) {
        const double a = (hole ? -2. : 2.) * PI * double(i) / double(segments);
        out.points.emplace_back(Point::new_scale(center.x() + radius * std::cos(a), center.y() + radius * std::sin(a)));
    }
    return out;
}

Polygon hole_mm(const Vec2d &center, double radius) { return circle_mm(center, radius, true); }

// The plan's linear t(d): zone_t inside the zone (d <= 0), base_t past the ramp (d >= ramp_length).
double ramp_thickness(double d, double ramp_length, double zone_t, double base_t)
{
    return zone_t + (base_t - zone_t) * std::clamp(d / ramp_length, 0., 1.);
}

double area_mm2(const ExPolygons &expolys) { return area(expolys) * SCALING_FACTOR * SCALING_FACTOR; }

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

bool strictly_inside(const ExPolygons &regions, const Vec2d &pt_mm)
{
    const Point pt = Point::new_scale(pt_mm.x(), pt_mm.y());
    return std::any_of(regions.begin(), regions.end(), [&pt](const ExPolygon &expoly) { return expoly.contains(pt, false); });
}

double smallest_area_mm2(const ExPolygons &expolys)
{
    double out = std::numeric_limits<double>::max();
    for (const ExPolygon &expoly : expolys)
        out = std::min(out, expoly.area() * SCALING_FACTOR * SCALING_FACTOR);
    return out;
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

// The band stays inside the island, and no piece of the inner region is too small to fill.
void check_band_and_inner(const WallBand &band, const ExPolygon &island)
{
    CHECK_THAT(area_mm2(diff_ex(band.band, ExPolygons{ island })), WithinAbs(0., 1e-3));
    if (! band.inner.empty())
        CHECK(smallest_area_mm2(band.inner) >= MIN_BAND_WIDTH * MIN_BAND_WIDTH);
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
    const std::optional<WallBand> band = build_wall_band(island, sampler([t](const Vec2d &) { return t; }, 2. * W), band_params());
    REQUIRE(band.has_value());

    const ExPolygons stock_band = diff_ex(island, offset_ex(island, - float(scale_(t))));
    CHECK_THAT(area(band->band), WithinRel(area(stock_band), 0.01));
    check_band_and_inner(*band, island);
}

TEST_CASE("The sampler gets the normals pointing out of the material", "[WallBand][WallBandShape]")
{
    const Vec2d center(15., 15.);
    ExPolygon   island = rectangle_mm(0., 0., 60., 30.);
    island.holes.emplace_back(hole_mm(center, 3.25));
    std::vector<std::pair<Points, std::vector<Vec2d>>> calls;
    const WallThicknessSampler recorder = [&calls](const Points &pts, const std::vector<Vec2d> &normals) {
        calls.emplace_back(pts, normals);
        return std::vector<WallThicknessSample>(pts.size(), WallThicknessSample{ 3. * W, 2. * W });
    };

    REQUIRE(build_wall_band(island, recorder, band_params()).has_value());
    REQUIRE(calls.size() == 2);
    for (size_t i = 0; i < calls[0].first.size(); ++ i) {
        const Vec2d pt = unscaled(calls[0].first[i]);
        // A sample in the middle of the top edge.
        if (std::abs(pt.y() - 30.) < 1e-6 && pt.x() > 1. && pt.x() < 59.) {
            CHECK_THAT(calls[0].second[i].x(), WithinAbs(0., 1e-6));
            CHECK_THAT(calls[0].second[i].y(), WithinAbs(1., 1e-6));
        }
    }
    // On the hole, the normals point to its center.
    for (size_t i = 0; i < calls[1].first.size(); ++ i) {
        const Vec2d to_center = (center - unscaled(calls[1].first[i])).normalized();
        CHECK_THAT(calls[1].second[i].dot(to_center), WithinAbs(1., 1e-3));
    }
}

TEST_CASE("The band width follows a thickness that ramps along the contour", "[WallBand][WallBandShape]")
{
    // A 60 x 20 mm rectangle, zone at x <= 20 mm, ramp over 9 mm. Thickening: 4w zone on 2w walls.
    // Thinning: 2w zone on 4w walls.
    const std::pair<double, double> walls = GENERATE(std::pair<double, double>{ 4., 2. }, std::pair<double, double>{ 2., 4. });
    const double zone_walls = walls.first;
    const double base_walls = walls.second;
    const double zone_end = 20.;
    const double ramp_length = 9.;
    const double height = 20.;
    const ExPolygon island = rectangle_mm(0., 0., 60., height);
    const auto t_of_x = [=](double x) { return ramp_thickness(x - zone_end, ramp_length, zone_walls * W, base_walls * W); };

    const std::optional<WallBand> band = build_wall_band(island, sampler([&](const Vec2d &pt) { return t_of_x(pt.x()); }, base_walls * W), band_params());
    REQUIRE(band.has_value());

    // Away from the left and right sides, the contour normal on the top and bottom edges is vertical.
    for (double x = 5.; x <= 55.; x += 0.25) {
        INFO("x = " << x);
        CHECK_THAT(top_band_width(*band, x, height), WithinAbs(t_of_x(x), WIDTH_TOLERANCE));
        CHECK_THAT(first_hit_mm(band->inner, Vec2d(x, 0.), Vec2d(0., 1.)), WithinAbs(t_of_x(x), WIDTH_TOLERANCE));
        CHECK(strictly_inside(band->band, Vec2d(x, height - 0.01)));
    }
    check_band_and_inner(*band, island);
}

TEST_CASE("A step in thickness between neighbouring samples gives a step in the band", "[WallBand][WallBandShape]")
{
    // 4w walls left of x = 30 mm, 2w walls from there on, on a base of 2w.
    const double height = 20.;
    const ExPolygon island = rectangle_mm(0., 0., 60., height);
    const auto t_of_x = [](double x) { return x < 30. ? 4. * W : 2. * W; };

    const std::optional<WallBand> band = build_wall_band(island, sampler([&](const Vec2d &pt) { return t_of_x(pt.x()); }, 2. * W), band_params());
    REQUIRE(band.has_value());

    // The step spans at most one sample spacing.
    for (const double x : { 5., 20., 29.5, 30.5, 40., 55. }) {
        INFO("x = " << x);
        CHECK_THAT(top_band_width(*band, x, height), WithinAbs(t_of_x(x), WIDTH_TOLERANCE));
    }
    CHECK(band->band.size() == 1);
    CHECK(band->inner.size() == 1);
    check_band_and_inner(*band, island);
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

    const std::optional<WallBand> band = build_wall_band(island, sampler([&](const Vec2d &pt) { return t_of_x(pt.x()); }, 2. * W), params);
    REQUIRE(band.has_value());

    // The band follows t while t is at least the minimum width.
    const double band_end = ramp_start + ramp_length * (1. - params.min_band_width / (2. * W));
    for (double x = 5.; x < band_end - params.sample_spacing; x += 0.25) {
        INFO("x = " << x);
        CHECK_THAT(top_band_width(*band, x, height), WithinAbs(t_of_x(x), WIDTH_TOLERANCE));
        CHECK(strictly_inside(band->band, Vec2d(x, height - 0.01)));
    }
    // It ends in one piece within a sample spacing of where t falls below the minimum width, and the
    // inner region covers the rest of the island up to its contour.
    CHECK(band->band.size() == 1);
    const ExPolygons past_end = intersection_ex(band->band, ExPolygons{ rectangle_mm(band_end + params.sample_spacing, -1., 61., height + 1.) });
    CHECK_THAT(area_mm2(past_end), WithinAbs(0., 1e-9));
    for (const double x : { 35., 45., 55. }) {
        INFO("x = " << x);
        CHECK(strictly_inside(band->inner, Vec2d(x, height - 0.01)));
        CHECK(strictly_inside(band->inner, Vec2d(x, 0.01)));
    }
    CHECK(strictly_inside(band->inner, Vec2d(60. - 0.01, height / 2.)));
    check_band_and_inner(*band, island);
}

TEST_CASE("An island with no walls anywhere has an empty band", "[WallBand][WallBandShape]")
{
    ExPolygon island = rectangle_mm(0., 0., 60., 30.);
    island.holes.emplace_back(hole_mm(Vec2d(15., 15.), 3.25));

    const std::optional<WallBand> band = build_wall_band(island, sampler([](const Vec2d &) { return 0.; }, 2. * W), band_params());
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
    const FieldMm t = [&](const Vec2d &pt) { return ramp_thickness((pt - center).norm() - 10., 9., 4. * W, 2. * W); };

    const std::optional<WallBand> band = build_wall_band(island, sampler(t, 2. * W), band_params());
    REQUIRE(band.has_value());

    // Measured along the hole's normals, which point away from its center. The hole is a 64-gon.
    for (int deg = 0; deg < 360; deg += 15) {
        INFO("angle = " << deg);
        const double a = double(deg) * PI / 180.;
        CHECK_THAT(first_hit_mm(band->inner, center, Vec2d(std::cos(a), std::sin(a))) - radius, WithinAbs(4. * W, WIDTH_TOLERANCE));
    }
    // The right side is past the end of the ramp.
    CHECK_THAT(first_hit_mm(band->inner, Vec2d(60., 15.), Vec2d(-1., 0.)), WithinAbs(2. * W, WIDTH_TOLERANCE));
    check_band_and_inner(*band, island);
}

TEST_CASE("The band around a hole follows a thickness that varies around it", "[WallBand][WallBandShape]")
{
    // t goes from 2w on the hole's left side to 4w on its right side, on a base of 2w.
    const Vec2d  center(15., 15.);
    const double radius = 3.25;
    ExPolygon    island = rectangle_mm(0., 0., 60., 30.);
    island.holes.emplace_back(hole_mm(center, radius));
    const FieldMm t = [&](const Vec2d &pt) { return ramp_thickness(center.x() + radius - pt.x(), 2. * radius, 4. * W, 2. * W); };

    const std::optional<WallBand> band = build_wall_band(island, sampler(t, 2. * W), band_params());
    REQUIRE(band.has_value());

    for (int deg = 0; deg < 360; deg += 30) {
        INFO("angle = " << deg);
        const Vec2d dir(std::cos(double(deg) * PI / 180.), std::sin(double(deg) * PI / 180.));
        CHECK_THAT(first_hit_mm(band->inner, center, dir) - radius, WithinAbs(t(center + radius * dir), 2. * WIDTH_TOLERANCE));
    }
    check_band_and_inner(*band, island);
}

TEST_CASE("Bands of a hole and of the contour merge where they are close", "[WallBand][WallBandShape]")
{
    // The hole's edge is 3 mm from the left side, under the 2 * 4w = 3.6 mm of the two bands.
    const Vec2d center(5., 10.);
    ExPolygon   island = rectangle_mm(0., 0., 30., 20.);
    island.holes.emplace_back(hole_mm(center, 2.));

    const std::optional<WallBand> band = build_wall_band(island, sampler([](const Vec2d &) { return 4. * W; }, 2. * W), band_params());
    REQUIRE(band.has_value());
    CHECK(strictly_inside(band->band, Vec2d(1.5, 10.)));
    CHECK(band->band.size() == 1);
    // The inner region goes around the hole, so it has no hole of its own.
    REQUIRE(band->inner.size() == 1);
    CHECK(band->inner.front().holes.empty());
    check_band_and_inner(*band, island);
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
        return build_wall_band(island, sampler([=](const Vec2d &pt) { return ramp_thickness(pt.x() - zone_end - shift, ramp_length, 4. * W, 2. * W); }, 2. * W),
                               band_params());
    };

    const std::optional<WallBand> unshifted = band_for_shift(0.);
    const std::optional<WallBand> shifted = band_for_shift(delta);
    REQUIRE(unshifted.has_value());
    REQUIRE(shifted.has_value());

    // Where the band is 3 beads wide, halfway down the ramp.
    const double crossing = top_width_crossing(*unshifted, height, 3. * W, 5., 55.);
    const double crossing_shifted = top_width_crossing(*shifted, height, 3. * W, 5., 55.);
    CHECK_THAT(crossing, WithinAbs(zone_end + ramp_length / 2., WIDTH_TOLERANCE));
    CHECK_THAT(crossing_shifted - crossing, WithinAbs(delta, WIDTH_TOLERANCE));
}

TEST_CASE("The band follows a ramp along a curved contour", "[WallBand][WallBandShape]")
{
    // A disc of radius 10 mm with t ramping from 4w at x = -5 mm to 2w at x = 4 mm, on a base of 2w.
    ExPolygon island;
    island.contour = circle_mm(Vec2d(0., 0.), 10., false);
    const FieldMm t = [](const Vec2d &pt) { return ramp_thickness(pt.x() + 5., 9., 4. * W, 2. * W); };

    const std::optional<WallBand> band = build_wall_band(island, sampler(t, 2. * W), band_params());
    REQUIRE(band.has_value());
    CHECK(band->band.size() == 1);
    REQUIRE(band->inner.size() == 1);
    CHECK(band->inner.front().holes.empty());
    // Measured along the radial normals; the contour is a 64-gon.
    for (int deg = 0; deg < 360; deg += 20) {
        INFO("angle = " << deg);
        const Vec2d dir(std::cos(double(deg) * PI / 180.), std::sin(double(deg) * PI / 180.));
        CHECK_THAT(10. - first_hit_mm(band->inner, Vec2d(0., 0.), dir), WithinAbs(t(10. * dir), 2. * WIDTH_TOLERANCE));
    }
    check_band_and_inner(*band, island);
}

TEST_CASE("A sharp convex corner is all band up to its tip", "[WallBand][WallBandShape]")
{
    // An isosceles triangle with a 20 degree tip at the origin, 30 mm long.
    const double half_angle = 10. * PI / 180.;
    const double length = 30.;
    ExPolygon    island;
    island.contour.points = { Point::new_scale(0., 0.), Point::new_scale(length, - length * std::tan(half_angle)),
                              Point::new_scale(length, length * std::tan(half_angle)) };

    const std::optional<WallBand> band = build_wall_band(island, sampler([](const Vec2d &) { return 3. * W; }, 2. * W), band_params());
    REQUIRE(band.has_value());
    // Near the tip the island is narrower than 2t, so it is all band, up to the tip.
    for (const double x : { 0.05, 0.5, 2. }) {
        INFO("x = " << x);
        CHECK(strictly_inside(band->band, Vec2d(x, 0.)));
        CHECK_FALSE(strictly_inside(band->inner, Vec2d(x, 0.)));
    }
    CHECK(band->band.size() == 1);
    CHECK(band->inner.size() == 1);
    check_band_and_inner(*band, island);
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

    CHECK_FALSE(build_wall_band(island, sampler([t](const Vec2d &) { return t; }, base), params).has_value());
}

TEST_CASE("No band is built where parts with different wall counts meet and there is no zone", "[WallBand][WallBandShape]")
{
    // Two adjacent stretches of contour: 2w walls at x < 30 mm, 3w walls from there on.
    const ExPolygon island = rectangle_mm(0., 0., 60., 30.);
    const FieldMm   base = [](const Vec2d &pt) { return pt.x() < 30. ? 2. * W : 3. * W; };

    CHECK_FALSE(build_wall_band(island, sampler(base, base), band_params()).has_value());
}

TEST_CASE("An island narrower than twice the thickness is all band", "[WallBand][WallBandShape]")
{
    // 1.5 mm wide, under 2 * 4w = 3.6 mm; and 0.2 mm wide, under the minimum band width as well.
    const double width = GENERATE(1.5, 0.2);
    const ExPolygon island = rectangle_mm(0., 0., 20., width);

    const std::optional<WallBand> band = build_wall_band(island, sampler([](const Vec2d &) { return 4. * W; }, 2. * W), band_params());
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
        const FieldMm t = [&](const Vec2d &pt) {
            return ramp_thickness((pt - center).norm() - zone_radius + layer.shift, ramp_length, layer.zone_walls * W, 2. * W);
        };
        const std::optional<WallBand> band = build_wall_band(island, sampler(t, 2. * W), band_params());
        REQUIRE(band.has_value());
        // The zone's thickness next to the hole, the base thickness at the far end.
        CHECK_THAT(first_hit_mm(band->inner, Vec2d(150., 35.), Vec2d(0., -1.)), WithinAbs(2. * W, WIDTH_TOLERANCE));
        if (layer.zone_walls > 0.)
            CHECK_THAT(first_hit_mm(band->inner, Vec2d(15., 35.), Vec2d(0., -1.)), WithinAbs(layer.zone_walls * W, WIDTH_TOLERANCE));
        else
            CHECK(strictly_inside(band->inner, Vec2d(15., 35. - 0.01)));
        check_band_and_inner(*band, island);

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
