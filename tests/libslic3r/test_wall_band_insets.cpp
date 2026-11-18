#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include "libslic3r/Arachne/WallToolPaths.hpp"
#include "libslic3r/Arachne/utils/ExtrusionLine.hpp"
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/WallBand.hpp"
#include "libslic3r/libslic3r.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

using namespace Slic3r;

namespace {

// A wall through the points given in mm, closed from the last to the first.
Arachne::ExtrusionLine wall_through(std::initializer_list<std::pair<double, double>> points, bool closed = false)
{
    Arachne::ExtrusionLine line(0, false);
    for (const auto &[x, y] : points)
        line.junctions.emplace_back(Point(scaled<coord_t>(x), scaled<coord_t>(y)), scaled<coord_t>(0.4), 0);
    // Arachne closes a wall by repeating its first junction.
    if (closed)
        line.junctions.emplace_back(line.junctions.front());
    line.is_closed = closed;
    return line;
}

double wall_length(const Arachne::ExtrusionLine &line)
{
    double length = 0.;
    for (size_t i = 1; i < line.junctions.size(); ++ i)
        length += unscaled<double>((line.junctions[i].p - line.junctions[i - 1].p).cast<double>().norm());
    return length;
}

ExPolygon square_island(double size)
{
    return ExPolygon(Polygon({ Point(0, 0), Point(scaled<coord_t>(size), 0), Point(scaled<coord_t>(size), scaled<coord_t>(size)), Point(0, scaled<coord_t>(size)) }));
}

Polygon square_hole(double from, double to)
{
    Polygon hole({ Point(scaled<coord_t>(from), scaled<coord_t>(from)), Point(scaled<coord_t>(to), scaled<coord_t>(from)),
                   Point(scaled<coord_t>(to), scaled<coord_t>(to)), Point(scaled<coord_t>(from), scaled<coord_t>(to)) });
    hole.make_clockwise();
    return hole;
}

BoundingBox wall_extents(const Arachne::ExtrusionLine &line)
{
    BoundingBox box;
    for (const Arachne::ExtrusionJunction &junction : line.junctions)
        box.merge(junction.p);
    return box;
}

} // namespace

// The inset of a wall can change where it has no junction, along a segment as long as the part's side.
TEST_CASE("A wall numbered by depth is split where its depth changes inside one segment", "[WallBand]")
{
    // The outer wall runs only over x = 8..12, the wall inside it over the whole side in one segment.
    std::vector<Arachne::VariableWidthLines> perimeters(1);
    perimeters[0].emplace_back(wall_through({ { 8., 0.2 }, { 12., 0.2 } }));
    perimeters[0].emplace_back(wall_through({ { 0.2, 0.6 }, { 19.8, 0.6 } }));
    number_wall_band_insets(perimeters, { square_island(20.) }, scaled<coord_t>(0.4));

    REQUIRE(perimeters.size() == 2);
    // The inner wall takes inset 1 behind the outer wall only.
    REQUIRE(perimeters[1].size() == 1);
    CHECK_THAT(wall_length(perimeters[1].front()), Catch::Matchers::WithinAbs(4., 0.1));
    CHECK(perimeters[0].size() == 3);
}

// A piece shorter than a bead takes the inset of its neighbour, and the wall stays one wall when that leaves one inset.
TEST_CASE("A wall numbered by depth has no piece shorter than the spacing", "[WallBand]")
{
    // The outer wall covers 0.15mm of the wall inside it.
    std::vector<Arachne::VariableWidthLines> perimeters(1);
    perimeters[0].emplace_back(wall_through({ { 8., 0.2 }, { 8.15, 0.2 } }));
    perimeters[0].emplace_back(wall_through({ { 0.2, 0.6 }, { 8.05, 0.6 }, { 8.1, 0.6 }, { 8.2, 0.6 }, { 19.8, 0.6 } }));
    number_wall_band_insets(perimeters, { square_island(20.) }, scaled<coord_t>(0.4));

    REQUIRE(perimeters.size() == 1);
    REQUIRE(perimeters[0].size() == 2);
    const double longest = std::max(wall_length(perimeters[0][0]), wall_length(perimeters[0][1]));
    CHECK_THAT(longest, Catch::Matchers::WithinAbs(19.6, 0.05));
}

TEST_CASE("A closed wall whose pieces merge back to one inset stays one closed wall", "[WallBand]")
{
    // A short wall at the outline covers 0.15mm of the loop, which leaves a piece shorter than a bead of another inset.
    std::vector<Arachne::VariableWidthLines> perimeters(1);
    perimeters[0].emplace_back(wall_through({ { 8., 0.2 }, { 8.15, 0.2 } }));
    perimeters[0].emplace_back(wall_through({ { 0.6, 0.6 }, { 8.05, 0.6 }, { 8.1, 0.6 }, { 8.2, 0.6 }, { 19.4, 0.6 }, { 19.4, 19.4 }, { 0.6, 19.4 } }, true));
    number_wall_band_insets(perimeters, { square_island(20.) }, scaled<coord_t>(0.4));

    REQUIRE(perimeters.size() == 1);
    size_t closed = 0;
    for (const Arachne::ExtrusionLine &line : perimeters[0])
        if (line.is_closed) {
            ++ closed;
            CHECK(line.junctions.front().p == line.junctions.back().p);
        }
    CHECK(closed == 1);
    CHECK(perimeters[0].size() == 2);
}

// The loops of a band island with a hole follow its contour or its hole, whatever their start and direction.
TEST_CASE("A closed wall numbered by depth runs as a contour or a hole by the outline most of it follows", "[WallBand]")
{
    ExPolygon island = square_island(20.);
    island.holes.emplace_back(square_hole(7., 13.));

    // Counterclockwise around the contour and clockwise around the hole, as Arachne does not run them.
    std::vector<Arachne::VariableWidthLines> perimeters(2);
    perimeters[0].emplace_back(wall_through({ { 0.2, 0.2 }, { 19.8, 0.2 }, { 19.8, 19.8 }, { 0.2, 19.8 } }, true));
    perimeters[0].emplace_back(wall_through({ { 6.8, 6.8 }, { 6.8, 13.2 }, { 13.2, 13.2 }, { 13.2, 6.8 } }, true));
    // A wall along the contour that starts on a dent towards the hole, nearest the hole there.
    perimeters[1].emplace_back(wall_through({ { 10., 5.5 }, { 18., 2. }, { 18., 18. }, { 2., 18. }, { 2., 2. } }, true));
    number_wall_band_insets(perimeters, { island }, scaled<coord_t>(0.4));

    REQUIRE(perimeters.size() == 2);
    REQUIRE(perimeters[0].size() == 2);
    REQUIRE(perimeters[1].size() == 1);
    for (const Arachne::ExtrusionLine &line : perimeters[0]) {
        REQUIRE(line.is_closed);
        // The loop along the contour is the larger one.
        if (wall_extents(line).size().x() > scaled<coord_t>(10.))
            CHECK(line.is_contour());
        else
            CHECK_FALSE(line.is_contour());
    }
    CHECK(perimeters[1].front().is_closed);
    CHECK(perimeters[1].front().is_contour());
}

// A few long sides outweigh the many junctions of a short stretch around a round hole.
TEST_CASE("A closed wall running close to a faceted hole for a short stretch stays a contour", "[WallBand]")
{
    constexpr double pi = 3.14159265358979323846;
    ExPolygon island = square_island(20.);
    Polygon   hole;
    for (int i = 0; i < 64; ++ i)
        hole.points.emplace_back(scaled<coord_t>(6. + 3. * std::cos(2. * pi * i / 64.)), scaled<coord_t>(10. + 3. * std::sin(2. * pi * i / 64.)));
    hole.make_clockwise();
    island.holes.emplace_back(std::move(hole));

    // Along the square's sides, with a dent of 61 junctions on an arc 1mm from the hole, which is near the left side.
    Arachne::ExtrusionLine wall = wall_through({ { 2., 2. }, { 18., 2. }, { 18., 18. }, { 2., 18. } });
    for (int a = 150; a <= 210; ++ a)
        wall.junctions.emplace_back(Point(scaled<coord_t>(6. + 4. * std::cos(a * pi / 180.)), scaled<coord_t>(10. + 4. * std::sin(a * pi / 180.))), scaled<coord_t>(0.4), 0);
    wall.junctions.emplace_back(wall.junctions.front());
    wall.is_closed = true;

    std::vector<Arachne::VariableWidthLines> perimeters(1);
    perimeters[0].emplace_back(std::move(wall));
    number_wall_band_insets(perimeters, { island }, scaled<coord_t>(0.4));

    REQUIRE(perimeters.size() == 1);
    REQUIRE(perimeters[0].size() == 1);
    CHECK(perimeters[0].front().is_closed);
    CHECK(perimeters[0].front().is_contour());
}
