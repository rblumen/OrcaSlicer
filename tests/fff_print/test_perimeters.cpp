#include <catch2/catch_all.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "libslic3r/ExtrusionEntity.hpp"
#include "libslic3r/ExtrusionEntityCollection.hpp"
#include "libslic3r/Layer.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/GCodeReader.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/TriangleMesh.hpp"

#include <algorithm>
#include <cmath>
#include "libslic3r/PrintConfig.hpp"
#include <cstddef>
#include "libslic3r/libslic3r.h"
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/Polyline.hpp"
#include "libslic3r/Config.hpp"
#include "libslic3r/PrintBase.hpp"
#include "libslic3r/TriangleSelector.hpp"
#include "libslic3r/Arachne/utils/ExtrusionJunction.hpp"
#include "libslic3r/Arachne/utils/ExtrusionLine.hpp"
#include "libslic3r/WallBand.hpp"
#include <functional>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "test_helpers.hpp"

using namespace Slic3r;
using namespace Slic3r::Test;

namespace {

// The layer at this Z is the last one of the base, so its top surface is the ledge.
const double ledge_z = 5.0;

// The first layer, at initial_layer_print_height.
const double first_layer_z = 0.2;

// TestMesh::step scaled 3x in X/Y: a 60x60x5 base carrying a 54x54 column up to z=10, leaving a 3mm
// top ledge around a feature that keeps rising. That is the geometry both only_one_wall_top and the
// top surface expansion act on. The ledge has to stay wider than the wall band plus two top-infill
// lines, or the expansion discards it as a sliver and the tests below assert nothing.
TriangleMesh step_with_ledge()
{
    TriangleMesh m = Slic3r::Test::mesh(TestMesh::step);
    m.scale(Vec3f(3.f, 3.f, 1.f));
    return m;
}

// Every setting the assertions depend on, so none of them rests on a default.
DynamicPrintConfig base_config(const char *wall_generator)
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    config.set_deserialize_strict({
        { "wall_generator",             wall_generator },
        { "layer_height",               0.2 },  // puts a layer boundary exactly on ledge_z
        { "initial_layer_print_height", 0.2 },
        { "wall_loops",                 3 },
        { "sparse_infill_density",      "15%" },
        { "top_shell_layers",           3 },
        { "bottom_shell_layers",        3 },
        { "top_surface_density",        "100%" },
        { "top_surface_expansion",      0.0 },
        { "only_one_wall_top",          false },
        { "only_one_wall_first_layer",  false },
        // Do not let the one-wall threshold discard the 3mm ledge before the feature sees it.
        { "min_width_top_surface",      0.0 },
    });
    return config;
}

double collection_length(const ExtrusionEntityCollection &coll)
{
    double len = 0.;
    for (const ExtrusionEntity *entity : coll.flatten().entities)
        if (! entity->is_collection())
            len += entity->length();
    return len;
}

// Extruded length per layer. Two slices are compared through this rather than through their G-code,
// because the G-code carries a config block that differs whenever any setting differs.
struct SliceLengths {
    std::vector<double> perimeters;
    std::vector<double> fills;
};

SliceLengths slice_lengths(const Print &print)
{
    SliceLengths out;
    for (const Layer *layer : print.objects().front()->layers()) {
        double perimeters = 0., fills = 0.;
        for (const LayerRegion *region : layer->regions()) {
            perimeters += collection_length(region->perimeters);
            fills      += collection_length(region->fills);
        }
        out.perimeters.push_back(perimeters);
        out.fills.push_back(fills);
    }
    return out;
}

double perimeter_length_at(const Print &print, double print_z)
{
    for (const Layer *layer : print.objects().front()->layers())
        if (std::abs(layer->print_z - print_z) < 1e-4) {
            double len = 0.;
            for (const LayerRegion *region : layer->regions())
                len += collection_length(region->perimeters);
            return len;
        }
    return 0.;
}

// Largest per-layer difference between two series; a negative result means they are not comparable.
double max_difference(const std::vector<double> &a, const std::vector<double> &b)
{
    if (a.size() != b.size() || a.empty())
        return -1.;
    double worst = 0.;
    for (size_t i = 0; i < a.size(); ++ i)
        worst = std::max(worst, std::abs(a[i] - b[i]));
    return worst;
}

} // namespace

// The expansion only retypes area as top solid infill, so it can do nothing where there is no top
// fill to begin with: zero top shell layers retypes the top surfaces as internal, and a top surface
// density of 0% leaves the top layer with walls only. The last section is the control - the same
// expansion on the same model does change the slice once a top fill exists - without which the two
// equality checks above it would hold for an unrelated reason.
TEST_CASE("Top surface expansion only acts where there is a top fill", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    auto lengths_for = [wall_generator](int top_shell_layers, const char *top_surface_density, double expansion) {
        DynamicPrintConfig config = base_config(wall_generator);
        config.set_deserialize_strict({
            { "top_shell_layers",      top_shell_layers },
            { "top_surface_density",   top_surface_density },
            { "top_surface_expansion", expansion },
        });
        Print print;
        init_and_process_print({ step_with_ledge() }, print, config);
        REQUIRE_FALSE(print.objects().empty());
        return slice_lengths(print);
    };

    SECTION("no top shell layers") {
        const SliceLengths off = lengths_for(0, "100%", 0.0);
        const SliceLengths on  = lengths_for(0, "100%", 2.0);
        REQUIRE(off.perimeters.size() == on.perimeters.size());
        CHECK_THAT(max_difference(off.perimeters, on.perimeters), Catch::Matchers::WithinAbs(0., 1.0));
        CHECK_THAT(max_difference(off.fills,      on.fills),      Catch::Matchers::WithinAbs(0., 1.0));
    }

    SECTION("zero top surface density") {
        const SliceLengths off = lengths_for(3, "0%", 0.0);
        const SliceLengths on  = lengths_for(3, "0%", 2.0);
        REQUIRE(off.perimeters.size() == on.perimeters.size());
        CHECK_THAT(max_difference(off.perimeters, on.perimeters), Catch::Matchers::WithinAbs(0., 1.0));
        CHECK_THAT(max_difference(off.fills,      on.fills),      Catch::Matchers::WithinAbs(0., 1.0));
    }

    SECTION("with a top fill the same expansion does change the slice") {
        const SliceLengths off = lengths_for(3, "100%", 0.0);
        const SliceLengths on  = lengths_for(3, "100%", 2.0);
        REQUIRE(off.fills.size() == on.fills.size());
        CHECK(max_difference(off.fills, on.fills) > scale_(0.5));
    }
}

// With no top shell the top surfaces are retyped as internal, so the top surface density has nothing
// left to control: there is no top fill, and only_one_wall_top - the one route from the density to the
// perimeters - is itself switched off for want of a top surface to act on.
TEST_CASE("Top surface density does not affect a slice without a top shell", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    auto lengths_for = [wall_generator](const char *top_surface_density) {
        DynamicPrintConfig config = base_config(wall_generator);
        config.set_deserialize_strict({
            { "top_shell_layers",    0 },
            { "only_one_wall_top",   true },
            { "top_surface_density", top_surface_density },
        });
        Print print;
        init_and_process_print({ step_with_ledge() }, print, config);
        REQUIRE_FALSE(print.objects().empty());
        return slice_lengths(print);
    };

    const SliceLengths solid = lengths_for("100%");
    const SliceLengths none  = lengths_for("0%");
    REQUIRE(solid.perimeters.size() == none.perimeters.size());
    CHECK_THAT(max_difference(solid.perimeters, none.perimeters), Catch::Matchers::WithinAbs(0., 1.0));
    CHECK_THAT(max_difference(solid.fills,      none.fills),      Catch::Matchers::WithinAbs(0., 1.0));
}

// On the ledge layer the inner walls are given up to the top fill, so that layer loses wall length.
// The handover needs a top fill that reaches the freed space: at a top surface density of 0% there is
// no top fill at all, and without top_surface_expansion the fill never grows over the walls. Either
// way the feature still runs, through the original generation, which keeps the inner walls up to the
// top boundary - putting that layer back between the plain and the one-wall slice.
TEST_CASE("Only one wall on top surfaces drops inner walls only where a top fill replaces them", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    auto ledge_perimeters_for = [wall_generator](bool only_one_wall_top, const char *top_surface_density, double expansion) {
        DynamicPrintConfig config = base_config(wall_generator);
        config.set_deserialize_strict({
            { "only_one_wall_top",     only_one_wall_top },
            { "top_surface_density",   top_surface_density },
            { "top_surface_expansion", expansion },
        });
        Print print;
        init_and_process_print({ step_with_ledge() }, print, config);
        REQUIRE_FALSE(print.objects().empty());
        return perimeter_length_at(print, ledge_z);
    };

    const double plain              = ledge_perimeters_for(false, "100%", 2.0);
    const double one_wall           = ledge_perimeters_for(true,  "100%", 2.0);
    const double one_wall_no_fill   = ledge_perimeters_for(true,  "0%",   2.0);
    const double one_wall_no_expand = ledge_perimeters_for(true,  "100%", 0.0);

    REQUIRE(plain > 0.);
    CHECK(one_wall < plain);
    // Both fall back to the original generation, which cuts the walls back to the top boundary but not past it.
    CHECK(one_wall_no_fill > one_wall);
    CHECK(one_wall_no_fill < plain);
    CHECK(one_wall_no_expand > one_wall);
    CHECK(one_wall_no_expand < plain);
}

// The bottom counterpart: the first layer is thinned to a single wall only where a bottom shell fills the
// space behind it. With no bottom shell layers the bottom surfaces are retyped as internal, so that wall
// would ring sparse infill on the bed - the option is switched off instead, and the GUI hides it in that
// state so a profile that left it enabled cannot act behind a hidden checkbox.
TEST_CASE("Only one wall on the first layer needs a bottom shell", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    auto first_layer_perimeters_for = [wall_generator](bool only_one_wall_first_layer, int bottom_shell_layers) {
        DynamicPrintConfig config = base_config(wall_generator);
        config.set_deserialize_strict({
            { "only_one_wall_first_layer", only_one_wall_first_layer },
            { "bottom_shell_layers",       bottom_shell_layers },
        });
        Print print;
        init_and_process_print({ step_with_ledge() }, print, config);
        REQUIRE_FALSE(print.objects().empty());
        return perimeter_length_at(print, first_layer_z);
    };

    const double plain             = first_layer_perimeters_for(false, 3);
    const double one_wall          = first_layer_perimeters_for(true,  3);
    // Both at zero bottom shell layers, so everything else that setting changes cancels out between them.
    const double plain_no_shell    = first_layer_perimeters_for(false, 0);
    const double one_wall_no_shell = first_layer_perimeters_for(true,  0);

    REQUIRE(plain > 0.);
    CHECK(one_wall < plain);
    // No bottom shell: the option is inert, down to the same walls an unchecked box gives.
    CHECK_THAT(one_wall_no_shell, Catch::Matchers::WithinAbs(plain_no_shell, 1.0));
}

namespace {

// The last layer of the tab, whose top surface shares an island with the tube walls rising past it.
const double tab_top_z = 5.0;

// With the widths below the tube walls are 1.10mm wide once the precise outer wall offset (0.043mm a side) is
// taken off. That is narrower than 3 outer wall spacings (3 x 0.377 = 1.131mm), so an Arachne pass limited to a
// single wall fills it by widening its 2 beads, yet wide enough for the full 2 wall pass to add a middle wall
// (from 1.062mm).
const double narrow_wall = 1.186;

// A 20x30x10 tube with narrow_wall thick walls, and a 20x8x5 tab against its -Y side.
Print &tube_with_tab(Print &print, Model &model, const DynamicPrintConfig &config)
{
    ModelObject *object = model.add_object();
    object->name = "tube_with_tab.stl";
    object->add_volume(make_cube(20., 30., 10.), ModelVolumeType::MODEL_PART, false);
    // Overlaps the tube wall by 0.5mm so the two parts slice as one island.
    TriangleMesh tab = make_cube(20., 8.5, 5.);
    tab.translate(0.f, -8.f, 0.f);
    object->add_volume(std::move(tab), ModelVolumeType::MODEL_PART, false);
    TriangleMesh bore = make_cube(20. - 2. * narrow_wall, 30. - 2. * narrow_wall, 12.);
    bore.translate(float(narrow_wall), float(narrow_wall), -1.f);
    object->add_volume(std::move(bore), ModelVolumeType::NEGATIVE_VOLUME, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    return print;
}

// Every width the narrow_wall arithmetic depends on, so none of them rests on a default.
DynamicPrintConfig narrow_wall_config(bool only_one_wall_top, double top_surface_expansion)
{
    DynamicPrintConfig config = base_config("arachne");
    config.set_deserialize_strict({
        { "wall_loops",            2 },
        { "nozzle_diameter",       "0.4" },
        { "line_width",            0.42 },
        { "outer_wall_line_width", 0.42 },
        { "inner_wall_line_width", 0.45 },
        { "min_bead_width",        "85%" },
        { "precise_outer_wall",    true },
        { "wall_sequence",         "inner wall/outer wall" },
        { "only_one_wall_top",     only_one_wall_top },
        { "top_surface_expansion", top_surface_expansion },
    });
    return config;
}

// Inner wall length the layer at print_z extrudes within 3mm of its +Y edge: the tube wall facing away from the tab.
double far_wall_inner_wall_length(const Print &print, double print_z)
{
    for (const Layer *layer : print.objects().front()->layers()) {
        if (std::abs(layer->print_z - print_z) > EPSILON)
            continue;
        BoundingBox band = get_extents(layer->lslices);
        band.min.y()     = band.max.y() - scaled<coord_t>(3.);

        Polylines inner_walls;
        auto      collect = [&inner_walls](const ExtrusionPaths &paths) {
            for (const ExtrusionPath &path : paths)
                if (path.role() == erPerimeter)
                    inner_walls.emplace_back(path.as_polyline());
        };
        for (const LayerRegion *region : layer->regions()) {
            const ExtrusionEntityCollection walls = region->perimeters.flatten();
            for (const ExtrusionEntity *entity : walls.entities) {
                if (const auto *loop = dynamic_cast<const ExtrusionLoop*>(entity))
                    collect(loop->paths);
                else if (const auto *multi_path = dynamic_cast<const ExtrusionMultiPath*>(entity))
                    collect(multi_path->paths);
                else if (const auto *path = dynamic_cast<const ExtrusionPath*>(entity))
                    collect({ *path });
            }
        }
        return unscaled<double>(total_length(intersection_pl(inner_walls, band.polygon())));
    }
    return 0.;
}

} // namespace

// only_one_wall_top first lays out an island with a single Arachne wall and generates the inner walls inside it.
// On a wall narrower than 3 outer wall spacings that single wall pass widens its 2 beads to fill the wall and leaves
// no room for the middle wall, which is only intended over the top surface. The tube walls away from the tab are not
// under the tab's top surface, so on the tab's last layer they keep the inner wall they get with the option off.
TEST_CASE("Only one wall on top surfaces keeps the inner walls of narrow walls away from the top surface", "[Perimeters]")
{
    // 0 re-onions the region beside the top surface, 2 clips the inner walls over it.
    const double top_surface_expansion = GENERATE(0.0, 2.0);
    CAPTURE(top_surface_expansion);

    struct TabTopLayer {
        double perimeters;
        double far_wall_inner_walls;
    };
    auto tab_top_layer_for = [top_surface_expansion](bool only_one_wall_top) {
        Print print;
        Model model;
        tube_with_tab(print, model, narrow_wall_config(only_one_wall_top, top_surface_expansion));
        print.process();
        REQUIRE_FALSE(print.objects().empty());
        return TabTopLayer{ perimeter_length_at(print, tab_top_z), far_wall_inner_wall_length(print, tab_top_z) };
    };

    const TabTopLayer plain    = tab_top_layer_for(false);
    const TabTopLayer one_wall = tab_top_layer_for(true);

    // The option acts on this layer: the inner walls under the tab's top surface are gone.
    REQUIRE(plain.far_wall_inner_walls > 10.);
    CHECK(one_wall.perimeters < plain.perimeters);
    CHECK_THAT(one_wall.far_wall_inner_walls, Catch::Matchers::WithinAbs(plain.far_wall_inner_walls, 1.0));
}

namespace {

// The layer that closes the cavity of box_over_cavity(), the first one printed over air.
const double cavity_ceiling_z = 6.2;

// A cone standing on its tip, flaring by 5mm of radius per mm of height: at a layer height of 0.2 every
// wall of a layer lands a full millimetre outside the one below, entirely off the layer below but right
// alongside the walls printed with it.
TriangleMesh flared_cone()
{
    TriangleMesh cone = make_cone(20., 4.);
    cone.mirror(Z);
    cone.translate(0., 0., 4.);
    return cone;
}

// A 30mm box holding a 20mm cavity from z=2 to z=6, with a 4mm hole punched down through the ceiling
// of that cavity. The layer at cavity_ceiling_z bridges the cavity, and the walls of the hole sit in
// the middle of that bridge, 15mm clear of anything the layer below supports.
Print &box_over_cavity(Print &print, Model &model, const DynamicPrintConfig &config)
{
    ModelObject *object = model.add_object();
    object->name = "box_over_cavity.stl";
    object->add_volume(make_cube(30., 30., 8.), ModelVolumeType::MODEL_PART, false);
    TriangleMesh cavity = make_cube(20., 20., 4.);
    cavity.translate(5.f, 5.f, 2.f);
    object->add_volume(std::move(cavity), ModelVolumeType::NEGATIVE_VOLUME, false);
    TriangleMesh hole = make_cube(4., 4., 6.);
    hole.translate(13.f, 13.f, 5.f);
    object->add_volume(std::move(hole), ModelVolumeType::NEGATIVE_VOLUME, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    return print;
}

// Every setting the assertions below depend on, so none of them rests on a default.
DynamicPrintConfig unsupported_walls_config(const char *wall_generator, bool unsupported_wall_last)
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    config.set_deserialize_strict({
        { "wall_generator",             wall_generator },
        { "layer_height",               0.2 },
        { "initial_layer_print_height", 0.2 },
        { "wall_loops",                 3 },
        { "detect_overhang_wall",       true },
        // Outer wall first, so an unsupported loop only ends up last if the feature puts it there.
        { "wall_sequence",              "outer wall/inner wall" },
        { "is_infill_first",            false },
        { "sparse_infill_density",      "15%" },
        { "unsupported_wall_last",      unsupported_wall_last },
        { "gcode_comments",             true },
    });
    return config;
}

// A loop extruded entirely in mid air: every one of its paths is an overhang.
bool unsupported_loop(const ExtrusionEntity *entity)
{
    if (! entity->is_loop())
        return false;
    const ExtrusionPaths &paths = static_cast<const ExtrusionLoop *>(entity)->paths;
    return ! paths.empty() && std::all_of(paths.begin(), paths.end(),
                                          [](const ExtrusionPath &path) { return path.role() == erOverhangPerimeter; });
}

// The loops of every wall island of the print, island by island, in extrusion order.
std::vector<std::vector<const ExtrusionLoop*>> wall_islands(const Print &print)
{
    std::vector<std::vector<const ExtrusionLoop*>> islands;
    for (const Layer *layer : print.objects().front()->layers())
        for (const LayerRegion *region : layer->regions())
            for (const ExtrusionEntity *island : region->perimeters.entities) {
                std::vector<const ExtrusionLoop*> loops;
                for (const ExtrusionEntity *entity : static_cast<const ExtrusionEntityCollection*>(island)->entities)
                    if (entity->is_loop())
                        loops.push_back(static_cast<const ExtrusionLoop*>(entity));
                islands.push_back(std::move(loops));
            }
    return islands;
}

// Islands where a loop that is anchored is extruded after one that is not.
int islands_with_a_supported_loop_last(const Print &print)
{
    int count = 0;
    for (const std::vector<const ExtrusionLoop*> &loops : wall_islands(print)) {
        bool seen_unsupported = false;
        for (const ExtrusionLoop *loop : loops) {
            if (unsupported_loop(loop))
                seen_unsupported = true;
            else if (seen_unsupported) {
                ++ count;
                break;
            }
        }
    }
    return count;
}

// The unsupported loops of the print, and those of them held back for the infill.
std::vector<const ExtrusionLoop*> unsupported_loops(const Print &print, double print_z = -1.)
{
    std::vector<const ExtrusionLoop*> loops;
    for (const Layer *layer : print.objects().front()->layers()) {
        if (print_z >= 0. && std::abs(layer->print_z - print_z) > EPSILON)
            continue;
        for (const LayerRegion *region : layer->regions())
            for (const ExtrusionEntity *island : region->perimeters.entities)
                for (const ExtrusionEntity *entity : static_cast<const ExtrusionEntityCollection*>(island)->entities)
                    if (unsupported_loop(entity))
                        loops.push_back(static_cast<const ExtrusionLoop*>(entity));
    }
    return loops;
}

int loops_held_back_for_infill(const std::vector<const ExtrusionLoop*> &loops)
{
    return int(std::count_if(loops.begin(), loops.end(), [](const ExtrusionLoop *loop) { return loop->print_after_infill; }));
}

// The G-code emitted at `print_z`, so the order of one layer can be read on its own.
std::string layer_gcode(const std::string &gcode, double print_z)
{
    std::string out;
    GCodeReader reader;
    reader.parse_buffer(gcode, [&out, print_z](GCodeReader &self, const GCodeReader::GCodeLine &line) {
        if (std::abs(self.z() - print_z) < EPSILON)
            out += line.raw() + "\n";
    });
    return out;
}

} // namespace

// Whatever the wall order asks for, a loop with nothing under it cannot be extruded before the loops it
// leans on. The flared cone gives every layer an outer wall that lands completely off the one below, and
// the outer wall first sequence would otherwise put it down before any of them.
TEST_CASE("Unsupported wall loops are extruded after the walls that anchor them", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    auto slice_cone = [wall_generator](bool unsupported_wall_last, Print &print) {
        init_and_process_print({ flared_cone() }, print, unsupported_walls_config(wall_generator, unsupported_wall_last));
        REQUIRE_FALSE(print.objects().empty());
    };

    Print on;
    slice_cone(true, on);
    // Without unsupported loops to reorder the rest of the test would pass on an empty print.
    REQUIRE(unsupported_loops(on).size() > 0);
    CHECK(islands_with_a_supported_loop_last(on) == 0);

    SECTION("the held back loops run innermost first") {
        for (const std::vector<const ExtrusionLoop*> &loops : wall_islands(on)) {
            int previous_inset = std::numeric_limits<int>::max();
            for (const ExtrusionLoop *loop : loops)
                if (unsupported_loop(loop)) {
                    CHECK(loop->inset_idx <= previous_inset);
                    previous_inset = loop->inset_idx;
                }
        }
    }

    SECTION("switched off, the configured wall order is left alone") {
        Print off;
        slice_cone(false, off);
        REQUIRE(unsupported_loops(off).size() == unsupported_loops(on).size());
        // Outer wall first puts the unsupported outer wall ahead of the walls behind it.
        CHECK(islands_with_a_supported_loop_last(off) > 0);
    }
}

// A loop the walls cannot reach is a different case: only the bridges of its own layer will ever hold it,
// so it has to wait for them - while a loop that runs alongside a wall keeps its place, because the
// bridges anchor on it instead.
TEST_CASE("A wall loop out of reach of the layer below waits for the infill", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print print;
    Model model;
    box_over_cavity(print, model, unsupported_walls_config(wall_generator, true));
    print.process();

    const std::vector<const ExtrusionLoop*> hole_loops = unsupported_loops(print, cavity_ceiling_z);
    REQUIRE(hole_loops.size() > 0);
    CHECK(loops_held_back_for_infill(hole_loops) == int(hole_loops.size()));

    SECTION("a loop alongside a supported wall is not held back") {
        Print cone;
        init_and_process_print({ flared_cone() }, cone, unsupported_walls_config(wall_generator, true));
        const std::vector<const ExtrusionLoop*> loops = unsupported_loops(cone);
        REQUIRE(loops.size() > 0);
        CHECK(loops_held_back_for_infill(loops) == 0);
    }

    SECTION("switched off, no loop is held back") {
        Print off;
        Model off_model;
        box_over_cavity(off, off_model, unsupported_walls_config(wall_generator, false));
        off.process();
        const std::vector<const ExtrusionLoop*> loops = unsupported_loops(off, cavity_ceiling_z);
        REQUIRE(loops.size() == hole_loops.size());
        CHECK(loops_held_back_for_infill(loops) == 0);
    }
}

// The held back loops reach the G-code in a second pass, after the infill of their layer: on the layer
// that closes the cavity the walls of the hole are extruded once the bridge is down, so the layer emits
// perimeters, then infill, then the perimeters that were waiting for it.
TEST_CASE("Loops waiting for the infill are extruded after it", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    auto ceiling_roles = [wall_generator](bool unsupported_wall_last) {
        Print print;
        Model model;
        box_over_cavity(print, model, unsupported_walls_config(wall_generator, unsupported_wall_last));
        const std::string layer = layer_gcode(gcode(print), cavity_ceiling_z);
        REQUIRE_FALSE(layer.empty());
        return role_sequence(layer, { "perimeter", "infill" });
    };

    CHECK(ceiling_roles(true)  == std::vector<std::string>{ "perimeter", "infill", "perimeter" });
    CHECK(ceiling_roles(false) == std::vector<std::string>{ "perimeter", "infill" });
}

namespace {

// The rib spans z=[0,5] and the slab z=[5,6], so this is the slab's first layer - the only one whose
// support comes from the rib rather than from the slab below it.
const double slab_first_layer_z = 5.2;

// Rib widths either side of what the wall generators can print. At a 0.4mm nozzle the classic generator
// builds nothing thinner than nozzle/3 = 0.133mm and Arachne drops anything below min_feature_size, 25%
// of the nozzle = 0.1mm. 0.08mm is under both thresholds, 0.3mm over both.
const double unprintable_rib = 0.08;
const double printable_rib   = 0.3;

// A 4x5mm anchor tower carrying a 20x5mm slab at z=[5,6], with a rib `rib_width` wide running the whole
// length of the slab beneath its y=0 edge; a `rib_width` of 0 leaves the rib out. Nothing else is under
// that edge, so whether the wall along it is an overhang rests entirely on the rib. Overhang detection
// grows the lower slices by half the nozzle diameter before it asks, which carries either rib past the
// 0.21mm from the slab edge to that wall - the unprintable one only fails to reach it once it is filtered
// out for being unprintable.
Print &slab_over_rib(Print &print, Model &model, double rib_width, const DynamicPrintConfig &config)
{
    ModelObject *object = model.add_object();
    object->name = "slab_over_rib.stl";
    object->add_volume(make_cube(4., 5., 6.), ModelVolumeType::MODEL_PART, false);
    if (rib_width > 0.) {
        TriangleMesh rib = make_cube(20., rib_width, 5.);
        rib.translate(4.f, 0.f, 0.f);
        object->add_volume(std::move(rib), ModelVolumeType::MODEL_PART, false);
    }
    TriangleMesh slab = make_cube(20., 5., 1.);
    slab.translate(4.f, 0.f, 5.f);
    object->add_volume(std::move(slab), ModelVolumeType::MODEL_PART, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    return print;
}

// Every setting the assertions below depend on, so none of them rests on a default. The wall line widths
// are pinned because the rib widths above are chosen against the distance from the slab edge to its outer
// wall, and min_feature_size because it is one of the two thresholds under test.
DynamicPrintConfig printable_rib_config(const char *wall_generator, bool detect_thin_wall)
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    config.set_deserialize_strict({
        { "wall_generator",                wall_generator },
        { "layer_height",                  0.2 },  // puts a layer boundary exactly on the top of the rib
        { "initial_layer_print_height",    0.2 },
        { "nozzle_diameter",               "0.4" },
        { "outer_wall_line_width",         0.42 },
        { "inner_wall_line_width",         0.45 },
        { "wall_loops",                    2 },
        { "detect_overhang_wall",          true },
        { "detect_thin_wall",              detect_thin_wall },
        { "min_feature_size",              "25%" },
        { "raft_layers",                   0 },
        // Anything that adds, drops or reorders walls would move length between the roles being counted.
        { "extra_perimeters_on_overhangs", false },
        { "overhang_reverse",              false },
        { "only_one_wall_top",             false },
        { "only_one_wall_first_layer",     false },
        { "unsupported_wall_last",         false },
        { "sparse_infill_density",         "15%" },
    });
    return config;
}

// Length of every overhang perimeter path on the layer at `print_z`, loops and open extrusions alike.
double overhang_length_at(const Print &print, double print_z)
{
    double len = 0.;
    const auto add_entity = [&len](const ExtrusionEntity *entity, auto &&self) -> void {
        const auto add_paths = [&len](const ExtrusionPaths &paths) {
            for (const ExtrusionPath &path : paths)
                if (path.role() == erOverhangPerimeter)
                    len += path.length();
        };
        if (const auto *coll = dynamic_cast<const ExtrusionEntityCollection*>(entity)) {
            for (const ExtrusionEntity *child : coll->entities)
                self(child, self);
        } else if (const auto *loop = dynamic_cast<const ExtrusionLoop*>(entity)) {
            add_paths(loop->paths);
        } else if (const auto *multi = dynamic_cast<const ExtrusionMultiPath*>(entity)) {
            add_paths(multi->paths);
        } else if (const auto *path = dynamic_cast<const ExtrusionPath*>(entity)) {
            if (path->role() == erOverhangPerimeter)
                len += path->length();
        }
    };

    for (const Layer *layer : print.objects().front()->layers()) {
        if (std::abs(layer->print_z - print_z) > EPSILON)
            continue;
        for (const LayerRegion *region : layer->regions())
            add_entity(&region->perimeters, add_entity);
    }
    return len;
}

} // namespace

// A sliver the wall generator prints nothing for holds nothing up, so it cannot be what decides that the
// wall above it is not an overhang. The rib under the slab is the only thing that edge of the slab could
// rest on: below the threshold of the active generator the slab has to come out exactly as it does with
// no rib at all, and the last check is the control - a rib the generator does print anchors that wall,
// without which the first check would hold for want of any sensitivity to the rib.
TEST_CASE("A lower layer sliver too thin to print does not support the wall above it", "[Perimeters]")
{
    const char *wall_generator   = GENERATE("classic", "arachne");
    const bool  detect_thin_wall = GENERATE(true, false);
    CAPTURE(wall_generator, detect_thin_wall);

    auto overhang_for = [wall_generator, detect_thin_wall](double rib_width) {
        Print print;
        Model model;
        slab_over_rib(print, model, rib_width, printable_rib_config(wall_generator, detect_thin_wall));
        print.process();
        REQUIRE_FALSE(print.objects().empty());
        return overhang_length_at(print, slab_first_layer_z);
    };

    const double no_rib      = overhang_for(0.);
    const double unprintable = overhang_for(unprintable_rib);
    const double printable   = overhang_for(printable_rib);

    // Only where the slab meets the tower is it held up from below, so both of its 20mm walls overhang.
    REQUIRE(no_rib > scale_(30.));
    CHECK_THAT(unprintable, Catch::Matchers::WithinAbs(no_rib, scale_(1.)));
    // A rib that does get printed takes the 20mm outer wall running along it out of the overhangs.
    CHECK(printable < no_rib - scale_(15.));
}

namespace {

// Every setting the fuzzy skin assertions below depend on.
DynamicPrintConfig fuzzy_skin_config(const char *wall_generator)
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    config.set_deserialize_strict({
        { "wall_generator",             wall_generator },
        { "layer_height",               0.2 },
        { "initial_layer_print_height", 0.2 },
        // One wall, so every wall point along the long sides belongs to the fuzzed outer wall.
        { "wall_loops",                 1 },
        { "fuzzy_skin",                 "external" },
        { "fuzzy_skin_noise_type",      "classic" },
        { "fuzzy_skin_thickness",       0.3 },
        { "fuzzy_skin_point_distance",  0.8 },
    });
    return config;
}

// How far the wall points over the middle 60% of the layer's length stray across its width, worst side.
// A negative result means there is no layer at `print_z`.
double mid_span_wall_spread(const Print &print, double print_z)
{
    for (const Layer *layer : print.objects().front()->layers()) {
        if (std::abs(layer->print_z - print_z) > 1e-4)
            continue;
        const BoundingBox bbox  = get_extents(layer->lslices);
        const coord_t     x_min = bbox.min.x() + bbox.size().x() / 5;
        const coord_t     x_max = bbox.max.x() - bbox.size().x() / 5;
        Points            points;
        for (const LayerRegion *region : layer->regions())
            region->perimeters.collect_points(points);
        coord_t spread = 0;
        for (const bool south : { true, false }) {
            coord_t lo = bbox.max.y(), hi = bbox.min.y();
            for (const Point &p : points)
                if (p.x() > x_min && p.x() < x_max && (p.y() < bbox.center().y()) == south) {
                    lo = std::min(lo, p.y());
                    hi = std::max(hi, p.y());
                }
            spread = std::max(spread, hi - lo);
        }
        return unscale<double>(spread);
    }
    return -1.;
}

} // namespace

// TestMesh::bridge is a 50x10mm deck from z=5 to z=8 on two 5mm-wide pillars, leaving a 40mm span. The deck's
// first layer (print_z 5.2) crosses the span unsupported; the layers above it rest on the deck.
TEST_CASE("Fuzzy skin leaves the walls of a bridge smooth", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print print;
    init_and_process_print({ TestMesh::bridge }, print, fuzzy_skin_config(wall_generator));
    REQUIRE_FALSE(print.objects().empty());

    // Control: one deck layer up the same walls rest on the deck, so they are fuzzed.
    CHECK(mid_span_wall_spread(print, 5.6) > 0.1);
    // Over the unsupported span the walls stay straight.
    const double bridged = mid_span_wall_spread(print, 5.2);
    CHECK(bridged >= 0.);
    CHECK(bridged < 0.001);
}

// One object: a 20x20x3mm block on the bed and a second one floating above it from z=5 to z=8. The layers in
// the gap are empty, so the floating block's first layer (print_z 5.2) has a layer below it with nothing
// printed on it; the layers above rest on the floating block.
TEST_CASE("Fuzzy skin leaves the walls over an empty layer smooth", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    TriangleMesh mesh     = make_cube(20., 20., 3.);
    TriangleMesh floating = make_cube(20., 20., 3.);
    floating.translate(0.f, 0.f, 5.f);
    mesh.merge(floating);

    Print print;
    init_and_process_print({ mesh }, print, fuzzy_skin_config(wall_generator));
    REQUIRE_FALSE(print.objects().empty());

    // Control: one layer up the walls rest on the floating block, so they are fuzzed.
    CHECK(mid_span_wall_spread(print, 5.6) > 0.1);
    // Nothing is printed under the first floating layer, so its walls stay straight.
    const double floating_first_layer = mid_span_wall_spread(print, 5.2);
    CHECK(floating_first_layer >= 0.);
    CHECK(floating_first_layer < 0.001);
}

namespace {

// A 20x20x5mm square tube whose walls are `wall` mm thick, except the far one at `far_wall` mm.
Print &square_tube(Print &print, Model &model, const DynamicPrintConfig &config, double wall, double far_wall)
{
    ModelObject *object = model.add_object();
    object->name = "square_tube.stl";
    object->add_volume(make_cube(20., 20., 5.), ModelVolumeType::MODEL_PART, false);
    TriangleMesh cavity = make_cube(20. - 2. * wall, 20. - wall - far_wall, 5.);
    cavity.translate(float(wall), float(wall), 0.f);
    object->add_volume(std::move(cavity), ModelVolumeType::NEGATIVE_VOLUME, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    print.process();
    return print;
}

// Every setting the wall thicknesses below are measured against. With these widths the two outer walls of
// a 0.8mm wall touch, a 1mm wall leaves a gap between them for gap fill, and an inner wall needs about 1.5mm.
// Both one wall options are on, so the first and the last layer have a single wall whatever wall_loops asks.
DynamicPrintConfig hole_direction_config(int wall_loops, const char *wall_direction)
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    config.set_deserialize_strict({
        { "wall_generator",             "classic" },
        { "wall_direction",             wall_direction },
        { "wall_loops",                 wall_loops },
        { "layer_height",               0.2 },
        { "initial_layer_print_height", 0.2 },
        { "outer_wall_line_width",      0.42 },
        { "inner_wall_line_width",      0.45 },
        { "detect_thin_wall",           false },
        { "filter_out_gap_fill",        0 },
        { "top_shell_layers",           3 },
        { "bottom_shell_layers",        3 },
        { "only_one_wall_top",          true },
        { "only_one_wall_first_layer",  true },
        { "overhang_reverse",           false },
        { "sparse_infill_density",      "15%" },
    });
    return config;
}

// The outer walls of a layer, contours and holes apart, and how many inner walls and gap fills it has.
struct WallDirections {
    std::vector<bool> contours_ccw;
    std::vector<bool> holes_ccw;
    int               inner_walls = 0;
    size_t            gap_fills   = 0;
};

std::vector<WallDirections> wall_directions(const Print &print)
{
    std::vector<WallDirections> out;
    for (const Layer *layer : print.objects().front()->layers()) {
        WallDirections &walls = out.emplace_back();
        for (const LayerRegion *region : layer->regions()) {
            walls.gap_fills += region->thin_fills.flatten().entities.size();
            for (const ExtrusionEntity *entity : region->perimeters.flatten().entities) {
                if (! entity->is_loop())
                    continue;
                const ExtrusionLoop *loop = static_cast<const ExtrusionLoop*>(entity);
                if (loop->inset_idx > 0)
                    ++ walls.inner_walls;
                else
                    (loop->loop_role() == elrHole ? walls.holes_ccw : walls.contours_ccw).push_back(loop->polygon().is_counter_clockwise());
            }
        }
    }
    return out;
}

} // namespace

// Holes run against the wall direction, so the inside of a hole keeps its direction on the layers where the
// hole opens into the contour. That holds on every layer, the single wall ones included, as soon as anything
// fits beside the outer wall of the hole: infill, an inner wall, or only gap fill. The 1mm tube with a 3mm far
// side is the shape that used to flip, where the far side has an inner wall on most layers and gap fill runs
// around the rest.
TEST_CASE("Holes run against the wall direction", "[Perimeters]")
{
    const auto [wall, far_wall] = GENERATE(std::make_pair(7., 7.), std::make_pair(1., 1.), std::make_pair(1., 3.));
    const int   wall_loops      = GENERATE(1, 2);
    const char *wall_direction  = GENERATE("ccw", "cw");
    CAPTURE(wall, far_wall, wall_loops, wall_direction);

    Print print;
    Model model;
    square_tube(print, model, hole_direction_config(wall_loops, wall_direction), wall, far_wall);
    const std::vector<WallDirections> layers = wall_directions(print);
    // 5mm at 0.2mm layers.
    REQUIRE(layers.size() == 25);
    // Without gap fill between the outer walls the thin tubes would test the case below instead.
    if (wall < 2.)
        REQUIRE(layers[layers.size() / 2].gap_fills > 0);

    const bool ccw = std::string(wall_direction) == "ccw";
    for (size_t i = 0; i < layers.size(); ++ i) {
        CAPTURE(i);
        REQUIRE(layers[i].contours_ccw.size() == 1);
        REQUIRE(layers[i].holes_ccw.size() == 1);
        CHECK(layers[i].contours_ccw.front() == ccw);
        CHECK(layers[i].holes_ccw.front() == ! ccw);
    }
}

// A hole whose outer wall touches the contour's all around, with nothing between them, runs with the contour
// on every layer, so the two walls of a thin tube are laid side by side in the same direction.
TEST_CASE("A hole whose outer wall touches the contour's runs with it", "[Perimeters]")
{
    const int   wall_loops     = GENERATE(1, 2);
    const char *wall_direction = GENERATE("ccw", "cw");
    CAPTURE(wall_loops, wall_direction);

    Print print;
    Model model;
    square_tube(print, model, hole_direction_config(wall_loops, wall_direction), 0.8, 0.8);
    const std::vector<WallDirections> layers = wall_directions(print);
    REQUIRE(layers.size() == 25);
    // Nothing fits between the two outer walls.
    REQUIRE(layers[layers.size() / 2].gap_fills == 0);
    REQUIRE(layers[layers.size() / 2].inner_walls == 0);

    const bool ccw = std::string(wall_direction) == "ccw";
    for (size_t i = 0; i < layers.size(); ++ i) {
        CAPTURE(i);
        REQUIRE(layers[i].contours_ccw.size() == 1);
        REQUIRE(layers[i].holes_ccw.size() == 1);
        CHECK(layers[i].contours_ccw.front() == ccw);
        CHECK(layers[i].holes_ccw.front() == ccw);
    }
}

namespace {

// The outer radius of round_tube().
const double tube_radius = 11.5;

// Every setting the tube walls below are measured against. With 0.5mm walls a 1mm tube wall holds only the two
// outer walls, touching, and every further 0.5mm makes room for one more inner wall.
DynamicPrintConfig tube_config(const char *wall_generator, int wall_loops)
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    config.set_deserialize_strict({
        { "wall_generator",             wall_generator },
        { "wall_loops",                 wall_loops },
        { "wall_sequence",              "inner wall/outer wall" },
        { "wall_direction",             "ccw" },
        { "layer_height",               0.2 },
        { "initial_layer_print_height", 0.2 },
        { "nozzle_diameter",            "0.4" },
        { "outer_wall_line_width",      0.5 },
        { "inner_wall_line_width",      0.5 },
        { "precise_outer_wall",         true },
        { "detect_thin_wall",           false },
        { "only_one_wall_top",          false },
        { "only_one_wall_first_layer",  false },
        { "brim_type",                  "no_brim" },
        { "detect_overhang_wall",       true },
        { "overhang_reverse",           false },
        { "overhang_reverse_threshold", "0%" },
    });
    return config;
}

// A 2mm tall round tube whose wall is `wall` mm thick.
Print &round_tube(Print &print, Model &model, const DynamicPrintConfig &config, double wall)
{
    ModelObject *object = model.add_object();
    object->name = "round_tube.stl";
    object->add_volume(make_cylinder(tube_radius, 2.), ModelVolumeType::MODEL_PART, false);
    object->add_volume(make_cylinder(tube_radius - wall, 2.), ModelVolumeType::NEGATIVE_VOLUME, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    print.process();
    return print;
}

// The side of the tube wall a wall loop lies on. The centre is the middle of the tube wall, where Arachne lays the
// inner wall both sides share when the tube wall is too narrow for a pair of them.
enum class TubeSide { Contour, Hole, Centre };

struct TubeLoop {
    TubeSide side;
    int      inset;
    bool     ccw;
};

// The wall loops of every layer of round_tube() in extrusion order.
std::vector<std::vector<TubeLoop>> tube_loops(const Print &print, double wall)
{
    const double centre = tube_radius - 0.5 * wall;
    std::vector<std::vector<TubeLoop>> out;
    for (const Layer *layer : print.objects().front()->layers()) {
        const Vec2d axis = get_extents(layer->lslices).center().cast<double>();
        std::vector<TubeLoop> &loops = out.emplace_back();
        for (const LayerRegion *region : layer->regions())
            for (const ExtrusionEntity *entity : region->perimeters.flatten().entities) {
                if (! entity->is_loop())
                    continue;
                const ExtrusionLoop *loop    = static_cast<const ExtrusionLoop*>(entity);
                const Polygon        polygon = loop->polygon();
                double               radius  = 0.;
                for (const Point &p : polygon.points)
                    radius += unscaled<double>((p.cast<double>() - axis).norm());
                radius /= double(polygon.points.size());
                const TubeSide side = radius > centre + 0.1 ? TubeSide::Contour : radius < centre - 0.1 ? TubeSide::Hole : TubeSide::Centre;
                loops.push_back({ side, loop->inset_idx, polygon.is_counter_clockwise() });
            }
    }
    return out;
}

// The insets one side of a wall is printed in, given its deepest inset: inside out for Inner/Outer, outside in for
// Outer/Inner, and for Inner/Outer/Inner the third wall onwards inside out, then the outer wall, then the first inner
// wall. Inner/Outer/Inner prints its first layer Inner/Outer, and outside in when there is no third wall.
std::vector<int> wall_sequence_insets(const std::string &wall_sequence, bool first_layer, int deepest)
{
    std::vector<int> insets;
    if (wall_sequence == "inner wall/outer wall" || (wall_sequence == "inner-outer-inner wall" && first_layer)) {
        for (int inset = deepest; inset >= 0; -- inset)
            insets.push_back(inset);
    } else if (wall_sequence == "outer wall/inner wall" || deepest < 2) {
        for (int inset = 0; inset <= deepest; ++ inset)
            insets.push_back(inset);
    } else {
        for (int inset = deepest; inset >= 2; -- inset)
            insets.push_back(inset);
        insets.push_back(0);
        insets.push_back(1);
    }
    return insets;
}

// Whether a wall of round_tube() runs with the contour: all but the walls of the hole, and those too when the hole
// is a thin wall hole, its outer wall touching the contour's with nothing between them.
bool runs_with_contour(const TubeLoop &loop, bool thin_wall_hole)
{
    return loop.side != TubeSide::Hole || thin_wall_hole;
}

// The insets of the walls on one side of a round_tube() layer, the shared centre line included, in extrusion order.
std::vector<int> side_insets(const std::vector<TubeLoop> &loops, TubeSide side)
{
    std::vector<int> insets;
    for (const TubeLoop &loop : loops)
        if (loop.side == side || loop.side == TubeSide::Centre)
            insets.push_back(loop.inset);
    return insets;
}

// The walls of a round_tube() layer in extrusion order, as side and inset: "c0 c1 h0 h1" is the contour's outer wall,
// then its first inner wall, then the same for the hole, and "m" marks the wall in the middle both sides share.
std::string wall_order(const std::vector<TubeLoop> &loops)
{
    std::string out;
    for (const TubeLoop &loop : loops)
        out += std::string(out.empty() ? "" : " ") + (loop.side == TubeSide::Contour ? "c" : loop.side == TubeSide::Hole ? "h" : "m") +
               std::to_string(loop.inset);
    return out;
}

} // namespace

// Each side of a tube wall prints its walls in the order the wall sequence asks for, from two walls a side up to
// four. The inner wall Arachne lays in the middle of a wall too narrow for a pair of them is an inner wall of both
// sides; Classic fills that middle with gap fill instead, so it has one wall less a side there.
TEST_CASE("Each side of a tube wall prints its walls in the wall sequence order", "[Perimeters]")
{
    struct TubeWall {
        double wall;
        int    wall_loops;
        // Walls a side, a shared inner wall included.
        int    arachne_walls;
        int    classic_walls;
    };
    const TubeWall tube = GENERATE(values<TubeWall>({
        { 1.5, 3, 2, 1 }, { 2.0, 2, 2, 2 }, { 2.5, 3, 3, 2 }, { 3.0, 3, 3, 3 }, { 4.0, 4, 4, 4 } }));
    const char       *wall_generator = GENERATE("classic", "arachne");
    const std::string wall_sequence  = GENERATE("inner wall/outer wall", "outer wall/inner wall", "inner-outer-inner wall");
    CAPTURE(tube.wall, tube.wall_loops, wall_generator, wall_sequence);

    DynamicPrintConfig config = tube_config(wall_generator, tube.wall_loops);
    config.set_deserialize_strict("wall_sequence", wall_sequence);
    Print print;
    Model model;
    round_tube(print, model, config, tube.wall);
    const std::vector<std::vector<TubeLoop>> layers = tube_loops(print, tube.wall);
    // 2mm at 0.2mm layers.
    REQUIRE(layers.size() == 10);

    const int walls = std::string(wall_generator) == "arachne" ? tube.arachne_walls : tube.classic_walls;
    for (size_t i = 0; i < layers.size(); ++ i) {
        for (const TubeSide side : { TubeSide::Contour, TubeSide::Hole }) {
            const bool contour_side = side == TubeSide::Contour;
            CAPTURE(i, contour_side);
            const std::vector<int> insets = side_insets(layers[i], side);
            REQUIRE(int(insets.size()) == walls);
            CHECK(insets == wall_sequence_insets(wall_sequence, i == 0, walls - 1));
        }
    }
}

// The outer wall of a thin wall hole touches the contour's with nothing between them. The hole is printed first, so
// the contour leans on it, unless the wall sequence puts the outer walls first.
TEST_CASE("A thin wall hole is printed before the contour unless the outer walls go first", "[Perimeters]")
{
    const char       *wall_generator = GENERATE("classic", "arachne");
    const std::string wall_sequence  = GENERATE("inner wall/outer wall", "outer wall/inner wall", "inner-outer-inner wall");
    CAPTURE(wall_generator, wall_sequence);

    DynamicPrintConfig config = tube_config(wall_generator, 3);
    config.set_deserialize_strict("wall_sequence", wall_sequence);
    Print print;
    Model model;
    round_tube(print, model, config, 1.);
    const std::vector<std::vector<TubeLoop>> layers = tube_loops(print, 1.);
    REQUIRE(layers.size() == 10);

    const TubeSide first = wall_sequence == "outer wall/inner wall" ? TubeSide::Contour : TubeSide::Hole;
    for (size_t i = 0; i < layers.size(); ++ i) {
        CAPTURE(i);
        REQUIRE(layers[i].size() == 2);
        REQUIRE(layers[i][0].side != layers[i][1].side);
        CHECK(layers[i].front().side == first);
    }
}

// A 1.5mm tube wall leaves Inner/Outer/Inner no third wall to sandwich, only the inner wall Arachne lays in the middle
// for both sides, or Classic's gap fill. Past the first layer it prints the hole first, then the contour, then the
// middle.
TEST_CASE("Inner/Outer/Inner prints the hole of a narrow tube wall first and the middle last", "[Perimeters]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    DynamicPrintConfig config = tube_config(wall_generator, 3);
    config.set_deserialize_strict("wall_sequence", "inner-outer-inner wall");
    Print print;
    Model model;
    round_tube(print, model, config, 1.5);
    const std::vector<std::vector<TubeLoop>> layers = tube_loops(print, 1.5);
    REQUIRE(layers.size() == 10);

    // Classic's gap fill is not a wall loop, and goes after the walls.
    const std::string expected = std::string(wall_generator) == "arachne" ? "h0 c0 m1" : "h0 c0";
    for (size_t i = 1; i < layers.size(); ++ i) {
        CAPTURE(i);
        CHECK(wall_order(layers[i]) == expected);
    }
}

namespace {

// Two 12mm radius cylinders 14mm apart, 2mm tall, with a round hole in the middle 1mm short of the waist on either
// side. At the waist the two outer walls touch with nothing between them, so each inner wall turns there and runs back
// along the other outer wall: an inner wall of both the contour and the hole.
Print &pinched_tube(Print &print, Model &model, const DynamicPrintConfig &config)
{
    ModelObject *object = model.add_object();
    object->name = "pinched_tube.stl";
    for (const float y : { 7.f, -7.f }) {
        TriangleMesh lobe = make_cylinder(12., 2.);
        lobe.translate(0.f, y, 0.f);
        object->add_volume(std::move(lobe), ModelVolumeType::MODEL_PART, false);
    }
    // The waist is sqrt(12^2 - 7^2) = 9.75mm from the centre.
    object->add_volume(make_cylinder(8.75, 2.), ModelVolumeType::NEGATIVE_VOLUME, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    print.process();
    return print;
}

} // namespace

// The inner walls the contour and the hole share go after both outer walls, and the walls beyond them before, so the
// hole's outer wall is sandwiched like the contour's instead of being printed last.
TEST_CASE("Inner/Outer/Inner prints both outer walls of a pinched wall before the inner walls they share", "[Perimeters]")
{
    DynamicPrintConfig config = tube_config("arachne", 3);
    config.set_deserialize_strict("wall_sequence", "inner-outer-inner wall");
    Print print;
    Model model;
    pinched_tube(print, model, config);
    const auto layers = print.objects().front()->layers();
    // 2mm at 0.2mm layers.
    REQUIRE(layers.size() == 10);

    // The first layer is printed Inner/Outer.
    for (size_t i = 1; i < layers.size(); ++ i) {
        CAPTURE(i);
        // The inset of each wall in extrusion order, and where the two outer walls are.
        std::vector<int> insets;
        int              contour = -1;
        int              hole    = -1;
        for (const LayerRegion *region : layers[i]->regions())
            for (const ExtrusionEntity *entity : region->perimeters.flatten().entities) {
                if (entity->inset_idx == 0 && entity->is_loop())
                    (static_cast<const ExtrusionLoop*>(entity)->loop_role() & elrHole ? hole : contour) = int(insets.size());
                insets.push_back(int(entity->inset_idx));
            }
        REQUIRE(contour >= 0);
        REQUIRE(hole >= 0);
        REQUIRE(std::count(insets.begin(), insets.end(), 1) > 0);
        REQUIRE(std::count(insets.begin(), insets.end(), 2) > 0);
        const int first_outer = std::min(contour, hole);
        const int last_outer  = std::max(contour, hole);
        CHECK(last_outer == first_outer + 1);
        for (int k = 0; k < int(insets.size()); ++ k) {
            CAPTURE(k, insets[k]);
            if (insets[k] >= 2)
                CHECK(k < first_outer);
            else if (insets[k] == 1)
                CHECK(k > last_outer);
        }
    }
}

namespace {

// A 24x24x2mm slab with a round hole of 3mm radius, and a 2mm wide collar around the hole rising to 4mm. On the slab's
// last layer the collar leaves a band around the hole that is not top surface, wide enough for three walls.
Print &collar_slab(Print &print, Model &model, const DynamicPrintConfig &config)
{
    ModelObject *object = model.add_object();
    object->name = "collar_slab.stl";
    TriangleMesh slab = make_cube(24., 24., 2.);
    slab.translate(-12.f, -12.f, 0.f);
    object->add_volume(std::move(slab), ModelVolumeType::MODEL_PART, false);
    object->add_volume(make_cylinder(5., 4.), ModelVolumeType::MODEL_PART, false);
    object->add_volume(make_cylinder(3., 4.), ModelVolumeType::NEGATIVE_VOLUME, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    print.process();
    return print;
}

// A 6mm radius cup, 3mm tall, with a 2mm wall and a 1.2mm floor. On the floor's last layer the cup's wall is a band
// beside the floor's top surface, wide enough for three walls inside the cup's outer wall.
Print &floored_cup(Print &print, Model &model, const DynamicPrintConfig &config)
{
    ModelObject *object = model.add_object();
    object->name = "floored_cup.stl";
    object->add_volume(make_cylinder(6., 3.), ModelVolumeType::MODEL_PART, false);
    TriangleMesh inside = make_cylinder(4., 3.);
    inside.translate(0.f, 0.f, 1.2f);
    object->add_volume(std::move(inside), ModelVolumeType::NEGATIVE_VOLUME, false);
    object->add_instance();
    object->ensure_on_bed();

    print.auto_assign_extruders(object);
    print.apply(model, config);
    print.validate();
    print.set_status_silent();
    print.process();
    return print;
}

} // namespace

// With only one wall on top and no top surface expansion, the inner walls of a layer with a top surface are generated in
// the band left beside it, starting from both of its edges. The wall along the top surface is still the third wall from
// the outer wall, so every wall sequence orders it by that depth: Inner/Outer prints it before the first inner wall,
// Outer/Inner after it, and Inner/Outer/Inner before the outer wall, with the first inner wall after it. Checked for a
// hole's outer wall beside a top surface (collar_slab) and for a contour beside one (floored_cup).
TEST_CASE("Walls beside a top surface keep their order by depth with only one wall on top", "[Perimeters]")
{
    const std::string shape         = GENERATE("hole", "contour");
    const std::string wall_sequence = GENERATE("inner wall/outer wall", "outer wall/inner wall", "inner-outer-inner wall");
    CAPTURE(shape, wall_sequence);
    DynamicPrintConfig config = tube_config("arachne", 3);
    config.set_deserialize_strict({
        { "wall_sequence",         wall_sequence },
        { "only_one_wall_top",     true },
        { "top_shell_layers",      3 },
        { "top_surface_density",   "100%" },
        { "top_surface_expansion", 0. },
    });
    Print print;
    Model model;
    if (shape == "hole")
        collar_slab(print, model, config);
    else
        floored_cup(print, model, config);

    // The slab's last layer at 2mm, or the floor's at 1.2mm. The walls around the axis as their mean distance from it, in
    // extrusion order.
    const double print_z = shape == "hole" ? 2. : 1.2;
    const Layer *layer   = nullptr;
    for (const Layer *l : print.objects().front()->layers())
        if (std::abs(l->print_z - print_z) < EPSILON)
            layer = l;
    REQUIRE(layer != nullptr);
    std::vector<double> radii;
    int                 outer = -1;
    for (const LayerRegion *region : layer->regions())
        for (const ExtrusionEntity *entity : region->perimeters.flatten().entities) {
            const Polyline polyline = entity->as_polyline();
            double         radius   = 0.;
            for (const Point &p : polyline.points)
                radius += unscaled<double>(p.cast<double>().norm());
            radius /= double(polyline.points.size());
            if (radius > 8.)
                continue; // the slab's own walls
            if (entity->inset_idx == 0)
                outer = int(radii.size());
            radii.push_back(radius);
        }
    REQUIRE(outer >= 0);
    // With 0.5mm walls the first inner wall is about 0.5mm from the outer wall, the third wall about 1mm. The positions
    // in extrusion order of the first inner walls and of the third walls and beyond.
    std::vector<int> first_inner;
    std::vector<int> third_wall;
    for (int k = 0; k < int(radii.size()); ++ k)
        if (k != outer)
            (std::abs(radii[k] - radii[outer]) > 0.75 ? third_wall : first_inner).push_back(k);
    CAPTURE(radii, outer);
    REQUIRE(! first_inner.empty());
    REQUIRE(! third_wall.empty());
    if (wall_sequence == "inner wall/outer wall") {
        CHECK(third_wall.back() < first_inner.front());
        CHECK(first_inner.back() < outer);
    } else if (wall_sequence == "outer wall/inner wall") {
        CHECK(outer < first_inner.front());
        CHECK(first_inner.back() < third_wall.front());
    } else {
        CHECK(third_wall.back() < outer);
        CHECK(outer < first_inner.front());
    }
}

// The contour's walls run in the wall direction and the hole's against it, except that the walls of a thin wall
// hole run with the contour, as does the inner wall Arachne shares between both sides of a 1.5mm tube wall.
TEST_CASE("Tube walls run in the wall direction and hole walls against it", "[Perimeters]")
{
    const double      wall           = GENERATE(1.0, 1.5, 2.0, 3.0);
    const char       *wall_generator = GENERATE("classic", "arachne");
    const std::string wall_direction = GENERATE("ccw", "cw");
    CAPTURE(wall, wall_generator, wall_direction);

    DynamicPrintConfig config = tube_config(wall_generator, 3);
    config.set_deserialize_strict("wall_direction", wall_direction);
    Print print;
    Model model;
    round_tube(print, model, config, wall);
    const std::vector<std::vector<TubeLoop>> layers = tube_loops(print, wall);
    REQUIRE(layers.size() == 10);

    const bool thin_wall_hole = wall == 1.;
    for (size_t i = 0; i < layers.size(); ++ i) {
        CAPTURE(i);
        REQUIRE(std::any_of(layers[i].begin(), layers[i].end(), [](const TubeLoop &loop) { return loop.side == TubeSide::Contour; }));
        REQUIRE(std::any_of(layers[i].begin(), layers[i].end(), [](const TubeLoop &loop) { return loop.side == TubeSide::Hole; }));
        for (const TubeLoop &loop : layers[i]) {
            CAPTURE(int(loop.side), loop.inset);
            CHECK(loop.ccw == (runs_with_contour(loop, thin_wall_hole) == (wall_direction == "ccw")));
        }
    }
}

// Reverse on even with no threshold reverses every wall of every other layer, or only the inner walls when it is
// restricted to them, and leaves the order of the walls alone.
TEST_CASE("Reverse on even reverses the tube walls of every other layer", "[Perimeters]")
{
    const double wall                  = GENERATE(1.0, 1.5, 3.0);
    const char  *wall_generator        = GENERATE("classic", "arachne");
    const bool   reverse_internal_only = GENERATE(false, true);
    CAPTURE(wall, wall_generator, reverse_internal_only);

    auto slice = [&](bool overhang_reverse) {
        DynamicPrintConfig config = tube_config(wall_generator, 3);
        config.set_deserialize_strict({
            { "overhang_reverse",               overhang_reverse },
            { "overhang_reverse_internal_only", reverse_internal_only },
        });
        Print print;
        Model model;
        round_tube(print, model, config, wall);
        return tube_loops(print, wall);
    };
    const std::vector<std::vector<TubeLoop>> plain    = slice(false);
    const std::vector<std::vector<TubeLoop>> reversed = slice(true);
    REQUIRE(plain.size() == 10);
    REQUIRE(reversed.size() == plain.size());

    for (size_t i = 0; i < plain.size(); ++ i) {
        CAPTURE(i);
        REQUIRE(reversed[i].size() == plain[i].size());
        for (size_t j = 0; j < plain[i].size(); ++ j) {
            CAPTURE(j);
            CHECK(reversed[i][j].side == plain[i][j].side);
            CHECK(reversed[i][j].inset == plain[i][j].inset);
            const bool flips = i % 2 == 1 && (! reverse_internal_only || plain[i][j].inset > 0);
            CHECK(reversed[i][j].ccw == (plain[i][j].ccw != flips));
        }
    }
}

namespace {

DynamicPrintConfig brim_tube_config(const char *wall_generator, const std::string &wall_sequence, const char *brim_type, double brim_width)
{
    DynamicPrintConfig config = tube_config(wall_generator, 3);
    config.set_deserialize_strict({
        { "wall_sequence", wall_sequence },
        { "brim_type",     brim_type },
        { "brim_width",    brim_width },
    });
    return config;
}

} // namespace

// A brim against the outer walls makes the first layer print them first, whatever the wall sequence, so the brim runs
// straight on into the outer wall: each side of the tube wall outside in, and the contour before a thin wall hole.
// Brims that may leave the outer walls bare keep the wall sequence, and so do the layers above.
TEST_CASE("A brim against the outer walls makes the first layer print them first", "[Perimeters]")
{
    struct Brim {
        const char *type;
        double      width;
        bool        outer_walls_first;
    };
    const Brim brim = GENERATE(values<Brim>({
        { "no_brim", 5., false }, { "inner_only", 5., false }, { "auto_brim", 5., false }, { "outer_only", 0., false },
        { "outer_only", 5., true }, { "outer_and_inner", 5., true }, { "leading_edge_only", 5., true } }));
    const double      wall           = GENERATE(1.0, 3.0);
    const char       *wall_generator = GENERATE("classic", "arachne");
    const std::string wall_sequence  = GENERATE("inner wall/outer wall", "inner-outer-inner wall");
    CAPTURE(brim.type, brim.width, wall, wall_generator, wall_sequence);

    Print print;
    Model model;
    round_tube(print, model, brim_tube_config(wall_generator, wall_sequence, brim.type, brim.width), wall);
    const std::vector<std::vector<TubeLoop>> layers = tube_loops(print, wall);
    REQUIRE(layers.size() == 10);

    // The thin wall hole has its outer wall a side, the 3mm tube wall three walls a side.
    const size_t walls = wall == 1. ? 1 : 3;
    for (const size_t i : { size_t(0), size_t(1) }) {
        const bool outer_walls_first = brim.outer_walls_first && i == 0;
        for (const TubeSide side : { TubeSide::Contour, TubeSide::Hole }) {
            const bool contour_side = side == TubeSide::Contour;
            CAPTURE(i, contour_side);
            const std::vector<int> insets = side_insets(layers[i], side);
            REQUIRE(insets.size() == walls);
            CHECK(insets == wall_sequence_insets(outer_walls_first ? "outer wall/inner wall" : wall_sequence, i == 0, int(walls) - 1));
        }
        if (wall == 1.)
            CHECK(layers[i].front().side == (outer_walls_first ? TubeSide::Contour : TubeSide::Hole));
    }
}

// Changing the brim of a sliced print regenerates the walls when the first layer's wall order changes with it, so
// the print gets the walls a slice from scratch with the new brim gets.
TEST_CASE("Changing the brim updates the wall order of the first layer", "[Perimeters]")
{
    struct BrimChange {
        const char *from_type;
        double      from_width;
        const char *to_type;
        double      to_width;
    };
    const BrimChange change = GENERATE(values<BrimChange>({
        { "outer_only", 0., "outer_only", 5. }, { "outer_only", 5., "outer_only", 0. },
        { "no_brim", 5., "outer_and_inner", 5. }, { "outer_and_inner", 5., "inner_only", 5. } }));
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(change.from_type, change.from_width, change.to_type, change.to_width, wall_generator);

    const std::string wall_sequence = "inner wall/outer wall";
    Print print;
    Model model;
    round_tube(print, model, brim_tube_config(wall_generator, wall_sequence, change.from_type, change.from_width), 3.);
    const std::string before = wall_order(tube_loops(print, 3.).front());
    print.apply(model, brim_tube_config(wall_generator, wall_sequence, change.to_type, change.to_width));
    print.process();
    const std::string after = wall_order(tube_loops(print, 3.).front());

    Print fresh_print;
    Model fresh_model;
    round_tube(fresh_print, fresh_model, brim_tube_config(wall_generator, wall_sequence, change.to_type, change.to_width), 3.);
    const std::string fresh = wall_order(tube_loops(fresh_print, 3.).front());

    // The change does turn the first layer around.
    REQUIRE(before != fresh);
    CHECK(after == fresh);
}

namespace {

// A layer inside the 2mm tall part, away from its top and bottom shells.
const double ramp_layer_z = 1.0;

// A 20x20x2 part, and a 10x30 modifier over its +X side whose outline crosses the part along x = 15, so the modifier
// covers the strip x = 15..20 of the part and the rest of it sticks out of the part.
Print &part_with_wall_modifier(Print &print, Model &model, const DynamicPrintConfig &config, const DynamicPrintConfig &modifier_config)
{
    Test::init_print({ make_cube(20., 20., 2.) }, print, model, config, nullptr, false);
    TriangleMesh modifier = make_cube(10., 30., 4.);
    modifier.translate(15.f, -5.f, -1.f);
    model.objects.front()->add_volume(std::move(modifier), ModelVolumeType::PARAMETER_MODIFIER, false)->config.apply(modifier_config);
    print.apply(model, config);
    print.process();
    return print;
}

// Every setting the wall positions below depend on, so none of them rests on a default.
DynamicPrintConfig wall_ramp_config(const char *wall_generator, int part_wall_loops = 2)
{
    DynamicPrintConfig config = base_config(wall_generator);
    config.set_deserialize_strict({
        { "wall_loops",            part_wall_loops },
        { "nozzle_diameter",       "0.4" },
        { "line_width",            0.45 },
        { "outer_wall_line_width", 0.45 },
        { "inner_wall_line_width", 0.45 },
        { "outer_wall_speed",      "60" },
        { "sparse_infill_density", "15%" },
        { "wall_ramp_taper_ratio", 10. },
    });
    return config;
}

DynamicPrintConfig wall_modifier_config(std::initializer_list<ConfigBase::SetDeserializeItem> items)
{
    DynamicPrintConfig config;
    config.set_deserialize_strict({ { "wall_loops", 4 } });
    config.set_deserialize_strict(items);
    return config;
}

const Layer *layer_at(const Print &print, double print_z)
{
    for (const Layer *layer : print.objects().front()->layers())
        if (std::abs(layer->print_z - print_z) < EPSILON)
            return layer;
    return nullptr;
}

// Wall length the layer extrudes within 1.5mm of the modifier's outline at x = outline_x, away from the outline of the
// object. Only walls closing the modifier's region, or the part's region around it, run there.
double buried_wall_length(const Layer &layer, double outline_x = 15.)
{
    const BoundingBox part = get_extents(layer.lslices);
    const BoundingBox strip(Point(part.min.x() + scaled<coord_t>(outline_x - 1.5), part.min.y() + scaled<coord_t>(3.)),
                            Point(part.min.x() + scaled<coord_t>(outline_x + 1.5), part.max.y() - scaled<coord_t>(3.)));
    Polylines walls;
    for (const LayerRegion *region : layer.regions())
        append(walls, region->perimeters.as_polylines());
    return unscaled<double>(total_length(intersection_pl(walls, strip.polygon())));
}

size_t regions_with_walls(const Layer &layer)
{
    return (size_t)std::count_if(layer.regions().begin(), layer.regions().end(), [](const LayerRegion *region) { return ! region->perimeters.empty(); });
}

// A box modifier over x = x0..x1 and z = z0..z1, spanning y = -5..25 so it sticks out of the parts in Y.
struct ModifierBox
{
    double             x0, x1;
    double             z0, z1;
    DynamicPrintConfig config;
};

// Touching 20x20x2 parts side by side from x = 0, the i-th at x = 20i..20i+20 with part_wall_loops[i] walls (-1 keeps
// the process value), and the modifiers in the order given. `edit` may change the object before it is sliced.
Print &slice_parts_with_modifiers(Print &print, Model &model, const DynamicPrintConfig &config, const std::vector<int> &part_wall_loops,
                                  const std::vector<ModifierBox> &modifiers, const std::function<void(ModelObject&)> &edit = {})
{
    Test::init_print({ make_cube(20., 20., 2.) }, print, model, config, nullptr, false);
    ModelObject *object = model.objects.front();
    for (size_t i = 0; i < part_wall_loops.size(); ++ i) {
        ModelVolume *part = object->volumes.front();
        if (i > 0) {
            TriangleMesh mesh = make_cube(20., 20., 2.);
            mesh.translate(float(20. * double(i)), 0.f, 0.f);
            part = object->add_volume(std::move(mesh), ModelVolumeType::MODEL_PART, false);
        }
        if (part_wall_loops[i] >= 0)
            part->config.set("wall_loops", part_wall_loops[i]);
    }
    for (const ModifierBox &box : modifiers) {
        TriangleMesh mesh = make_cube(box.x1 - box.x0, 30., box.z1 - box.z0);
        mesh.translate(float(box.x0), -5.f, float(box.z0));
        object->add_volume(std::move(mesh), ModelVolumeType::PARAMETER_MODIFIER, false)->config.apply(box.config);
    }
    if (edit)
        edit(*object);
    print.apply(model, config);
    print.process();
    return print;
}

// A modifier over the whole height of the parts.
ModifierBox modifier_box(double x0, double x1, const DynamicPrintConfig &config) { return { x0, x1, -1., 3., config }; }

// The base wall counts the regions of modifiers with a ramped transition are keyed to, in increasing order.
std::vector<int> wall_ramp_base_loops(const Print &print)
{
    const PrintObject &object = *print.objects().front();
    std::vector<int> out;
    for (size_t i = 0; i < object.num_printing_regions(); ++ i)
        if (const int base = object.printing_region(i).wall_ramp_base_loops(); base >= 0)
            out.emplace_back(base);
    std::sort(out.begin(), out.end());
    return out;
}

std::string perimeter_warnings(const Print &print)
{
    std::string text;
    for (const auto &warning : print.objects().front()->step_state_with_warnings(posPerimeters).warnings)
        text += warning.message;
    return text;
}

} // namespace

// Hard gives the modifier closed walls of its own, so walls run inside the part along the modifier's outline. The Ramp
// transition merges the modifier's region with the part's, which walls the outline of the part only.
TEST_CASE("A modifier with more walls and a ramped transition shares the part's walls instead of walling its outline inside the part", "[Perimeters][WallRamp]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print hard_print, ramp_print;
    Model hard_model, ramp_model;
    part_with_wall_modifier(hard_print, hard_model, wall_ramp_config(wall_generator), wall_modifier_config({}));
    part_with_wall_modifier(ramp_print, ramp_model, wall_ramp_config(wall_generator), wall_modifier_config({ { "wall_loops_transition", "ramp" } }));
    const Layer *hard = layer_at(hard_print, ramp_layer_z);
    const Layer *ramp = layer_at(ramp_print, ramp_layer_z);
    REQUIRE(hard != nullptr);
    REQUIRE(ramp != nullptr);

    // At least one wall of each region along the 14mm of the strip.
    CHECK(buried_wall_length(*hard) > 2. * 14.);
    CHECK(regions_with_walls(*hard) == 2);
    CHECK_THAT(buried_wall_length(*ramp), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK(regions_with_walls(*ramp) == 1);
    CHECK(perimeter_warnings(ramp_print).empty());
}

// The taper ratio only acts with the Ramp transition, so a modifier on Hard slices as one that sets neither key.
TEST_CASE("A modifier on the Hard transition slices as one without the wall count transition settings", "[Perimeters][WallRamp]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print unset_print, hard_print;
    Model unset_model, hard_model;
    part_with_wall_modifier(unset_print, unset_model, wall_ramp_config(wall_generator), wall_modifier_config({}));
    part_with_wall_modifier(hard_print, hard_model, wall_ramp_config(wall_generator),
                            wall_modifier_config({ { "wall_loops_transition", "hard" }, { "wall_ramp_taper_ratio", 5. } }));
    const SliceLengths unset = slice_lengths(unset_print);
    const SliceLengths hard  = slice_lengths(hard_print);
    REQUIRE(unset.perimeters.size() == hard.perimeters.size());
    CHECK_THAT(max_difference(unset.perimeters, hard.perimeters), Catch::Matchers::WithinAbs(0., 1.));
    CHECK_THAT(max_difference(unset.fills,      hard.fills),      Catch::Matchers::WithinAbs(0., 1.));
    CHECK(perimeter_warnings(hard_print).empty());
}

// A modifier thinning the walls merges with the part like one thickening them. Until the ramp is built, the merged
// walls are the part's.
TEST_CASE("A modifier with fewer walls and a ramped transition shares the part's walls instead of walling its outline inside the part", "[Perimeters][WallRamp]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print hard_print, ramp_print;
    Model hard_model, ramp_model;
    part_with_wall_modifier(hard_print, hard_model, wall_ramp_config(wall_generator, 4), wall_modifier_config({ { "wall_loops", 2 } }));
    part_with_wall_modifier(ramp_print, ramp_model, wall_ramp_config(wall_generator, 4),
                            wall_modifier_config({ { "wall_loops", 2 }, { "wall_loops_transition", "ramp" } }));
    const Layer *hard = layer_at(hard_print, ramp_layer_z);
    const Layer *ramp = layer_at(ramp_print, ramp_layer_z);
    REQUIRE(hard != nullptr);
    REQUIRE(ramp != nullptr);

    // At least one wall of each region along the 14mm of the strip.
    CHECK(buried_wall_length(*hard) > 2. * 14.);
    CHECK(regions_with_walls(*hard) == 2);
    CHECK_THAT(buried_wall_length(*ramp), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK(regions_with_walls(*ramp) == 1);
    CHECK(perimeter_warnings(ramp_print).empty());
}

// A modifier may thin the walls down to none. It merges with the part like any other, without a warning.
TEST_CASE("A modifier without walls and with a ramped transition shares the part's walls", "[Perimeters][WallRamp]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print hard_print, ramp_print;
    Model hard_model, ramp_model;
    part_with_wall_modifier(hard_print, hard_model, wall_ramp_config(wall_generator), wall_modifier_config({ { "wall_loops", 0 } }));
    part_with_wall_modifier(ramp_print, ramp_model, wall_ramp_config(wall_generator),
                            wall_modifier_config({ { "wall_loops", 0 }, { "wall_loops_transition", "ramp" } }));
    const Layer *hard = layer_at(hard_print, ramp_layer_z);
    const Layer *ramp = layer_at(ramp_print, ramp_layer_z);
    REQUIRE(hard != nullptr);
    REQUIRE(ramp != nullptr);

    // On Hard the part walls its side of the modifier's outline.
    CHECK(buried_wall_length(*hard) > 14.);
    CHECK_THAT(buried_wall_length(*ramp), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK(regions_with_walls(*ramp) == 1);
    CHECK(perimeter_warnings(ramp_print).empty());
}

// The Ramp transition needs a taper ratio above 0, and the part's walls can only be shared when the rest of the wall
// settings agree. Otherwise the modifier keeps its own walls, with a warning.
TEST_CASE("A modifier with a ramped transition that cannot share the part's walls keeps the hard step and warns", "[Perimeters][WallRamp]")
{
    const std::string case_name = GENERATE("zero taper ratio", "other wall speed");
    CAPTURE(case_name);
    DynamicPrintConfig modifier_config = wall_modifier_config({ { "wall_loops_transition", "ramp" } });
    if (case_name == "zero taper ratio")
        modifier_config.set_deserialize_strict({ { "wall_ramp_taper_ratio", 0. } });
    else
        modifier_config.set_deserialize_strict({ { "outer_wall_speed", "30" } });

    Print print;
    Model model;
    part_with_wall_modifier(print, model, wall_ramp_config("classic"), modifier_config);
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    CHECK(buried_wall_length(*layer) > 2. * 14.);
    CHECK(regions_with_walls(*layer) == 2);
    CHECK_FALSE(perimeter_warnings(print).empty());
}

// Over two parts with different wall counts, a modifier with a ramped transition gets a region per base wall count, and
// each one shares the walls of its own part.
TEST_CASE("A modifier with a ramped transition over parts with different wall counts shares the walls of each part", "[Perimeters][WallRamp]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config(wall_generator), { 2, 3 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" } })) });
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);

    CHECK(wall_ramp_base_loops(print) == std::vector<int>{ 2, 3 });
    CHECK_THAT(buried_wall_length(*layer, 15.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK_THAT(buried_wall_length(*layer, 25.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK(perimeter_warnings(print).empty());
}

// The base wall count only splits the modifier's regions where the parts differ.
TEST_CASE("A modifier with a ramped transition over parts with the same wall count keeps one region", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config("classic"), { 2, 2 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" } })) });
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);

    CHECK(wall_ramp_base_loops(print) == std::vector<int>{ 2 });
    CHECK_THAT(buried_wall_length(*layer, 15.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK_THAT(buried_wall_length(*layer, 25.), Catch::Matchers::WithinAbs(0., 0.5));
}

// The base wall count follows the part, so the modifier's region keeps merging with it.
TEST_CASE("Changing the part's wall count re-keys the region of a modifier with a ramped transition", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    const DynamicPrintConfig config = wall_ramp_config("classic");
    part_with_wall_modifier(print, model, config, wall_modifier_config({ { "wall_loops_transition", "ramp" } }));
    REQUIRE(wall_ramp_base_loops(print) == std::vector<int>{ 2 });

    ModelVolume *part = model.objects.front()->volumes.front();
    REQUIRE(part->is_model_part());
    part->config.set("wall_loops", 3);
    print.apply(model, config);
    CHECK(wall_ramp_base_loops(print) == std::vector<int>{ 3 });
    CHECK(print.objects().front()->is_step_done(posSlice));
    CHECK_FALSE(print.objects().front()->is_step_done(posPerimeters));

    print.process();
    REQUIRE(layer_at(print, ramp_layer_z) != nullptr);
    CHECK_THAT(buried_wall_length(*layer_at(print, ramp_layer_z)), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK(regions_with_walls(*layer_at(print, ramp_layer_z)) == 1);
}

// Switching a modifier to the Ramp transition regenerates the walls from the existing slices.
TEST_CASE("Switching a modifier to the Ramp transition regenerates its walls without reslicing", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    const DynamicPrintConfig config = wall_ramp_config("classic");
    part_with_wall_modifier(print, model, config, wall_modifier_config({}));
    REQUIRE(layer_at(print, ramp_layer_z) != nullptr);
    REQUIRE(buried_wall_length(*layer_at(print, ramp_layer_z)) > 2. * 14.);

    ModelVolume *modifier = model.objects.front()->volumes.back();
    REQUIRE(modifier->is_modifier());
    modifier->config.apply(wall_modifier_config({ { "wall_loops_transition", "ramp" } }));
    print.apply(model, config);
    CHECK(print.objects().front()->is_step_done(posSlice));
    CHECK_FALSE(print.objects().front()->is_step_done(posPerimeters));

    print.process();
    REQUIRE(layer_at(print, ramp_layer_z) != nullptr);
    CHECK_THAT(buried_wall_length(*layer_at(print, ramp_layer_z)), Catch::Matchers::WithinAbs(0., 0.5));
}

// Switching back to Hard drops the key and walls the modifier's outline again, from the existing slices.
TEST_CASE("Switching a modifier back to the Hard transition walls its outline again without reslicing", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    const DynamicPrintConfig config = wall_ramp_config("classic");
    part_with_wall_modifier(print, model, config, wall_modifier_config({ { "wall_loops_transition", "ramp" } }));
    REQUIRE(layer_at(print, ramp_layer_z) != nullptr);
    REQUIRE_THAT(buried_wall_length(*layer_at(print, ramp_layer_z)), Catch::Matchers::WithinAbs(0., 0.5));

    ModelVolume *modifier = model.objects.front()->volumes.back();
    REQUIRE(modifier->is_modifier());
    modifier->config.apply(wall_modifier_config({ { "wall_loops_transition", "hard" } }));
    print.apply(model, config);
    CHECK(wall_ramp_base_loops(print).empty());
    CHECK(print.objects().front()->is_step_done(posSlice));
    CHECK_FALSE(print.objects().front()->is_step_done(posPerimeters));

    print.process();
    REQUIRE(layer_at(print, ramp_layer_z) != nullptr);
    CHECK(buried_wall_length(*layer_at(print, ramp_layer_z)) > 2. * 14.);
}

// The taper ratio alone decides between the ramp and the hard step: 0 brings the hard step back.
TEST_CASE("Changing the taper ratio of a modifier with a ramped transition regenerates its walls without reslicing", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    const DynamicPrintConfig config = wall_ramp_config("classic");
    part_with_wall_modifier(print, model, config, wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 10. } }));
    REQUIRE(layer_at(print, ramp_layer_z) != nullptr);
    REQUIRE_THAT(buried_wall_length(*layer_at(print, ramp_layer_z)), Catch::Matchers::WithinAbs(0., 0.5));

    ModelVolume *modifier = model.objects.front()->volumes.back();
    REQUIRE(modifier->is_modifier());
    modifier->config.apply(wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 0. } }));
    print.apply(model, config);
    CHECK(print.objects().front()->is_step_done(posSlice));
    CHECK_FALSE(print.objects().front()->is_step_done(posPerimeters));

    print.process();
    REQUIRE(layer_at(print, ramp_layer_z) != nullptr);
    CHECK(buried_wall_length(*layer_at(print, ramp_layer_z)) > 2. * 14.);
    CHECK_FALSE(perimeter_warnings(print).empty());
}

// The piece of the modifier over the part with 3 walls has as many walls as the part with 2, which it must not join.
TEST_CASE("A modifier with a ramped transition shares walls only with the part it is keyed to, not with a part of its own wall count", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config("classic"), { 2, 3 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops", 2 }, { "wall_loops_transition", "ramp" } })) });
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    CHECK(wall_ramp_base_loops(print) == std::vector<int>{ 2, 3 });
    CHECK_THAT(buried_wall_length(*layer, 25.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK_THAT(buried_wall_length(*layer, 15.), Catch::Matchers::WithinAbs(0., 0.5));
}

// The part with 4 walls comes first, and the modifier over the other part has 4 walls too.
TEST_CASE("A modifier with a ramped transition is not taken by an earlier part of its own wall count", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config("classic"), { 4, 2 },
                               { modifier_box(25., 35., wall_modifier_config({ { "wall_loops_transition", "ramp" } })) });
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    CHECK_THAT(buried_wall_length(*layer, 25.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK_THAT(buried_wall_length(*layer, 35.), Catch::Matchers::WithinAbs(0., 0.5));
}

// A modifier with 6 walls inside one with 4, both with a ramped transition, on a part with 2: all three share walls.
TEST_CASE("Nested modifiers with ramped transitions share the part's walls", "[Perimeters][WallRamp]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config(wall_generator), { -1 },
                               { modifier_box(10., 30., wall_modifier_config({ { "wall_loops_transition", "ramp" } })),
                                 modifier_box(14., 30., wall_modifier_config({ { "wall_loops", 6 }, { "wall_loops_transition", "ramp" } })) });
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    CHECK_THAT(buried_wall_length(*layer, 10.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK_THAT(buried_wall_length(*layer, 14.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK(regions_with_walls(*layer) == 1);
    CHECK(perimeter_warnings(print).empty());
}

// The transition is the modifier's own setting: one nested inside a modifier with a ramped transition, which does not
// set it, keeps the hard step.
TEST_CASE("A modifier nested inside one with a ramped transition does not inherit it", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config("classic"), { -1 },
                               { modifier_box(10., 30., wall_modifier_config({ { "wall_loops_transition", "ramp" } })),
                                 modifier_box(14., 30., wall_modifier_config({ { "wall_loops", 6 } })) });
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    // Only the outer modifier is keyed.
    CHECK(wall_ramp_base_loops(print) == std::vector<int>{ 2 });
    CHECK_THAT(buried_wall_length(*layer, 10.), Catch::Matchers::WithinAbs(0., 0.5));
    CHECK(buried_wall_length(*layer, 14.) > 2. * 14.);
}

// The Ramp transition set in the process or on a part is not a modifier's own setting, so it changes nothing.
TEST_CASE("The Ramp transition set in the process or on a part has no effect", "[Perimeters][WallRamp]")
{
    const std::string where = GENERATE("process", "part");
    CAPTURE(where);

    DynamicPrintConfig ramp_config = wall_ramp_config("classic");
    if (where == "process")
        ramp_config.set_deserialize_strict({ { "wall_loops_transition", "ramp" } });
    Print plain_print, ramp_print;
    Model plain_model, ramp_model;
    slice_parts_with_modifiers(plain_print, plain_model, wall_ramp_config("classic"), { -1 }, { modifier_box(15., 25., wall_modifier_config({})) });
    slice_parts_with_modifiers(ramp_print, ramp_model, ramp_config, { -1 }, { modifier_box(15., 25., wall_modifier_config({})) },
                               [&where](ModelObject &object) {
                                   if (where == "part")
                                       object.volumes.front()->config.set_key_value("wall_loops_transition",
                                           new ConfigOptionEnum<WallLoopsTransition>(WallLoopsTransition::Ramp));
                               });
    CHECK(wall_ramp_base_loops(ramp_print).empty());
    const SliceLengths plain = slice_lengths(plain_print);
    const SliceLengths ramp  = slice_lengths(ramp_print);
    REQUIRE(plain.perimeters.size() == ramp.perimeters.size());
    CHECK_THAT(max_difference(plain.perimeters, ramp.perimeters), Catch::Matchers::WithinAbs(0., 1.));
    CHECK_THAT(max_difference(plain.fills,      ramp.fills),      Catch::Matchers::WithinAbs(0., 1.));
}

// The part supplies the walls of the merged group even where the modifier's denser infill would otherwise pick the
// modifier's settings, so the walls are those of a modifier with the part's infill.
TEST_CASE("The part supplies the walls a modifier with a ramped transition and denser infill shares", "[Perimeters][WallRamp]")
{
    Print sparse_print, dense_print;
    Model sparse_model, dense_model;
    slice_parts_with_modifiers(sparse_print, sparse_model, wall_ramp_config("classic"), { -1 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "sparse_infill_density", "15%" } })) });
    slice_parts_with_modifiers(dense_print, dense_model, wall_ramp_config("classic"), { -1 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "sparse_infill_density", "50%" } })) });
    const SliceLengths sparse = slice_lengths(sparse_print);
    const SliceLengths dense  = slice_lengths(dense_print);
    REQUIRE(sparse.perimeters.size() == dense.perimeters.size());
    CHECK_THAT(max_difference(sparse.perimeters, dense.perimeters), Catch::Matchers::WithinAbs(0., 1.));
}

// Where the modifier covers the whole cross-section of its part, it is walled on its own, and the part beside it with
// as many walls as the modifier does not take it in: the boundary between the two keeps its walls.
TEST_CASE("A modifier with a ramped transition covering its part's whole cross-section is walled on its own", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    // Covers the part at x = 0..20 from z = 1 up.
    slice_parts_with_modifiers(print, model, wall_ramp_config("classic"), { 2, 4 },
                               { ModifierBox{ -5., 20., 1., 3., wall_modifier_config({ { "wall_loops_transition", "ramp" } }) } });
    const Layer *covered = layer_at(print, 1.6);
    REQUIRE(covered != nullptr);
    CHECK(regions_with_walls(*covered) == 2);
    CHECK(buried_wall_length(*covered, 20.) > 2. * 14.);
}

// A height range changing the part's wall count keys the modifier's region per range, and both merge.
TEST_CASE("A height range with its own wall count keys a modifier with a ramped transition per range", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config("classic"), { -1 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" } })) },
                               [](ModelObject &object) {
                                   DynamicPrintConfig range;
                                   // Every layer range must carry a layer_height (see layer_height_profile_from_ranges).
                                   range.set_deserialize_strict({ { "wall_loops", 3 }, { "layer_height", 0.2 } });
                                   object.layer_config_ranges[{ 1., 2. }].assign_config(std::move(range));
                               });
    CHECK(wall_ramp_base_loops(print) == std::vector<int>{ 2, 3 });
    for (const double z : { 0.6, 1.6 }) {
        CAPTURE(z);
        const Layer *layer = layer_at(print, z);
        REQUIRE(layer != nullptr);
        CHECK_THAT(buried_wall_length(*layer), Catch::Matchers::WithinAbs(0., 0.5));
    }
}

// Fuzzy skin painting copies the modifier's region, key included, so the copy shares the walls like the original.
TEST_CASE("A fuzzy skin painted copy of a modifier region with a ramped transition keeps its key", "[Perimeters][WallRamp]")
{
    DynamicPrintConfig config = wall_ramp_config("classic");
    config.set_deserialize_strict({ { "fuzzy_skin", "external" } });
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, config, { -1 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" } })) },
                               [](ModelObject &object) {
                                   ModelVolume     &part = *object.volumes.front();
                                   TriangleSelector selector(part.mesh());
                                   selector.set_facet(0, EnforcerBlockerType::FUZZY_SKIN);
                                   part.fuzzy_skin_facets.set(selector);
                               });
    // The modifier's region and its fuzzy skin copy.
    CHECK(wall_ramp_base_loops(print) == std::vector<int>{ 2, 2 });
}

namespace {

// A wall crossing the layer's y = 0 side: how far in from the side, its place in the extrusion order, its inset and
// whether it is an outer wall.
struct SideCrossing
{
    double           depth;
    size_t           order;
    int              inset;
    bool             external;
    // The insets of all pieces of the wall there.
    std::vector<int> insets;
    // Whether the extrusion starts or ends within the strip, as the pieces of a wall do at a split.
    bool             ends_here = false;
};

// The pieces of the layer's walls within a 0.02mm wide strip across its y = 0 side at x mm from the part's -X side,
// running from 1mm outside the side to 3mm inside it, from the outside in.
std::vector<SideCrossing> raw_side_crossings(const Layer &layer, double x)
{
    const BoundingBox part = get_extents(layer.lslices);
    const BoundingBox strip(Point(part.min.x() + scaled<coord_t>(x - 0.01), part.min.y() - scaled<coord_t>(1.)),
                            Point(part.min.x() + scaled<coord_t>(x + 0.01), part.min.y() + scaled<coord_t>(3.)));
    std::vector<SideCrossing> crossings;
    size_t                    order = 0;
    for (const LayerRegion *region : layer.regions())
        for (const ExtrusionEntity *entity : region->perimeters.flatten().entities) {
            const Polylines whole     = entity->as_polylines();
            bool            ends_here = false;
            for (const Polyline &polyline : whole)
                ends_here = ends_here || strip.contains(polyline.first_point()) || strip.contains(polyline.last_point());
            for (const Polyline &piece : intersection_pl(whole, strip.polygon()))
                crossings.push_back({ unscaled<double>(piece.first_point().y() + piece.last_point().y()) / 2. - unscaled<double>(part.min.y()),
                                      order, entity->inset_idx, entity->role() == erExternalPerimeter, { entity->inset_idx }, ends_here });
            ++ order;
        }
    std::sort(crossings.begin(), crossings.end(), [](const SideCrossing &a, const SideCrossing &b) { return a.depth < b.depth; });
    return crossings;
}

// The walls the layer lays across its y = 0 side at x mm from the part's -X side, from the outside in. Pieces less than
// 0.1mm apart are one wall, split where its inset changes; it takes the order and inset of the piece extruded first.
std::vector<SideCrossing> side_crossings(const Layer &layer, double x)
{
    std::vector<SideCrossing> walls;
    for (const SideCrossing &crossing : raw_side_crossings(layer, x))
        if (walls.empty() || crossing.depth - walls.back().depth > 0.1)
            walls.push_back(crossing);
        else {
            std::vector<int> insets = std::move(walls.back().insets);
            insets.push_back(crossing.inset);
            if (crossing.order < walls.back().order)
                walls.back() = crossing;
            walls.back().insets = std::move(insets);
        }
    return walls;
}

// Where the layer lays two walls on top of each other across its y = 0 side at x mm from the part's -X side: pieces of
// two walls less than a quarter of a spacing apart, where the walls of a bead count transition squeeze closer than
// half a spacing without lying on each other. A wall that starts or ends there is a split point of the wall or the tip of
// a bead count transition, where walls meet, and one wall doubling back is not two walls.
bool walls_overlap_across_side(const Layer &layer, double x, double spacing)
{
    const std::vector<SideCrossing> crossings = raw_side_crossings(layer, x);
    for (size_t i = 1; i < crossings.size(); ++ i)
        if (crossings[i].depth - crossings[i - 1].depth < 0.25 * spacing && crossings[i].order != crossings[i - 1].order &&
            ! crossings[i].ends_here && ! crossings[i - 1].ends_here)
            return true;
    return false;
}

// Number of walls the layer lays across its y = 0 side at x mm from the part's -X side, see side_crossings().
size_t walls_across_side(const Layer &layer, double x)
{
    return side_crossings(layer, x).size();
}

// The spacing of the 0.45mm wide walls 0.2mm high of wall_ramp_config().
const double ramp_wall_spacing = 0.45 - 0.2 * (1. - 0.25 * M_PI);

// Where, from x = 17.5 outward to x = 8 mm in 0.1mm steps, the layer lays two walls on top of each other across its side.
std::string overlapping_walls(const Layer &layer)
{
    std::string overlapped;
    for (double x = 17.5; x >= 8.; x -= 0.1)
        if (walls_overlap_across_side(layer, x, ramp_wall_spacing))
            overlapped += "x=" + std::to_string(x) + "; ";
    return overlapped;
}

} // namespace

// The part has 2 walls and the modifier over x = 15..25 has 4. With a taper ratio of 5 the ramp runs 2 * 5 bead widths,
// about 4.1mm at the 0.41mm spacing of 0.45mm walls 0.2mm high, out from the modifier's outline at x = 15, shifted by a
// quarter of a drop spacing either way on alternate layers. The walls drop one at a time along it.
TEST_CASE("A modifier with more walls and a ramped transition tapers the walls from its count to the part's", "[Perimeters][WallRamp]")
{
    const double print_z = GENERATE(1.0, 1.2);
    CAPTURE(print_z);

    Print print;
    Model model;
    part_with_wall_modifier(print, model, wall_ramp_config("arachne"),
                            wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } }));
    const Layer *layer = layer_at(print, print_z);
    REQUIRE(layer != nullptr);

    // In the zone, and beyond the ramp.
    CHECK(walls_across_side(*layer, 17.5) == 4);
    CHECK(walls_across_side(*layer, 8.) == 2);
    // From the zone outward the count never grows, and passes through 3.
    std::vector<size_t> counts;
    for (double x = 17.5; x >= 8.; x -= 0.1)
        counts.emplace_back(walls_across_side(*layer, x));
    CAPTURE(counts);
    CHECK(std::is_sorted(counts.rbegin(), counts.rend()));
    CHECK(std::find(counts.begin(), counts.end(), 3) != counts.end());
    // The walls the count merges, 0.1mm apart, are not walls laid on top of each other.
    const std::string overlapped = overlapping_walls(*layer);
    CAPTURE(overlapped);
    CHECK(overlapped.empty());
    CHECK(perimeter_warnings(print).empty());
}

// Walls the band does not support yet keep the stock walls, which the object reports rather than leaving silent.
TEST_CASE("An island in a ramp zone that takes alternating extra walls keeps the stock walls and warns", "[Perimeters][WallRamp]")
{
    DynamicPrintConfig config = wall_ramp_config("arachne");
    config.set_deserialize_strict({ { "alternate_extra_wall", 1 }, { "sparse_infill_density", "15%" } });
    Print print;
    Model model;
    part_with_wall_modifier(print, model, config, wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } }));
    CHECK_THAT(perimeter_warnings(print), Catch::Matchers::ContainsSubstring("Alternate extra wall"));
}

// The band wins over only_one_wall_top: the walls of the zone are all kept, and the one-wall-top second pass does not
// add inner walls on top of them.
TEST_CASE("The ramp walls are kept in the zone where only one wall on top surfaces is enabled", "[Perimeters][WallRamp]")
{
    DynamicPrintConfig config = wall_ramp_config("arachne");
    config.set_deserialize_strict({ { "only_one_wall_top", 1 }, { "top_shell_layers", 3 } });
    Print print;
    Model model;
    part_with_wall_modifier(print, model, config, wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } }));
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    CHECK(walls_across_side(*layer, 17.5) == 4);
    CHECK(perimeter_warnings(print).empty());

    // A 1mm slab over x = 0..8 on top of the part leaves a top surface across the zone on the part's last layer
    // below it, where only one wall on top surfaces would lay inner walls of its own. The walls there are the
    // ones the option leaves out.
    auto ledge_wall_length = [&](int one_wall_top) {
        DynamicPrintConfig ledge_config = config;
        ledge_config.set_deserialize_strict({ { "only_one_wall_top", one_wall_top } });
        Print ledge_print;
        Model ledge_model;
        slice_parts_with_modifiers(ledge_print, ledge_model, ledge_config, { -1 },
                                   { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } })) },
                                   [](ModelObject &object) {
                                       TriangleMesh slab = make_cube(8., 20., 1.);
                                       slab.translate(0.f, 0.f, 2.f);
                                       object.add_volume(std::move(slab), ModelVolumeType::MODEL_PART, false);
                                   });
        const Layer *ledge = layer_at(ledge_print, 2.0);
        REQUIRE(ledge != nullptr);
        Polylines walls;
        for (const LayerRegion *region : ledge->regions())
            append(walls, region->perimeters.as_polylines());
        return unscaled<double>(total_length(walls));
    };
    CHECK_THAT(ledge_wall_length(1), Catch::Matchers::WithinAbs(ledge_wall_length(0), 1.));
}

// Alternate extra wall adds a wall only with sparse infill to alternate it with, as in process_arachne().
TEST_CASE("The ramp walls apply without a warning where alternate extra wall has no sparse infill to act on", "[Perimeters][WallRamp]")
{
    DynamicPrintConfig config = wall_ramp_config("arachne");
    config.set_deserialize_strict({ { "alternate_extra_wall", 1 }, { "sparse_infill_density", "0%" } });
    Print print;
    Model model;
    part_with_wall_modifier(print, model, config, wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } }));
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    CHECK(walls_across_side(*layer, 17.5) == 4);
    CHECK(perimeter_warnings(print).empty());
}

// Only one wall on top surfaces set on a modifier that keeps all its walls under top surfaces contradicts it.
TEST_CASE("Print validation rejects a ramp modifier with its own only one wall on top surfaces setting", "[Perimeters][WallRamp]")
{
    struct Variant { const char *name; const char *transition; bool explicit_on_modifier; int wall_loops; bool error; };
    const Variant variant = GENERATE(
        Variant{ "explicit on a ramp modifier",        "ramp", true,  4, true },
        Variant{ "inherited from the process",         "ramp", false, 4, false },
        Variant{ "explicit on a hard modifier",        "hard", true,  4, false },
        Variant{ "explicit on a one wall modifier",    "ramp", true,  1, false });
    CAPTURE(variant.name);

    DynamicPrintConfig config = wall_ramp_config("arachne");
    config.set_deserialize_strict({ { "only_one_wall_top", variant.explicit_on_modifier ? 0 : 1 }, { "top_shell_layers", 3 } });
    DynamicPrintConfig modifier = wall_modifier_config({ { "wall_loops", variant.wall_loops }, { "wall_loops_transition", variant.transition },
                                                         { "wall_ramp_taper_ratio", 5. } });
    if (variant.explicit_on_modifier)
        modifier.set_deserialize_strict({ { "only_one_wall_top", 1 } });
    Print print;
    Model model;
    part_with_wall_modifier(print, model, config, modifier);
    const std::string error = print.validate().string;
    // The test config trips unrelated validation errors, so look for this one only.
    const bool reported = error.find("uses the Ramp wall count transition") != std::string::npos;
    CHECK(reported == variant.error);
}

TEST_CASE("A steep ramp raises the transition angle no further than the option allows", "[Perimeters][WallRamp]")
{
    const double max_angle = print_config_def.get("wall_transition_angle")->max;
    CHECK_THAT(wall_band_transition_angle(10.f, 0.5), Catch::Matchers::WithinAbs(max_angle, 1e-3));
    // A gentle ramp raises it only by what it needs.
    CHECK_THAT(wall_band_transition_angle(10.f, 10.), Catch::Matchers::WithinAbs(std::atan(0.1) * 180. / M_PI + 5., 1e-3));

    Print print;
    Model model;
    part_with_wall_modifier(print, model, wall_ramp_config("arachne"),
                            wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 0.5 } }));
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);
    CHECK(walls_across_side(*layer, 17.5) == 4);
}

// The zone spans the object's height, so a modifier cannot take one wall count over one height range and another over
// the next.
TEST_CASE("A modifier with a ramped transition whose base wall count differs between height ranges keeps the stock walls and warns", "[Perimeters][WallRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, wall_ramp_config("arachne"), { -1 },
                               { modifier_box(15., 25., wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } })) },
                               [](ModelObject &object) {
                                   DynamicPrintConfig range;
                                   range.set_deserialize_strict({ { "wall_loops", 3 }, { "layer_height", 0.2 } });
                                   object.layer_config_ranges[{ 1., 2. }].assign_config(std::move(range));
                               });
    const Layer *lower = layer_at(print, 0.6);
    const Layer *upper = layer_at(print, 1.6);
    REQUIRE(lower != nullptr);
    REQUIRE(upper != nullptr);
    // The part's 2 walls below z = 1 and 3 above, also inside the modifier.
    CHECK(walls_across_side(*lower, 17.5) == 2);
    CHECK(walls_across_side(*upper, 17.5) == 3);
    CHECK_THAT(perimeter_warnings(print), Catch::Matchers::ContainsSubstring("every height range"));
}

namespace {

// The ramp fixture of the tests below: the part's 2 walls rising to 4 in the modifier, over a ramp 5 bead widths
// long per wall.
DynamicPrintConfig wall_ramp_modifier_config()
{
    return wall_modifier_config({ { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } });
}

// Length of the G-code's extrusions of one feature type, such as "Outer wall".
double feature_length(const std::string &gcode, const std::string &feature)
{
    double      length = 0.;
    std::string current;
    GCodeReader reader;
    reader.parse_buffer(gcode, [&](GCodeReader &self, const GCodeReader::GCodeLine &line) {
        const std::string &raw = line.raw();
        for (const std::string_view tag : { std::string_view("; FEATURE: "), std::string_view(";TYPE:") })
            if (raw.rfind(tag, 0) == 0)
                current = raw.substr(tag.size());
        if (current == feature && line.extruding(self))
            length += line.dist_XY(self);
    });
    return length;
}

// How far in from the part's y = 0 side `area` starts at x mm from the part's -X side.
double depth_from_side(const Layer &layer, const ExPolygons &area, double x)
{
    const BoundingBox part = get_extents(layer.lslices);
    const coord_t     px   = part.min.x() + scaled<coord_t>(x);
    const Polyline    probe(Point(px, part.min.y() - scaled<coord_t>(1.)), Point(px, part.center().y()));
    coord_t           first = part.center().y();
    for (const Polyline &piece : intersection_pl(Polylines{ probe }, area))
        first = std::min({ first, piece.first_point().y(), piece.last_point().y() });
    return unscaled<double>(first - part.min.y());
}

// How far the infill runs into the walls at x mm from the part's -X side: the region the walls leave unfilled
// minus the infill region, measured in from the y = 0 side.
double infill_wall_overlap_at(const Layer &layer, double x)
{
    ExPolygons infill, unfilled;
    for (const LayerRegion *region : layer.regions()) {
        append(infill, region->fill_expolygons);
        append(unfilled, region->fill_no_overlap_expolygons);
    }
    return depth_from_side(layer, union_ex(unfilled), x) - depth_from_side(layer, union_ex(infill), x);
}

} // namespace

// Arachne walls the band from both of its edges, but only its edge on the slice contour is an outer wall.
TEST_CASE("A layer with a ramp band has one closed outer wall per contour", "[Perimeters][WallRamp]")
{
    const double print_z = GENERATE(1.0, 1.2);
    CAPTURE(print_z);

    Print print;
    Model model;
    part_with_wall_modifier(print, model, wall_ramp_config("arachne"), wall_ramp_modifier_config());
    const Layer *layer = layer_at(print, print_z);
    REQUIRE(layer != nullptr);

    size_t contours = 0;
    for (const ExPolygon &island : layer->lslices)
        contours += 1 + island.holes.size();
    size_t outer_loops = 0, outer_open = 0;
    for (const LayerRegion *region : layer->regions())
        for (const ExtrusionEntity *entity : region->perimeters.flatten().entities)
            if (entity->role() == erExternalPerimeter)
                ++ (entity->is_loop() ? outer_loops : outer_open);
    CHECK(contours == 1);
    CHECK(outer_loops == contours);
    CHECK(outer_open == 0);

    // The G-code lays one outer wall loop, 0.45mm wide, around the 20x20 contour, give or take the seam gap.
    const double outer_wall = feature_length(layer_gcode(gcode(print), print_z), "Outer wall");
    CHECK_THAT(outer_wall, Catch::Matchers::WithinAbs(4. * (20. - 0.45), 1.));
}

// Inset 0 is the outer wall and each wall further in takes the next inset, through the zone, the ramp and beyond.
TEST_CASE("A layer with a ramp band numbers its walls inward from the outer wall", "[Perimeters][WallRamp]")
{
    const double print_z = GENERATE(1.0, 1.2);
    CAPTURE(print_z);

    Print print;
    Model model;
    part_with_wall_modifier(print, model, wall_ramp_config("arachne"), wall_ramp_modifier_config());
    const Layer *layer = layer_at(print, print_z);
    REQUIRE(layer != nullptr);

    REQUIRE(walls_across_side(*layer, 17.5) == 4);
    // Where the walls across the side, from the outside in, are not insets 0, 1, 2...
    std::string misnumbered;
    for (double x = 17.5; x >= 8.; x -= 0.1) {
        std::string insets;
        bool        in_order = true;
        int         next     = 0;
        // Where a wall is split, either piece may take its place.
        for (const SideCrossing &wall : side_crossings(*layer, x)) {
            in_order = in_order && std::find(wall.insets.begin(), wall.insets.end(), next ++) != wall.insets.end();
            insets += " " + std::to_string(wall.inset);
        }
        if (! in_order)
            misnumbered += "x=" + std::to_string(x) + ":" + insets + "; ";
    }
    CAPTURE(misnumbered);
    CHECK(misnumbered.empty());
    const std::string overlapped = overlapping_walls(*layer);
    CAPTURE(overlapped);
    CHECK(overlapped.empty());
}

// The infill overlaps the walls by a share of the inner wall spacing wherever there is more than one wall. A band
// numbered from both edges would count half its walls, and use the outer wall spacing at 2 walls.
TEST_CASE("The infill overlaps the walls of a ramp band as it does stock walls of the same count", "[Perimeters][WallRamp]")
{
    // Outer walls narrower than the inner ones, so an overlap taken from the outer wall spacing shows.
    DynamicPrintConfig config = wall_ramp_config("arachne", 1);
    config.set_deserialize_strict({ { "outer_wall_line_width", 0.42 }, { "inner_wall_line_width", 0.5 }, { "line_width", 0.5 },
                                    { "infill_wall_overlap", "50%" } });

    Print stock_print, ramp_print;
    Model stock_model, ramp_model;
    slice_parts_with_modifiers(stock_print, stock_model, config, { 2 }, {});
    part_with_wall_modifier(ramp_print, ramp_model, config,
                            wall_modifier_config({ { "wall_loops", 2 }, { "wall_loops_transition", "ramp" }, { "wall_ramp_taper_ratio", 5. } }));
    const Layer *stock = layer_at(stock_print, ramp_layer_z);
    const Layer *ramp  = layer_at(ramp_print, ramp_layer_z);
    REQUIRE(stock != nullptr);
    REQUIRE(ramp != nullptr);

    // In the zone the band has the stock layer's 2 walls.
    REQUIRE(walls_across_side(*stock, 17.5) == 2);
    REQUIRE(walls_across_side(*ramp, 17.5) == 2);
    const double stock_overlap = infill_wall_overlap_at(*stock, 17.5);
    // Half the inner wall spacing: 0.5mm wide, 0.2mm high.
    CHECK_THAT(stock_overlap, Catch::Matchers::WithinAbs(0.5 * (0.5 - 0.2 * (1. - 0.25 * M_PI)), 0.01));
    CHECK_THAT(infill_wall_overlap_at(*ramp, 17.5), Catch::Matchers::WithinAbs(stock_overlap, 0.01));
}

// Each wall sequence orders the band's walls as it orders stock walls, by their depth from the outer wall.
TEST_CASE("A layer with a ramp band prints its walls in the wall sequence order", "[Perimeters][WallRamp]")
{
    const std::string wall_sequence = GENERATE("inner wall/outer wall", "outer wall/inner wall", "inner-outer-inner wall");
    CAPTURE(wall_sequence);

    DynamicPrintConfig config = wall_ramp_config("arachne");
    config.set_deserialize_strict({ { "wall_sequence", wall_sequence } });
    Print print;
    Model model;
    part_with_wall_modifier(print, model, config, wall_ramp_modifier_config());
    const Layer *layer = layer_at(print, ramp_layer_z);
    REQUIRE(layer != nullptr);

    // In the zone, the 4 walls by depth, then in extrusion order.
    std::vector<SideCrossing> walls = side_crossings(*layer, 17.5);
    REQUIRE(walls.size() == 4);
    std::vector<std::pair<size_t, int>> order_and_depth;
    for (size_t depth = 0; depth < walls.size(); ++ depth)
        order_and_depth.emplace_back(walls[depth].order, int(depth));
    std::sort(order_and_depth.begin(), order_and_depth.end());
    std::vector<int> printed;
    for (const auto &[order, depth] : order_and_depth)
        printed.push_back(depth);
    CHECK(printed == wall_sequence_insets(wall_sequence, false, 3));
}

// A piece of a wall shorter than a bead cannot be printed as one, so the wall's neighbouring piece takes it over.
TEST_CASE("A layer with a ramp band has no wall piece shorter than the wall spacing", "[Perimeters][WallRamp]")
{
    const double print_z = GENERATE(1.0, 1.2);
    CAPTURE(print_z);

    Print print;
    Model model;
    part_with_wall_modifier(print, model, wall_ramp_config("arachne"), wall_ramp_modifier_config());
    const Layer *layer = layer_at(print, print_z);
    REQUIRE(layer != nullptr);

    // Only the pieces of a wall split where its depth changes are merged when short; an open Arachne wall that is
    // short to begin with is left alone. A split piece ends at a point it shares with the neighbouring piece of the
    // same wall, which has another inset, so the check is on open walls that meet another open wall of a different
    // inset at one of their ends.
    // flatten() returns a collection that owns its entities, so keep each one alive while `open` points into it.
    std::vector<ExtrusionEntityCollection> walls;
    for (const LayerRegion *region : layer->regions())
        walls.emplace_back(region->perimeters.flatten());
    std::vector<const ExtrusionEntity *> open;
    for (const ExtrusionEntityCollection &collection : walls)
        for (const ExtrusionEntity *entity : collection.entities)
            if (! entity->is_loop())
                open.push_back(entity);
    size_t pieces = 0;
    double shortest = std::numeric_limits<double>::max();
    for (const ExtrusionEntity *entity : open) {
        const bool split = std::any_of(open.begin(), open.end(), [entity](const ExtrusionEntity *other) {
            return other != entity && other->inset_idx != entity->inset_idx &&
                   (other->first_point() == entity->first_point() || other->first_point() == entity->last_point() ||
                    other->last_point() == entity->first_point() || other->last_point() == entity->last_point());
        });
        if (split) {
            ++ pieces;
            shortest = std::min(shortest, unscaled<double>(total_length(entity->as_polylines())));
        }
    }
    REQUIRE(pieces > 0);
    CHECK(shortest >= ramp_wall_spacing);
}
