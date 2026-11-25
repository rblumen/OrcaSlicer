#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/Config.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Flow.hpp"
#include "libslic3r/Layer.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/PrintBase.hpp"
#include "libslic3r/PrintConfig.hpp"
#include "libslic3r/ShellDepthField.hpp"
#include "libslic3r/Surface.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/TriangleSelector.hpp"
#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"

#include <admesh/stl.h>
#include <boost/algorithm/string/predicate.hpp>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "test_helpers.hpp"

using namespace Slic3r;
using namespace Slic3r::Test;

namespace {

// Every setting the shell counts and the region keys depend on, so none of them rests on a default. 0.2mm layers make
// the 2mm tall parts below 10 layers.
DynamicPrintConfig shell_config(const char *wall_generator = "arachne")
{
    DynamicPrintConfig config = DynamicPrintConfig::full_print_config();
    config.set_deserialize_strict({
        { "wall_generator",             wall_generator },
        { "layer_height",               0.2 },
        { "initial_layer_print_height", 0.2 },
        { "wall_loops",                 2 },
        { "sparse_infill_density",      "15%" },
        { "top_shell_layers",           3 },
        { "top_shell_thickness",        0.4 },
        { "bottom_shell_layers",        3 },
        { "bottom_shell_thickness",     0.2 },
        { "ensure_vertical_shell_thickness", "ensure_all" },
        { "top_shell_ramp_taper_ratio",    10. },
        { "bottom_shell_ramp_taper_ratio", 10. },
    });
    return config;
}

// A modifier deepening both shells to 5 layers, with the given further settings.
DynamicPrintConfig shell_modifier_config(std::initializer_list<ConfigBase::SetDeserializeItem> items)
{
    DynamicPrintConfig config;
    config.set_deserialize_strict({ { "top_shell_layers", 5 }, { "bottom_shell_layers", 5 } });
    config.set_deserialize_strict(items);
    return config;
}

// A box modifier over x = x0..x1 and z = z0..z1, spanning y = -5..25 so it sticks out of the parts in Y.
struct ModifierBox
{
    double             x0, x1;
    double             z0, z1;
    DynamicPrintConfig config;
};

// A modifier over the whole height of the parts.
ModifierBox modifier_box(double x0, double x1, const DynamicPrintConfig &config) { return { x0, x1, -1., 3., config }; }

void add_modifiers(ModelObject &object, const std::vector<ModifierBox> &modifiers)
{
    for (const ModifierBox &box : modifiers) {
        TriangleMesh mesh = make_cube(box.x1 - box.x0, 30., box.z1 - box.z0);
        mesh.translate(float(box.x0), -5.f, float(box.z0));
        object.add_volume(std::move(mesh), ModelVolumeType::PARAMETER_MODIFIER, false)->config.apply(box.config);
    }
}

// Touching 20x20x2 parts side by side from x = 0, the i-th at x = 20i..20i+20 with the config part_configs[i], and the
// modifiers in the order given. `edit` may change the object before it is applied. Only applied, not processed.
void apply_parts_with_modifiers(Print &print, Model &model, const DynamicPrintConfig &config, const std::vector<DynamicPrintConfig> &part_configs,
                                const std::vector<ModifierBox> &modifiers, const std::function<void(ModelObject&)> &edit = {})
{
    Test::init_print({ make_cube(20., 20., 2.) }, print, model, config, nullptr, false);
    ModelObject *object = model.objects.front();
    for (size_t i = 0; i < part_configs.size(); ++ i) {
        ModelVolume *part = object->volumes.front();
        if (i > 0) {
            TriangleMesh mesh = make_cube(20., 20., 2.);
            mesh.translate(float(20. * double(i)), 0.f, 0.f);
            part = object->add_volume(std::move(mesh), ModelVolumeType::MODEL_PART, false);
        }
        part->config.apply(part_configs[i]);
    }
    add_modifiers(*object, modifiers);
    if (edit)
        edit(*object);
    print.apply(model, config);
}

Print &slice_parts_with_modifiers(Print &print, Model &model, const DynamicPrintConfig &config, const std::vector<DynamicPrintConfig> &part_configs,
                                  const std::vector<ModifierBox> &modifiers, const std::function<void(ModelObject&)> &edit = {})
{
    apply_parts_with_modifiers(print, model, config, part_configs, modifiers, edit);
    print.process();
    return print;
}

// One object of the given parts, in the frame they are given in, with the modifiers in the order given, sliced.
const PrintObject &slice_meshes_with_modifiers(Print &print, Model &model, const DynamicPrintConfig &config, std::vector<TriangleMesh> parts,
                                               const std::vector<ModifierBox> &modifiers)
{
    Test::init_print({ parts.front() }, print, model, config, nullptr, false);
    ModelObject *object = model.objects.front();
    for (size_t i = 1; i < parts.size(); ++ i)
        object->add_volume(std::move(parts[i]), ModelVolumeType::MODEL_PART, false);
    add_modifiers(*object, modifiers);
    print.apply(model, config);
    print.process();
    return *print.objects().front();
}

// Samples the solid infill of a sliced object, at points given in mm from the corner of its bounding box in XY.
struct ShellProbe
{
    const PrintObject &object;
    Point              origin;

    explicit ShellProbe(const PrintObject &object) : object(object)
    {
        BoundingBox box;
        for (const Layer *layer : object.layers())
            box.merge(get_extents(layer->lslices));
        origin = box.min;
    }

    Point at(double x, double y) const { return Point(origin.x() + scaled(x), origin.y() + scaled(y)); }
    // Whether any region fills the layer at p, with infill of a type passing `filter`, rather than walls or nothing.
    bool filled(size_t layer, const Point &p, bool (*filter)(const Surface &)) const
    {
        for (const LayerRegion *layerm : object.get_layer(int(layer))->regions())
            for (const Surface &surface : layerm->fill_surfaces.surfaces)
                if (filter(surface) && surface.expolygon.contains(p))
                    return true;
        return false;
    }
    bool inside(size_t layer, const Point &p) const { return this->filled(layer, p, [](const Surface &) { return true; }); }
    // Top, bottom, bridge or internal solid infill.
    bool solid(size_t layer, const Point &p) const { return this->filled(layer, p, [](const Surface &surface) { return surface.is_solid(); }); }
    // How many layers in a row are solid at (x, y), from the topmost (bottommost) layer filled there down (up). On a slope
    // the layers above (below) only wall the point.
    size_t top_depth(double x, double y = 10.) const
    {
        const Point p = this->at(x, y);
        int         l = int(object.layers().size()) - 1;
        while (l >= 0 && ! this->inside(size_t(l), p))
            -- l;
        size_t depth = 0;
        for (; l >= 0 && this->solid(size_t(l), p); -- l)
            ++ depth;
        return depth;
    }
    size_t bottom_depth(double x, double y = 10.) const
    {
        const Point p = this->at(x, y);
        size_t      l = 0;
        while (l < object.layers().size() && ! this->inside(l, p))
            ++ l;
        size_t depth = 0;
        for (; l < object.layers().size() && this->solid(l, p); ++ l)
            ++ depth;
        return depth;
    }
    // How far in mm the solid infill of a layer runs on along y = 10 from x, in the direction dir (+1 or -1).
    double solid_run(size_t layer, double x, double dir) const
    {
        double d = 0.;
        while (d < 20. && this->solid(layer, this->at(x + dir * d, 10.)))
            d += 0.05;
        return d;
    }
    // Where along y = 10 the solid infill of a layer ends, coming from `far` towards x = near; how far that is from near,
    // or 0 if there is none in between. Unlike solid_run() it does not stop at walls next to near.
    double solid_edge(size_t layer, double near, double far) const
    {
        const double dir = far > near ? 1. : -1.;
        for (double d = std::abs(far - near); d > 0.; d -= 0.05)
            if (this->solid(layer, this->at(near + dir * d, 10.)))
                return d;
        return 0.;
    }
};

// Number of walls a layer lays across its y = 0 side at x mm from its -X side: pieces of walls crossing a 0.02 mm wide
// strip from 1 mm outside the side to 3 mm inside it, pieces less than 0.1 mm apart counted as one wall.
size_t walls_across_side(const Layer &layer, double x)
{
    const BoundingBox part = get_extents(layer.lslices);
    const BoundingBox strip(Point(part.min.x() + scaled<coord_t>(x - 0.01), part.min.y() - scaled<coord_t>(1.)),
                            Point(part.min.x() + scaled<coord_t>(x + 0.01), part.min.y() + scaled<coord_t>(3.)));
    std::vector<coord_t> depths;
    for (const LayerRegion *region : layer.regions())
        for (const Polyline &piece : intersection_pl(region->perimeters.as_polylines(), strip.polygon()))
            depths.emplace_back((piece.first_point().y() + piece.last_point().y()) / 2);
    std::sort(depths.begin(), depths.end());
    size_t walls = 0;
    for (size_t i = 0; i < depths.size(); ++ i)
        if (i == 0 || depths[i] - depths[i - 1] > scaled<coord_t>(0.1))
            ++ walls;
    return walls;
}

// The fixture of the Shell Ramp projection tests: a 60 x 20 x 4 mm strap, 20 layers of 0.2 mm, its top and bottom shells
// 3 layers with a 0.4 mm and 0.2 mm minimum, so 0.6 mm deep, and a modifier box over x = 25..35 through its whole height.
DynamicPrintConfig projection_config(const char *ensure_vertical_shell_thickness, const char *wall_generator = "arachne")
{
    DynamicPrintConfig config = shell_config(wall_generator);
    config.set_deserialize_strict({ { "ensure_vertical_shell_thickness", ensure_vertical_shell_thickness } });
    return config;
}

// A modifier over x = 25..35 setting a 7 layer (1.4 mm) top shell on the Ramp with a taper ratio of 5: its ramp runs
// (1.4 - 0.6) * 5 = 4 mm outside it, over which the shell steps from 7 layers down to 3, the layer k layers below the top
// in the shell out to 7 - k mm from the modifier.
ModifierBox top_ramp_modifier(std::initializer_list<ConfigBase::SetDeserializeItem> items = { { "top_shell_layers", 7 } }, double z0 = -1.)
{
    DynamicPrintConfig config;
    config.set_deserialize_strict({ { "top_shell_layers_transition", "ramp" }, { "top_shell_ramp_taper_ratio", 5. } });
    config.set_deserialize_strict(items);
    return { 25., 35., z0, 7., config };
}

// One part, of the process's settings.
const std::vector<DynamicPrintConfig> one_part { DynamicPrintConfig() };

DynamicPrintConfig part_config(std::initializer_list<ConfigBase::SetDeserializeItem> items)
{
    DynamicPrintConfig config;
    config.set_deserialize_strict(items);
    return config;
}

// The regions whose top (bottom) shell is keyed to a base count, that base count in increasing order.
std::vector<int> top_shell_bases(const Print &print)
{
    const PrintObject &object = *print.objects().front();
    std::vector<int> out;
    for (size_t i = 0; i < object.num_printing_regions(); ++ i)
        if (const int base = object.printing_region(i).ramp_key().top_base_layers; base >= 0)
            out.emplace_back(base);
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<int> bottom_shell_bases(const Print &print)
{
    const PrintObject &object = *print.objects().front();
    std::vector<int> out;
    for (size_t i = 0; i < object.num_printing_regions(); ++ i)
        if (const int base = object.printing_region(i).ramp_key().bottom_base_layers; base >= 0)
            out.emplace_back(base);
    std::sort(out.begin(), out.end());
    return out;
}

// The region of the object whose config sets the given top shell count.
const PrintRegion *region_with_top_shell_layers(const Print &print, int top_shell_layers)
{
    const PrintObject &object = *print.objects().front();
    for (size_t i = 0; i < object.num_printing_regions(); ++ i)
        if (object.printing_region(i).config().top_shell_layers.value == top_shell_layers)
            return &object.printing_region(i);
    return nullptr;
}

std::string prepare_infill_warnings(const Print &print)
{
    std::string text;
    for (const auto &warning : print.objects().front()->step_state_with_warnings(posPrepareInfill).warnings)
        text += warning.message;
    return text;
}

// The G-code's lines that are not comments, and the comments that mark the extrusion role. The other comments carry
// the header's time stamp and version, and the config block, which lists the process's Ramp settings.
std::vector<std::string> gcode_commands(const std::string &gcode)
{
    std::vector<std::string> out;
    std::istringstream       in(gcode);
    std::string              line;
    while (std::getline(in, line))
        if (! line.empty() && (line.front() != ';' || boost::starts_with(line, ";TYPE:") || boost::starts_with(line, "; FEATURE:")))
            out.push_back(line);
    return out;
}

// The first command two G-codes differ at, empty if they are the same.
std::string first_difference(const std::vector<std::string> &a, const std::vector<std::string> &b)
{
    for (size_t i = 0; i < std::min(a.size(), b.size()); ++ i)
        if (a[i] != b[i])
            return "command " + std::to_string(i) + ": " + a[i] + " | " + b[i];
    return a.size() == b.size() ? std::string() : "lengths " + std::to_string(a.size()) + " | " + std::to_string(b.size());
}

} // namespace

// The part has 3 top layers with a 0.4mm minimum and 3 bottom layers with a 0.2mm minimum; the modifier sets 5 of each.
// The key carries the part's counts, the modifier region keeps its own config, and its default shell is the part's.
TEST_CASE("A modifier on the Ramp shell transitions is keyed to the shell counts of the region it modifies", "[Shells][ShellRamp]")
{
    const char *wall_generator = GENERATE("classic", "arachne");
    CAPTURE(wall_generator);

    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, shell_config(wall_generator), one_part,
        { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "ramp" }, { "bottom_shell_layers_transition", "ramp" } })) });
    const PrintRegion *modifier = region_with_top_shell_layers(print, 5);
    REQUIRE(modifier != nullptr);
    const RampKey &key = modifier->ramp_key();
    CHECK(key.top_base_layers == 3);
    CHECK_THAT(key.top_base_thickness, Catch::Matchers::WithinAbs(0.4, 1e-9));
    CHECK(key.bottom_base_layers == 3);
    CHECK_THAT(key.bottom_base_thickness, Catch::Matchers::WithinAbs(0.2, 1e-9));
    // The walls are not on the Ramp.
    CHECK(key.wall_base == -1);
    CHECK_FALSE(modifier->has_wall_ramp());
    CHECK(modifier->has_top_shell_ramp(0.2));
    CHECK(modifier->has_bottom_shell_ramp(0.2));
    CHECK(modifier->config().bottom_shell_layers.value == 5);

    const ShellCounts counts = modifier->shell_default_counts(0.2);
    CHECK(counts.top_layers == 3);
    CHECK_THAT(counts.top_thickness, Catch::Matchers::WithinAbs(0.4, 1e-9));
    CHECK(counts.bottom_layers == 3);
    CHECK_THAT(counts.bottom_thickness, Catch::Matchers::WithinAbs(0.2, 1e-9));
}

// Each shell transition keys only its own part of the key, and a region not on the Ramp builds its own shell.
TEST_CASE("Each shell transition keys only its own shell", "[Shells][ShellRamp]")
{
    struct Variant { const char *top; const char *bottom; };
    const Variant variant = GENERATE(Variant{ "ramp", "hard" }, Variant{ "hard", "ramp" }, Variant{ "hard", "hard" });
    CAPTURE(variant.top, variant.bottom);

    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, shell_config(), one_part,
        { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", variant.top },
                                                         { "bottom_shell_layers_transition", variant.bottom } })) });
    const bool top_ramp    = std::string(variant.top) == "ramp";
    const bool bottom_ramp = std::string(variant.bottom) == "ramp";
    CHECK(top_shell_bases(print) == (top_ramp ? std::vector<int>{ 3 } : std::vector<int>{}));
    CHECK(bottom_shell_bases(print) == (bottom_ramp ? std::vector<int>{ 3 } : std::vector<int>{}));

    const PrintRegion *modifier = region_with_top_shell_layers(print, 5);
    REQUIRE(modifier != nullptr);
    CHECK(modifier->has_top_shell_ramp(0.2) == top_ramp);
    CHECK(modifier->has_bottom_shell_ramp(0.2) == bottom_ramp);
    const ShellCounts counts = modifier->shell_default_counts(0.2);
    CHECK(counts.top_layers == (top_ramp ? 3 : 5));
    CHECK(counts.bottom_layers == (bottom_ramp ? 3 : 5));
    // The part is never keyed and builds its own shell.
    const PrintRegion *part = region_with_top_shell_layers(print, 3);
    REQUIRE(part != nullptr);
    CHECK(part->ramp_key() == RampKey{});
    CHECK(part->shell_default_counts(0.2).top_layers == 3);
}

// The same modifier config over parts with 3 and 4 top layers would otherwise make one region, losing the count the
// ramp starts from on one of the parts.
TEST_CASE("A modifier on the Ramp shell transition over parts with different shell counts gets a region per part", "[Shells][ShellRamp]")
{
    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, shell_config(),
        { part_config({ { "top_shell_layers", 3 }, { "bottom_shell_layers", 3 } }), part_config({ { "top_shell_layers", 4 }, { "bottom_shell_layers", 2 } }) },
        { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "ramp" }, { "bottom_shell_layers_transition", "ramp" } })) });
    CHECK(top_shell_bases(print) == std::vector<int>{ 3, 4 });
    CHECK(bottom_shell_bases(print) == std::vector<int>{ 2, 3 });
}

// A height range changing the part's top shell count keys the modifier's region per range.
TEST_CASE("A height range with its own shell count keys a modifier on the Ramp shell transition per range", "[Shells][ShellRamp]")
{
    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, shell_config(), one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "ramp" } })) },
                               [](ModelObject &object) {
                                   DynamicPrintConfig range;
                                   // Every layer range must carry a layer_height (see layer_height_profile_from_ranges).
                                   range.set_deserialize_strict({ { "top_shell_layers", 4 }, { "layer_height", 0.2 } });
                                   object.layer_config_ranges[{ 1., 2. }].assign_config(std::move(range));
                               });
    CHECK(top_shell_bases(print) == std::vector<int>{ 3, 4 });
}

// Fuzzy skin painting copies the modifier's region, key included.
TEST_CASE("A fuzzy skin painted copy of a modifier region on the Ramp shell transition keeps its key", "[Shells][ShellRamp]")
{
    DynamicPrintConfig config = shell_config();
    config.set_deserialize_strict({ { "fuzzy_skin", "external" } });
    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, config, one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "ramp" } })) },
                               [](ModelObject &object) {
                                   ModelVolume     &part = *object.volumes.front();
                                   TriangleSelector selector(part.mesh());
                                   selector.set_facet(0, EnforcerBlockerType::FUZZY_SKIN);
                                   part.fuzzy_skin_facets.set(selector);
                               });
    // The modifier's region and its fuzzy skin copy.
    CHECK(top_shell_bases(print) == std::vector<int>{ 3, 3 });
}

// The transitions only act set on a modifier: set in the process or on the part they key nothing.
TEST_CASE("The Ramp shell transitions set in the process or on a part key no region", "[Shells][ShellRamp]")
{
    const std::string where = GENERATE("process", "part");
    CAPTURE(where);

    DynamicPrintConfig config = shell_config();
    if (where == "process")
        config.set_deserialize_strict({ { "top_shell_layers_transition", "ramp" }, { "bottom_shell_layers_transition", "ramp" } });
    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, config, one_part, { modifier_box(15., 25., shell_modifier_config({})) },
                               [&where](ModelObject &object) {
                                   if (where == "part")
                                       object.volumes.front()->config.apply(part_config(
                                           { { "top_shell_layers_transition", "ramp" }, { "bottom_shell_layers_transition", "ramp" } }));
                               });
    CHECK(top_shell_bases(print).empty());
    CHECK(bottom_shell_bases(print).empty());
}

// The Shell Ramp is applied when the shells are projected, so switching the transition or changing the taper ratio
// keeps the slices and the walls and redoes only the infill regions.
TEST_CASE("Switching a modifier's shell transition or taper ratio re-keys it without reslicing", "[Shells][ShellRamp]")
{
    struct Variant { const char *transition_key; const char *taper_key; };
    const Variant variant = GENERATE(Variant{ "top_shell_layers_transition", "top_shell_ramp_taper_ratio" },
                                     Variant{ "bottom_shell_layers_transition", "bottom_shell_ramp_taper_ratio" });
    const std::string change = GENERATE("hard to ramp", "ramp to hard", "taper ratio");
    CAPTURE(variant.transition_key, change);
    const bool top = std::string(variant.transition_key) == "top_shell_layers_transition";
    auto bases = [top](const Print &print) { return top ? top_shell_bases(print) : bottom_shell_bases(print); };

    const char *before = change == "hard to ramp" ? "hard" : "ramp";
    const char *after  = change == "ramp to hard" ? "hard" : "ramp";
    Print print;
    Model model;
    const DynamicPrintConfig config = shell_config();
    slice_parts_with_modifiers(print, model, config, one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { variant.transition_key, before } })) });
    REQUIRE(print.objects().front()->is_step_done(posPrepareInfill));
    CHECK(bases(print) == (std::string(before) == "ramp" ? std::vector<int>{ 3 } : std::vector<int>{}));

    ModelVolume *modifier = model.objects.front()->volumes.back();
    REQUIRE(modifier->is_modifier());
    modifier->config.apply(part_config({ { variant.transition_key, after } }));
    if (change == "taper ratio")
        modifier->config.apply(part_config({ { variant.taper_key, 5. } }));
    print.apply(model, config);
    CHECK(bases(print) == (std::string(after) == "ramp" ? std::vector<int>{ 3 } : std::vector<int>{}));
    CHECK(print.objects().front()->is_step_done(posSlice));
    CHECK(print.objects().front()->is_step_done(posPerimeters));
    CHECK_FALSE(print.objects().front()->is_step_done(posPrepareInfill));
}

// Spiral vase mode prints no top shell and a fixed bottom, so validation rejects a Shell Ramp modifier there.
TEST_CASE("Print validation rejects a modifier on the Ramp shell transition in spiral vase mode", "[Shells][ShellRamp]")
{
    struct Variant { const char *key; const char *transition; bool spiral_mode; bool error; };
    const Variant variant = GENERATE(
        Variant{ "top_shell_layers_transition",    "ramp", true,  true },
        Variant{ "bottom_shell_layers_transition", "ramp", true,  true },
        Variant{ "top_shell_layers_transition",    "hard", true,  false },
        Variant{ "bottom_shell_layers_transition", "hard", true,  false },
        Variant{ "top_shell_layers_transition",    "ramp", false, false },
        Variant{ "bottom_shell_layers_transition", "ramp", false, false });
    CAPTURE(variant.key, variant.transition, variant.spiral_mode);

    // Resetting the extruder at each layer keeps relative extrusion from failing validation for its own reason.
    DynamicPrintConfig config = shell_config();
    config.set_deserialize_strict({ { "spiral_mode", variant.spiral_mode ? 1 : 0 }, { "layer_change_gcode", "G92 E0" } });
    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, config, one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { variant.key, variant.transition } })) });

    std::vector<StringObjectException> warnings;
    const StringObjectException result = print.validate(&warnings);
    if (variant.error) {
        CHECK_THAT(result.string, Catch::Matchers::ContainsSubstring("shell count transition"));
        CHECK_THAT(result.string, Catch::Matchers::ContainsSubstring("can't be used with Spiral vase mode"));
        CHECK(result.opt_key == variant.key);
    } else {
        CHECK(result.string.empty());
    }
}

// Without shells to ramp in spiral vase mode the modifier is not keyed, and toggling the mode re-keys it.
TEST_CASE("Toggling spiral vase mode re-keys the region of a modifier on the Ramp shell transition", "[Shells][ShellRamp]")
{
    DynamicPrintConfig config = shell_config();
    Print print;
    Model model;
    apply_parts_with_modifiers(print, model, config, one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "ramp" } })) });
    CHECK(top_shell_bases(print) == std::vector<int>{ 3 });

    config.set_deserialize_strict({ { "spiral_mode", 1 } });
    print.apply(model, config);
    CHECK(top_shell_bases(print).empty());

    config.set_deserialize_strict({ { "spiral_mode", 0 } });
    print.apply(model, config);
    CHECK(top_shell_bases(print) == std::vector<int>{ 3 });
}

// A taper ratio of 0 keeps the hard step, which a warning says, unless the modifier does not change the shell.
TEST_CASE("A modifier on the Ramp shell transition with a taper ratio of 0 warns that it keeps the hard step", "[Shells][ShellRamp]")
{
    struct Variant { const char *transition_key; const char *taper_key; const char *count_key; const char *direction; };
    const Variant variant = GENERATE(
        Variant{ "top_shell_layers_transition",    "top_shell_ramp_taper_ratio",    "top_shell_layers",    "top" },
        Variant{ "bottom_shell_layers_transition", "bottom_shell_ramp_taper_ratio", "bottom_shell_layers", "bottom" });
    const double taper_ratio  = GENERATE(0., 5.);
    const int    shell_layers = GENERATE(3, 5);
    CAPTURE(variant.transition_key, taper_ratio, shell_layers);

    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, shell_config(), one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { variant.transition_key, "ramp" },
                                                                                { variant.taper_key, taper_ratio },
                                                                                { variant.count_key, shell_layers } })) });
    const std::string warnings = prepare_infill_warnings(print);
    CAPTURE(warnings);
    if (taper_ratio == 0. && shell_layers != 3) {
        CHECK_THAT(warnings, Catch::Matchers::ContainsSubstring(std::string("Hard ") + variant.direction + " shell count transition"));
        CHECK_THAT(warnings, Catch::Matchers::ContainsSubstring("taper ratio above 0"));
    } else {
        CHECK(warnings.empty());
    }
}

// Hard is the transition of a modifier that sets neither key, the Ramp shell transitions only act set on a modifier,
// and the key of a Shell Ramp modifier changing no count leaves its shells those of a modifier on Hard, so none of these
// changes a single command of the G-code.
TEST_CASE("Without a modifier changing a shell on the Ramp transition the G-code is that of a slice without the Ramp settings", "[Shells][ShellRamp]")
{
    const char       *wall_generator = GENERATE("classic", "arachne");
    // Both shell projections: discover_vertical_shells() at ensure_all, discover_horizontal_shells() otherwise.
    const char       *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    const std::string case_name      = GENERATE("modifiers on Hard", "Ramp set in the process and on the part", "Ramp modifier changing no count");
    CAPTURE(wall_generator, ensure_vertical_shell_thickness, case_name);

    DynamicPrintConfig plain_config = shell_config(wall_generator);
    plain_config.set_deserialize_strict({ { "ensure_vertical_shell_thickness", ensure_vertical_shell_thickness } });
    DynamicPrintConfig       keyed_config = plain_config;
    std::vector<ModifierBox> plain_modifiers, keyed_modifiers;
    if (case_name == "modifiers on Hard") {
        // One modifier deepening the shells and one thinning them.
        plain_modifiers = { modifier_box(15., 25., shell_modifier_config({})),
                            modifier_box(-5., 5., shell_modifier_config({ { "top_shell_layers", 1 }, { "bottom_shell_layers", 1 } })) };
        keyed_modifiers = { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "hard" }, { "top_shell_ramp_taper_ratio", 5. },
                                                                           { "bottom_shell_layers_transition", "hard" }, { "bottom_shell_ramp_taper_ratio", 5. } })),
                            modifier_box(-5., 5., shell_modifier_config({ { "top_shell_layers", 1 }, { "bottom_shell_layers", 1 },
                                                                          { "top_shell_layers_transition", "hard" }, { "top_shell_ramp_taper_ratio", 5. },
                                                                          { "bottom_shell_layers_transition", "hard" }, { "bottom_shell_ramp_taper_ratio", 5. } })) };
    } else if (case_name == "Ramp set in the process and on the part") {
        keyed_config.set_deserialize_strict({ { "top_shell_layers_transition", "ramp" }, { "top_shell_ramp_taper_ratio", 5. },
                                              { "bottom_shell_layers_transition", "ramp" }, { "bottom_shell_ramp_taper_ratio", 5. } });
    } else {
        // A modifier region of its own on both sides, which by itself splits the part's fills, so that only the key differs.
        plain_modifiers = { modifier_box(15., 25., part_config({ { "top_shell_ramp_taper_ratio", 5. }, { "bottom_shell_ramp_taper_ratio", 5. } })) };
        keyed_modifiers = { modifier_box(15., 25., part_config({ { "top_shell_ramp_taper_ratio", 5. }, { "bottom_shell_ramp_taper_ratio", 5. },
                                                                 { "top_shell_layers_transition", "ramp" }, { "bottom_shell_layers_transition", "ramp" } })) };
    }

    Print plain_print, keyed_print;
    Model plain_model, keyed_model;
    slice_parts_with_modifiers(plain_print, plain_model, plain_config, one_part, plain_modifiers);
    slice_parts_with_modifiers(keyed_print, keyed_model, keyed_config, one_part, keyed_modifiers, [&case_name](ModelObject &object) {
        if (case_name == "Ramp set in the process and on the part")
            object.volumes.front()->config.apply(part_config({ { "top_shell_layers_transition", "ramp" }, { "bottom_shell_layers_transition", "ramp" } }));
    });
    const std::vector<std::string> plain = gcode_commands(gcode(plain_print));
    const std::vector<std::string> keyed = gcode_commands(gcode(keyed_print));
    REQUIRE(plain.size() > 1000);
    const std::string difference = first_difference(plain, keyed);
    CAPTURE(difference);
    CHECK(difference.empty());
}

// Both transitions on Ramp: the wall part of the key is what a modifier on the Ramp wall transition alone gets, and
// the shell parts are set beside it.
TEST_CASE("A modifier on the Ramp wall and shell transitions is keyed for both", "[Shells][ShellRamp]")
{
    DynamicPrintConfig config = shell_config("arachne");
    config.set_deserialize_strict({ { "wall_ramp_taper_ratio", 10. } });
    auto modifier_key = [&config](const DynamicPrintConfig &modifier_config) {
        Print print;
        Model model;
        apply_parts_with_modifiers(print, model, config, one_part, { modifier_box(15., 25., modifier_config) });
        const PrintRegion *modifier = region_with_top_shell_layers(print, 5);
        REQUIRE(modifier != nullptr);
        return modifier->ramp_key();
    };
    const RampKey walls = modifier_key(shell_modifier_config({ { "wall_loops", 4 }, { "wall_loops_transition", "ramp" } }));
    const RampKey both  = modifier_key(shell_modifier_config({ { "wall_loops", 4 }, { "wall_loops_transition", "ramp" },
                                                               { "top_shell_layers_transition", "ramp" }, { "bottom_shell_layers_transition", "ramp" } }));
    REQUIRE(walls.wall_base == 2);
    CHECK(walls.top_base_layers == -1);
    CHECK(both.wall_base == walls.wall_base);
    CHECK(both.wall_root == walls.wall_root);
    CHECK(both.top_base_layers == 3);
    CHECK_THAT(both.top_base_thickness, Catch::Matchers::WithinAbs(0.4, 1e-9));
    CHECK(both.bottom_base_layers == 3);
    CHECK_THAT(both.bottom_base_thickness, Catch::Matchers::WithinAbs(0.2, 1e-9));
}

// The part's top shell is 3 layers with a 0.4 mm minimum and its bottom shell 3 layers with a 0.2 mm minimum. A modifier
// on the Ramp with a taper ratio of 0 that sets those same shells changes nothing and does not warn.
TEST_CASE("A modifier on the Ramp shell transition with a taper ratio of 0 keeping the shell does not warn", "[Shells][ShellRamp]")
{
    struct Variant { const char *transition_key; const char *taper_key; const char *count_key; const char *thickness_key; double thickness; };
    const Variant variant = GENERATE(
        Variant{ "top_shell_layers_transition",    "top_shell_ramp_taper_ratio",    "top_shell_layers",    "top_shell_thickness",    0.4 },
        Variant{ "bottom_shell_layers_transition", "bottom_shell_ramp_taper_ratio", "bottom_shell_layers", "bottom_shell_thickness", 0.2 });
    CAPTURE(variant.transition_key);

    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, shell_config(), one_part,
                               { modifier_box(15., 25., part_config({ { variant.transition_key, "ramp" }, { variant.taper_key, 0. },
                                                                      { variant.count_key, 3 }, { variant.thickness_key, variant.thickness } })) });
    const std::string warnings = prepare_infill_warnings(print);
    CAPTURE(warnings);
    CHECK(warnings.empty());
}

// The part's shells are 0.6 mm deep at 0.2 mm layers, 3 layers with a minimum of 0.4 mm on top and 0.2 mm at the bottom.
// A modifier setting 2 layers with a 0.6 mm minimum is as deep at the nominal layer height, yet is 3 layers under
// thinner ones: it has no ramp, builds its own shell and says it keeps the hard step, whatever its taper ratio.
TEST_CASE("A modifier on the Ramp shell transition as deep as the shell it modifies but set differently falls back to Hard and warns", "[Shells][ShellRamp]")
{
    struct Variant { const char *transition_key; const char *taper_key; const char *count_key; const char *thickness_key; const char *direction; };
    const Variant variant = GENERATE(
        Variant{ "top_shell_layers_transition",    "top_shell_ramp_taper_ratio",    "top_shell_layers",    "top_shell_thickness",    "top" },
        Variant{ "bottom_shell_layers_transition", "bottom_shell_ramp_taper_ratio", "bottom_shell_layers", "bottom_shell_thickness", "bottom" });
    const double taper_ratio = GENERATE(0., 5.);
    CAPTURE(variant.transition_key, taper_ratio);

    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, shell_config(), one_part,
                               { modifier_box(15., 25., part_config({ { variant.transition_key, "ramp" }, { variant.taper_key, taper_ratio },
                                                                      { variant.count_key, 2 }, { variant.thickness_key, 0.6 } })) });
    const PrintObject &object   = *print.objects().front();
    const PrintRegion *modifier = nullptr;
    for (size_t i = 0; i < object.num_printing_regions(); ++ i)
        if (const RampKey &key = object.printing_region(i).ramp_key(); key.top_base_layers >= 0 || key.bottom_base_layers >= 0)
            modifier = &object.printing_region(i);
    REQUIRE(modifier != nullptr);
    CHECK_FALSE(modifier->has_top_shell_ramp(0.2));
    CHECK_FALSE(modifier->has_bottom_shell_ramp(0.2));
    // The region keeps its own counts, not those of the part.
    const ShellCounts counts = modifier->shell_default_counts(0.2);
    CHECK(counts.top_layers == (std::string(variant.direction) == "top" ? 2 : 3));
    CHECK(counts.bottom_layers == (std::string(variant.direction) == "top" ? 3 : 2));

    const std::string warnings = prepare_infill_warnings(print);
    CAPTURE(warnings);
    CHECK_THAT(warnings, Catch::Matchers::ContainsSubstring(std::string("Hard ") + variant.direction + " shell count transition"));
}

// The part keyed by its top shell count in two height ranges, 3 layers below z = 1 and 4 above, gives the modifier no
// single base depth: it gets no zone, and a warning.
TEST_CASE("A modifier on the Ramp shell transition over different shells in two height ranges warns and gets no zone", "[Shells][ShellRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, shell_config(), one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "ramp" } })) },
                               [](ModelObject &object) {
                                   DynamicPrintConfig range;
                                   // Every layer range must carry a layer_height (see layer_height_profile_from_ranges).
                                   range.set_deserialize_strict({ { "top_shell_layers", 4 }, { "layer_height", 0.2 } });
                                   object.layer_config_ranges[{ 1., 2. }].assign_config(std::move(range));
                                   object.name                  = "Strap";
                                   object.volumes.front()->name = "Base";
                                   object.volumes.back()->name  = "Reinforcement";
                               });
    REQUIRE(top_shell_bases(print) == std::vector<int>{ 3, 4 });
    const std::string warnings = prepare_infill_warnings(print);
    CAPTURE(warnings);
    CHECK_THAT(warnings, Catch::Matchers::ContainsSubstring("same shell settings in every height range"));

    // The warning names the modifier, and the part of the object it modifies.
    CHECK_THAT(warnings, Catch::Matchers::ContainsSubstring("Reinforcement"));
    CHECK_THAT(warnings, Catch::Matchers::ContainsSubstring("Base"));
    CHECK_THAT(warnings, Catch::Matchers::ContainsSubstring("Strap"));

    std::vector<ModifierOnPart> inconsistent;
    CHECK(build_shell_depth_field(*print.objects().front(), &inconsistent) == nullptr);
    REQUIRE(inconsistent.size() == 1);
    CHECK(inconsistent.front().first == print.model().objects.front()->volumes.back());
    CHECK(inconsistent.front().second == print.model().objects.front()->volumes.front());
}

// The part is 20 x 20 x 2 mm, 10 layers of 0.2 mm, its top shell 3 layers with a 0.4 mm minimum (0.6 mm); the modifier
// over x = 15..25 sets 5 layers (1 mm) with a taper ratio of 10, so the ramp runs 4 mm across the top face from x = 15,
// over which the shell steps from 5 layers down to 3.
TEST_CASE("The shell depth field of a print follows a modifier on the Ramp top shell transition", "[Shells][ShellRamp]")
{
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, shell_config(), one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { "top_shell_layers_transition", "ramp" } })) });
    const PrintObject &object = *print.objects().front();
    REQUIRE(object.layers().size() == 10);
    const auto field = build_shell_depth_field(object);
    REQUIRE(field != nullptr);

    // The part's corner in the frame the layers were sliced in.
    const Layer      &top     = *object.layers().back();
    const BoundingBox extents = get_extents(top.lslices);
    const double      x0 = unscaled(extents.min.x()), y0 = unscaled(extents.min.y());
    const std::vector<double> offsets { 0., 0.2, 0.4, 0.6, 0.8, 1., 1.2 };
    auto layers = [&](double x) { return field->solid_layers(ShellSide::Top, Vec3d(x0 + x, y0 + 10., top.slice_z), offsets); };
    CHECK(layers(17.5) == std::optional<size_t>(5));
    // 1 and 3 mm into the ramp: n = 4.5 and 3.5.
    CHECK(layers(14.) == std::optional<size_t>(5));
    CHECK(layers(12.) == std::optional<size_t>(4));
    CHECK_FALSE(layers(5.).has_value());
    CHECK_FALSE(field->solid_layers(ShellSide::Bottom, Vec3d(x0 + 17.5, y0 + 10., object.layers().front()->slice_z), offsets).has_value());

    // Under the top layer the shell reaches 5 layers down inside the zone.
    const std::vector<LayerShellLevels> levels = make_shell_levels(*field, object);
    REQUIRE(levels.size() == 10);
    CHECK(levels.back().top.level.size() == 5);
    CHECK(levels.back().bottom.empty());
}

// Layers of 0.28 mm over a first layer of 0.2 mm, so the bottom layers lie 0, 0.2, 0.48, 0.76, 1.04 and 1.32 mm above the
// bed, and the part's bottom shell is 3 layers without a minimum thickness. A modifier over x = 15..25 on the Ramp sets
// 1 layer or 5, and its ramp is 3 * 0.28 - 0.28 = 0.56 mm * 10 = 5.6 mm long either way. On the bed layer, the shell is
// the modifier's inside the zone and steps towards the part's 3 layers across the ramp. Counted as stock counts them,
// by layer: a depth in mm compared with the real layer heights would put one more layer in, 2 or 6 inside the zone,
// and 4 at the end of the thinning ramp.
TEST_CASE("The bottom shell levels of a print with a thinner first layer count layers as stock does", "[Shells][ShellRamp]")
{
    const int layers = GENERATE(1, 5);
    CAPTURE(layers);

    DynamicPrintConfig config = shell_config();
    config.set_deserialize_strict({ { "layer_height", 0.28 }, { "initial_layer_print_height", 0.2 }, { "bottom_shell_thickness", 0. } });
    Print print;
    Model model;
    slice_parts_with_modifiers(print, model, config, one_part,
                               { modifier_box(15., 25., shell_modifier_config({ { "bottom_shell_layers", layers }, { "bottom_shell_layers_transition", "ramp" } })) });
    const PrintObject &object = *print.objects().front();
    REQUIRE(object.layers().size() >= 6);
    REQUIRE_THAT(object.layers()[0]->height, Catch::Matchers::WithinAbs(0.2, 1e-6));
    REQUIRE_THAT(object.layers()[1]->height, Catch::Matchers::WithinAbs(0.28, 1e-6));
    const auto field = build_shell_depth_field(object);
    REQUIRE(field != nullptr);

    const std::vector<LayerShellLevels> levels = make_shell_levels(*field, object);
    const ShellLevels                  &bottom = levels.front().bottom;
    REQUIRE_FALSE(bottom.empty());
    // As many levels as the thicker of the two shells.
    CHECK(bottom.level.size() == size_t(std::max(layers, 3)));

    // The levels at a point of the bed layer, in the frame the layers were sliced in.
    const BoundingBox extents = get_extents(object.layers().front()->lslices);
    const double      x0 = unscaled(extents.min.x()), y0 = unscaled(extents.min.y());
    auto levels_at = [&bottom, y0](double x) {
        const Point p = Point::new_scale(x, y0 + 10.);
        size_t      k = 0;
        while (k < bottom.level.size() && contains(bottom.level[k], p))
            ++ k;
        return k;
    };
    CHECK(levels_at(x0 + 17.5) == size_t(layers));
    // Just inside where the ramp ends, about 5 mm from the zone, n = 2.8 or 3.2: thinning, the part's 3 layers; thickening,
    // the last step of 4.
    const BoundingBox reach = get_extents(bottom.reach);
    REQUIRE_THAT(unscaled(reach.min.x()) - x0, Catch::Matchers::WithinAbs(15. - 5.6, 0.55));
    CHECK(levels_at(unscaled(reach.min.x()) + 0.6) == size_t(layers < 3 ? 3 : 4));
}

// Two modifiers with the same config, which sets the top shell minimum, over parts with a top shell minimum of 0.4 and
// 0.8 mm: on the Ramp they are keyed apart into two regions, each tapering to its own part's shell, on Hard they share one.
TEST_CASE("Shell Ramp modifiers over parts with different shell thickness get a region per part", "[Shells][ShellRamp]")
{
    const std::vector<DynamicPrintConfig> parts { part_config({ { "top_shell_thickness", 0.4 } }), part_config({ { "top_shell_thickness", 0.8 } }) };
    auto modifiers = [](const char *transition) {
        const DynamicPrintConfig config = shell_modifier_config({ { "top_shell_thickness", 0.4 }, { "top_shell_layers_transition", transition },
                                                                         { "bottom_shell_layers_transition", transition } });
        return std::vector<ModifierBox>{ modifier_box(5., 15., config), modifier_box(25., 35., config) };
    };
    Print hard_print, ramp_print;
    Model hard_model, ramp_model;
    apply_parts_with_modifiers(hard_print, hard_model, shell_config(), parts, modifiers("hard"));
    apply_parts_with_modifiers(ramp_print, ramp_model, shell_config(), parts, modifiers("ramp"));

    // The regions of the modifiers, by their top shell count of 5, and the top shell minimum each is keyed to.
    auto modifier_regions = [](const Print &print) {
        const PrintObject  &object = *print.objects().front();
        std::vector<double> top_base_thicknesses;
        for (size_t i = 0; i < object.num_printing_regions(); ++ i)
            if (object.printing_region(i).config().top_shell_layers.value == 5)
                top_base_thicknesses.emplace_back(object.printing_region(i).ramp_key().top_base_thickness);
        std::sort(top_base_thicknesses.begin(), top_base_thicknesses.end());
        return top_base_thicknesses;
    };
    CHECK(modifier_regions(hard_print).size() == 1);
    const std::vector<double> ramp_regions = modifier_regions(ramp_print);
    REQUIRE(ramp_regions.size() == 2);
    CHECK_THAT(ramp_regions[0], Catch::Matchers::WithinAbs(0.4, 1e-9));
    CHECK_THAT(ramp_regions[1], Catch::Matchers::WithinAbs(0.8, 1e-9));
}

// The tests below run on both shell projections: discover_vertical_shells() at ensure_all, discover_horizontal_shells()
// otherwise. See projection_config() and top_ramp_modifier() for the fixture.

// Stock builds the modifier's 7 layers under it too, but not the ramp: just outside the modifier, where n = 6.75, the
// shell is still 7 layers deep. Well beyond the 4 mm ramp it is the strap's 3.
TEST_CASE("A modifier on the Ramp top shell transition deepens the shell under it and ramps it back outside", "[Shells][ShellRamp]")
{
    const char *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    const char *wall_generator                  = GENERATE("classic", "arachne");
    CAPTURE(ensure_vertical_shell_thickness, wall_generator);

    Print print;
    Model model;
    const PrintObject &object = slice_meshes_with_modifiers(print, model, projection_config(ensure_vertical_shell_thickness, wall_generator),
                                                            { make_cube(60., 20., 4.) }, { top_ramp_modifier() });
    REQUIRE(object.layers().size() == 20);
    const ShellProbe probe(object);
    CHECK(probe.top_depth(30.) == 7);
    // Next to the modifier the layer under the shell can be solid too, where its infill meets the modifier region's.
    CHECK(probe.top_depth(35.25) >= 7);
    CHECK(probe.top_depth(24.75) >= 7);
    CHECK(probe.top_depth(45.) == 3);
    CHECK(probe.top_depth(15.) == 3);
}

// The layer k layers below the top is in the shell out to 7 - k mm from the modifier, on both sides of it, to within the
// 0.5 mm grid the field is sampled on. The deepest of them, 6 layers down, lies over sparse infill: bridge_over_infill()
// turns it into an internal bridge and widens it to anchor it, so it is only checked to reach its level.
// The same holds with the modifier's 1.4 mm deep shell set as a minimum thickness over 3 layers: with the strap's
// minimum at 0.6 mm, its 3 layers' depth, the two rules then agree at every point of the ramp.
TEST_CASE("Across the ramp of a Ramp top shell modifier the shell steps down one layer per taper length", "[Shells][ShellRamp]")
{
    const char       *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    const std::string modifier_shell                  = GENERATE("7 layers", "1.4 mm minimum");
    CAPTURE(ensure_vertical_shell_thickness, modifier_shell);

    DynamicPrintConfig config = projection_config(ensure_vertical_shell_thickness);
    config.set_deserialize_strict({ { "top_shell_thickness", 0.6 } });
    const ModifierBox modifier = modifier_shell == "7 layers" ? top_ramp_modifier() :
                                                                top_ramp_modifier({ { "top_shell_layers", 3 }, { "top_shell_thickness", 1.4 } });
    Print print;
    Model model;
    const PrintObject &object = slice_meshes_with_modifiers(print, model, config, { make_cube(60., 20., 4.) }, { modifier });
    REQUIRE(object.layers().size() == 20);
    const ShellProbe probe(object);
    CHECK(probe.top_depth(30.) == 7);
    CHECK(probe.top_depth(45.) == 3);

    const double grid = 0.5;
    for (size_t k = 3; k <= 6; ++ k) {
        const size_t layer    = object.layers().size() - 1 - k;
        const double expected = 7. - double(k);
        CAPTURE(k);
        const double right = probe.solid_run(layer, 35., 1.);
        const double left  = probe.solid_run(layer, 25., -1.);
        CHECK(right > expected - grid);
        CHECK(left > expected - grid);
        if (k < 6) {
            CHECK(right < expected + grid);
            CHECK(left < expected + grid);
        }
    }
    // Below the shell, nothing outside the modifier.
    CHECK(probe.solid_run(object.layers().size() - 8, 35., 1.) < grid);
}

// A modifier with no top shell on the Ramp: no solid layer under its top, and outside it the shell ramps from none to the
// strap's 3 layers over (0.6 - 0) * 5 = 3 mm, n = d / 1 mm. The top layer itself stays the strap's own top surface.
TEST_CASE("A modifier on the Ramp top shell transition thinning the shell to none ramps it back outside", "[Shells][ShellRamp]")
{
    const char *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    CAPTURE(ensure_vertical_shell_thickness);

    Print print;
    Model model;
    const PrintObject &object = slice_meshes_with_modifiers(print, model, projection_config(ensure_vertical_shell_thickness), { make_cube(60., 20., 4.) },
                                                            { top_ramp_modifier({ { "top_shell_layers", 0 } }) });
    REQUIRE(object.layers().size() == 20);
    const ShellProbe probe(object);
    const size_t top = object.layers().size() - 1;
    CHECK_FALSE(probe.solid(top - 1, probe.at(30., 10.)));
    CHECK_FALSE(probe.solid(top - 2, probe.at(30., 10.)));
    CHECK(probe.top_depth(45.) == 3);
    CHECK(probe.top_depth(15.) == 3);
    // Half a millimetre out, n = 0.5: the top layer only. Stock has the strap's 3 layers right up to the modifier.
    CHECK(probe.top_depth(35.5) < 3);
    CHECK(probe.top_depth(24.5) < 3);
    // Layer 1 below the top is in the shell from 1 mm out, so from x = 39 it runs 3 mm towards the modifier, to within the
    // 0.5 mm grid the field is sampled on. Layer 2, from 2 mm out, lies over sparse infill, and bridge_over_infill() widens
    // it to anchor it, so it is not checked here.
    CHECK_THAT(probe.solid_run(top - 1, 39., -1.), Catch::Matchers::WithinAbs(3., 0.5));
    CHECK_THAT(probe.solid_run(top - 1, 21., 1.), Catch::Matchers::WithinAbs(3., 0.5));
}

// The bottom shell mirrors the top: a 7 layer bottom shell on the Ramp deepens the 3 layers of the strap's bottom under
// the modifier and ramps back outside it, over the bed and over an overhang alike. For the overhang, the strap stands
// 2 mm up on a 10 mm wide pillar at x = 0..10, so its bottom is bridged from x = 10 on.
TEST_CASE("A modifier on the Ramp bottom shell transition deepens the bottom shell under it and ramps it back outside", "[Shells][ShellRamp]")
{
    const char       *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    const std::string bottom                          = GENERATE("on the bed", "overhanging");
    CAPTURE(ensure_vertical_shell_thickness, bottom);

    std::vector<TriangleMesh> parts;
    if (bottom == "on the bed")
        parts.emplace_back(make_cube(60., 20., 4.));
    else {
        parts.emplace_back(make_cube(10., 20., 2.));
        TriangleMesh strap = make_cube(60., 20., 4.);
        strap.translate(0.f, 0.f, 2.f);
        parts.emplace_back(std::move(strap));
    }
    const ModifierBox modifier { 25., 35., -1., 7., part_config({ { "bottom_shell_layers", 7 }, { "bottom_shell_layers_transition", "ramp" },
                                                                   { "bottom_shell_ramp_taper_ratio", 5. } }) };
    Print print;
    Model model;
    const PrintObject &object = slice_meshes_with_modifiers(print, model, projection_config(ensure_vertical_shell_thickness), std::move(parts), { modifier });
    REQUIRE(object.layers().size() == (bottom == "on the bed" ? 20 : 30));
    const ShellProbe probe(object);
    CHECK(probe.bottom_depth(30.) == 7);
    CHECK(probe.bottom_depth(35.25) >= 7);
    CHECK(probe.bottom_depth(24.75) >= 7);
    CHECK(probe.bottom_depth(45.) == 3);
    CHECK(probe.bottom_depth(15.) == 3);
}

// The modifier covers only the top 2 layers of the strap, so its region does not reach down the 7 layers of its shell.
// The shell is defined on the surface, so the zone still gives 7 layers under its top: on the ensure_all projection
// because it projects the surfaces of every region, on the other one because the other regions' surfaces within the
// reach are projected too.
TEST_CASE("A Ramp top shell modifier shorter than its shell still gives its depth under its top", "[Shells][ShellRamp]")
{
    const char *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    CAPTURE(ensure_vertical_shell_thickness);

    Print print;
    Model model;
    const PrintObject &object = slice_meshes_with_modifiers(print, model, projection_config(ensure_vertical_shell_thickness), { make_cube(60., 20., 4.) },
                                                            { top_ramp_modifier({ { "top_shell_layers", 7 } }, 3.55) });
    REQUIRE(object.layers().size() == 20);
    const ShellProbe probe(object);
    CHECK(probe.top_depth(30.) == 7);
    CHECK(probe.top_depth(35.25) >= 7);
    CHECK(probe.top_depth(45.) == 3);
}

// A wedge, its top sloping up from 1 mm at y = 0 to 4 mm at y = 20, so the sloped face drops a 0.2 mm layer every 1.33 mm.
// The modifier's zone covers its slope over x = 25..35. Under the slope the shell is the modifier's 7 layers in the zone,
// tapers across the ramp, and is what the wedge gets without the modifier well beyond it. Counted from the topmost layer
// filled at a point, which on a slope may lie under the walls of the layer above, so one layer short of the shell.
TEST_CASE("A Ramp top shell modifier on a slope deepens the shell under the slope", "[Shells][ShellRamp]")
{
    const char *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    CAPTURE(ensure_vertical_shell_thickness);

    auto wedge = []() {
        indexed_triangle_set its = its_make_cube(60., 20., 4.);
        for (stl_vertex &v : its.vertices)
            if (v.y() == 0.f && v.z() == 4.f)
                v.z() = 1.f;
        return TriangleMesh(std::move(its));
    };
    const DynamicPrintConfig config = projection_config(ensure_vertical_shell_thickness);
    Print plain_print, print;
    Model plain_model, model;
    const PrintObject &plain  = slice_meshes_with_modifiers(plain_print, plain_model, config, { wedge() }, {});
    const PrintObject &object = slice_meshes_with_modifiers(print, model, config, { wedge() }, { top_ramp_modifier() });
    const ShellProbe   plain_probe(plain);
    const ShellProbe   probe(object);
    for (const double y : { 6., 10., 14. }) {
        CAPTURE(y);
        CHECK(plain_probe.top_depth(30., y) <= 4);
        CHECK(probe.top_depth(30., y) >= 6);
        CHECK(probe.top_depth(10., y) == plain_probe.top_depth(10., y));
        CHECK(probe.top_depth(50., y) == plain_probe.top_depth(50., y));
        // 3 mm into the ramp, n = 4.
        CHECK(probe.top_depth(38., y) > plain_probe.top_depth(38., y));
        CHECK(probe.top_depth(38., y) < probe.top_depth(30., y));
    }
}

// A region with no top shell layers has no top shell to anchor, so on an object that also has a Ramp zone it still gets
// nothing under its top surfaces outside the reach. The step is at x = 10, 15 mm from the modifier and well beyond its
// ramp; the top of the lower part at layer 9 would otherwise get a ring of solid infill at the foot of the step. Only
// the ensure_all projection adds such an anchor.
TEST_CASE("A region without top shell layers gets no anchor outside the reach of a Ramp zone", "[Shells][ShellRamp]")
{
    DynamicPrintConfig config = projection_config("ensure_all");
    config.set_deserialize_strict({ { "top_shell_layers", 0 }, { "top_shell_thickness", 0. } });
    auto parts = []() {
        TriangleMesh step = make_cube(10., 20., 2.);
        step.translate(0.f, 0.f, 2.f);
        return std::vector<TriangleMesh>{ make_cube(60., 20., 2.), std::move(step) };
    };
    Print plain_print, print;
    Model plain_model, model;
    const PrintObject &plain  = slice_meshes_with_modifiers(plain_print, plain_model, config, parts(), {});
    const PrintObject &object = slice_meshes_with_modifiers(print, model, config, parts(), { top_ramp_modifier() });
    REQUIRE(object.layers().size() == 20);
    const ShellProbe plain_probe(plain);
    const ShellProbe probe(object);
    const Point      foot = probe.at(9.8, 10.);
    CHECK_FALSE(plain_probe.solid(9, plain_probe.at(9.8, 10.)));
    CHECK_FALSE(probe.solid(9, foot));
}

// The bottom side of the same: a 4 mm strap stands 2 mm up on a pillar at x = 0..10 and has no bottom shell layers, and a
// modifier deepening the bottom shell to 7 layers on the Ramp over x = 25..35 is 15 mm away. The bottom of the strap
// at layer 10 would otherwise get a ring of solid infill over the edge of the pillar.
TEST_CASE("A region without bottom shell layers gets no anchor outside the reach of a Ramp zone", "[Shells][ShellRamp]")
{
    DynamicPrintConfig config = projection_config("ensure_all");
    config.set_deserialize_strict({ { "bottom_shell_layers", 0 }, { "bottom_shell_thickness", 0. } });
    auto parts = []() {
        TriangleMesh strap = make_cube(60., 20., 4.);
        strap.translate(0.f, 0.f, 2.f);
        return std::vector<TriangleMesh>{ make_cube(10., 20., 2.), std::move(strap) };
    };
    const ModifierBox modifier { 25., 35., -1., 7., part_config({ { "bottom_shell_layers", 7 }, { "bottom_shell_layers_transition", "ramp" },
                                                                   { "bottom_shell_ramp_taper_ratio", 5. } }) };
    Print plain_print, print;
    Model plain_model, model;
    const PrintObject &plain  = slice_meshes_with_modifiers(plain_print, plain_model, config, parts(), {});
    const PrintObject &object = slice_meshes_with_modifiers(print, model, config, parts(), { modifier });
    REQUIRE(object.layers().size() == 30);
    const ShellProbe plain_probe(plain);
    const ShellProbe probe(object);
    REQUIRE(probe.bottom_depth(30.) == 7);
    CHECK_FALSE(plain_probe.solid(10, plain_probe.at(9.8, 10.)));
    CHECK_FALSE(probe.solid(10, probe.at(9.8, 10.)));
}

// One modifier over x = 25..35 reinforcing walls and shells: 4 walls over the strap's 2, and the 7 layer top shell on the
// Ramp of top_ramp_modifier(). The walls taper over 2 * 5 bead widths, about 4 mm, on the Ramp with Arachne, and step
// at the modifier on Hard or with Classic, which keeps the hard step. The shell ramps the same either way: the layer k
// layers below the top is in the shell out to 7 - k mm from the modifier, to within the 0.5 mm grid of the field.
TEST_CASE("A modifier reinforcing walls and shells ramps each by its own transition", "[Shells][ShellRamp]")
{
    struct Variant { const char *wall_generator; const char *wall_transition; };
    const Variant variant = GENERATE(Variant{ "arachne", "ramp" }, Variant{ "arachne", "hard" }, Variant{ "classic", "ramp" }, Variant{ "classic", "hard" });
    CAPTURE(variant.wall_generator, variant.wall_transition);
    const bool walls_ramp = std::string(variant.wall_generator) == "arachne" && std::string(variant.wall_transition) == "ramp";

    ModifierBox modifier = top_ramp_modifier();
    modifier.config.set_deserialize_strict({ { "wall_loops", 4 }, { "wall_loops_transition", variant.wall_transition }, { "wall_ramp_taper_ratio", 5. } });
    Print print;
    Model model;
    const PrintObject &object = slice_meshes_with_modifiers(print, model, projection_config("ensure_all", variant.wall_generator),
                                                            { make_cube(60., 20., 4.) }, { modifier });
    REQUIRE(object.layers().size() == 20);

    // The walls, half way up.
    const Layer &layer = *object.get_layer(10);
    CHECK(walls_across_side(layer, 30.) == 4);
    CHECK(walls_across_side(layer, 45.) == 2);
    // From beyond the walls a hard step lays along the modifier's outline at x = 35.
    bool three_walls = false;
    for (double x = 36.5; x < 40.; x += 0.1)
        three_walls = three_walls || walls_across_side(layer, x) == 3;
    CHECK(three_walls == walls_ramp);

    // The shell.
    const ShellProbe probe(object);
    CHECK(probe.top_depth(30.) == 7);
    CHECK(probe.top_depth(45.) == 3);
    CHECK(probe.top_depth(15.) == 3);
    const double grid = 0.5;
    for (size_t k = 3; k <= 5; ++ k) {
        CAPTURE(k);
        const size_t layer_id = object.layers().size() - 1 - k;
        CHECK_THAT(probe.solid_edge(layer_id, 35., 45.), Catch::Matchers::WithinAbs(7. - double(k), grid));
        CHECK_THAT(probe.solid_edge(layer_id, 25., 15.), Catch::Matchers::WithinAbs(7. - double(k), grid));
    }
}

// The staircase of a thickening top ramp lies over sparse infill at each step. bridge_over_infill() turns a step wider
// than the support it finds below into an internal bridge, and widens that bridge to anchor it, at most to the step
// above: going down, the shell only steps inward. It leaves no hole in any layer, also where the extra bridge layer
// turns the solid over a bridge into a second one, and no bridge narrower than a line. The ramp of top_ramp_modifier()
// at a taper ratio of rho runs (1.4 - 0.6) * rho mm, so the layer k layers below the top, below the 0.4 mm minimum the
// strap and the modifier share, is in the shell out to (7 - k) * 0.2 * rho mm from the modifier: 1 mm per layer at 5,
// 2 mm at 10, wider than the four solid spacings by which a step finds support on the one below.
TEST_CASE("The staircase of a Ramp top shell steps inward going down without holes or bridges narrower than a line", "[Shells][ShellRamp]")
{
    const char  *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    const double taper_ratio                     = GENERATE(5., 10.);
    const char  *extra_bridge_layer              = GENERATE("disabled", "apply_to_all");
    CAPTURE(ensure_vertical_shell_thickness, taper_ratio, extra_bridge_layer);

    DynamicPrintConfig config = projection_config(ensure_vertical_shell_thickness);
    config.set_deserialize_strict({ { "enable_extra_bridge_layer", extra_bridge_layer } });
    ModifierBox modifier = top_ramp_modifier();
    modifier.config.set_deserialize_strict({ { "top_shell_ramp_taper_ratio", taper_ratio } });
    Print print;
    Model model;
    const PrintObject &object = slice_meshes_with_modifiers(print, model, config, { make_cube(60., 20., 4.) }, { modifier });
    REQUIRE(object.layers().size() == 20);
    const ShellProbe probe(object);
    const size_t     top = object.layers().size() - 1;
    // The second bridge layer replaces solid infill, it does not add to the shell.
    CHECK(probe.top_depth(30.) == 7);
    CHECK(probe.top_depth(48.) == 3);
    CHECK(probe.top_depth(12.) == 3);

    const double grid = 0.5;
    for (const double near : { 35., 25. }) {
        const double far  = near == 35. ? 48. : 12.;
        double       above = std::abs(far - near);
        for (size_t k = 1; k <= 6; ++ k) {
            CAPTURE(near, k);
            const double edge  = probe.solid_edge(top - k, near, far);
            const double level = k < 2 ? std::abs(far - near) : (7. - double(k)) * 0.2 * taper_ratio;
            CHECK(edge > level - grid);
            // The anchor of a bridge may overlap the step above by a fraction of a line.
            CHECK(edge < above + 0.25);
            above = edge;
        }
    }
    for (size_t k = 0; k <= 8; ++ k) {
        CAPTURE(k);
        std::string holes;
        for (double x = 12.; x <= 48.; x += 0.05)
            if (! probe.inside(top - k, probe.at(x, 10.)))
                holes += std::to_string(x) + "; ";
        CHECK(holes.empty());
        for (const LayerRegion *layerm : object.get_layer(int(top - k))->regions())
            for (const Surface &surface : layerm->fill_surfaces.surfaces)
                if (surface.surface_type == stInternalBridge)
                    CHECK_FALSE(shrink_ex({ surface.expolygon }, float(0.5 * layerm->flow(frSolidInfill).scaled_spacing())).empty());
    }
}

// Infill combination prints the sparse infill of two layers as one where both have it, so a combined layer never takes
// sparse infill from a layer the staircase of a Ramp top shell made solid, and the shell is the same as without it.
TEST_CASE("Infill combination keeps the staircase of a Ramp top shell", "[Shells][ShellRamp]")
{
    const char *ensure_vertical_shell_thickness = GENERATE("ensure_all", "none");
    CAPTURE(ensure_vertical_shell_thickness);

    auto slice = [ensure_vertical_shell_thickness](Print &print, Model &model, bool combine) -> const PrintObject & {
        DynamicPrintConfig config = projection_config(ensure_vertical_shell_thickness);
        // Two 0.2 mm layers combined into one of 0.4 mm.
        config.set_deserialize_strict({ { "infill_combination", combine }, { "infill_combination_max_layer_height", "0.4" } });
        return slice_meshes_with_modifiers(print, model, config, { make_cube(60., 20., 4.) }, { top_ramp_modifier() });
    };
    Print plain_print, print;
    Model plain_model, model;
    const PrintObject &plain  = slice(plain_print, plain_model, false);
    const PrintObject &object = slice(print, model, true);
    REQUIRE(object.layers().size() == 20);
    auto solid = [](const Layer &layer) {
        Polygons out;
        for (const LayerRegion *layerm : layer.regions())
            for (const Surface &surface : layerm->fill_surfaces.surfaces)
                if (surface.is_solid())
                    append(out, to_polygons(surface.expolygon));
        return union_(out);
    };
    const double tiny_area = scaled<double>(0.1) * scaled<double>(0.1);
    bool         combined  = false;
    for (size_t i = 0; i < object.layers().size(); ++ i) {
        CAPTURE(i);
        const Polygons layer_solid = solid(*object.get_layer(int(i)));
        const Polygons plain_solid = solid(*plain.get_layer(int(i)));
        CHECK(area(diff(layer_solid, plain_solid)) < tiny_area);
        CHECK(area(diff(plain_solid, layer_solid)) < tiny_area);
        for (const LayerRegion *layerm : object.get_layer(int(i))->regions())
            for (const Surface &surface : layerm->fill_surfaces.surfaces)
                if (surface.surface_type == stInternal && surface.thickness_layers > 1) {
                    combined = true;
                    for (size_t j = i + 1 - surface.thickness_layers; j <= i; ++ j)
                        CHECK(area(intersection(to_polygons(surface.expolygon), solid(*object.get_layer(int(j))))) < tiny_area);
                }
    }
    CHECK(combined);
    const ShellProbe probe(object);
    CHECK(probe.top_depth(30.) == 7);
    CHECK(probe.top_depth(45.) == 3);
}

