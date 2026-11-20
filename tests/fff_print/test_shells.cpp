#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "libslic3r/Config.hpp"
#include "libslic3r/Model.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/PrintBase.hpp"
#include "libslic3r/PrintConfig.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include "libslic3r/TriangleSelector.hpp"

#include <boost/algorithm/string/predicate.hpp>
#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
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
    for (const ModifierBox &box : modifiers) {
        TriangleMesh mesh = make_cube(box.x1 - box.x0, 30., box.z1 - box.z0);
        mesh.translate(float(box.x0), -5.f, float(box.z0));
        object->add_volume(std::move(mesh), ModelVolumeType::PARAMETER_MODIFIER, false)->config.apply(box.config);
    }
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
    CHECK(modifier->has_top_shell_ramp());
    CHECK(modifier->has_bottom_shell_ramp());
    CHECK(modifier->config().bottom_shell_layers.value == 5);

    const ShellCounts counts = modifier->shell_default_counts();
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
    CHECK(modifier->has_top_shell_ramp() == top_ramp);
    CHECK(modifier->has_bottom_shell_ramp() == bottom_ramp);
    const ShellCounts counts = modifier->shell_default_counts();
    CHECK(counts.top_layers == (top_ramp ? 3 : 5));
    CHECK(counts.bottom_layers == (bottom_ramp ? 3 : 5));
    // The part is never keyed and builds its own shell.
    const PrintRegion *part = region_with_top_shell_layers(print, 3);
    REQUIRE(part != nullptr);
    CHECK(part->ramp_key() == RampKey{});
    CHECK(part->shell_default_counts().top_layers == 3);
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
    const std::string case_name      = GENERATE("modifiers on Hard", "Ramp set in the process and on the part", "Ramp modifier changing no count");
    CAPTURE(wall_generator, case_name);

    const DynamicPrintConfig plain_config = shell_config(wall_generator);
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
