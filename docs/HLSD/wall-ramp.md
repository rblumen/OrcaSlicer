# Wall and Shell Ramp — High Level Design

## Purpose and scope

A volume modifier that sets a different wall count or shell count from its part
changes the walls and shells in one step at its outline. With the Hard
transition, which is the default, a modifier with more walls gets closed walls
of its own around its outline, including inside the part. The result is a
sudden change in stiffness and a seam at the edge of the reinforced area.

The Ramp transition grades that change instead. Inside the modifier the walls
(or shells) have the modifier's count. Outside it, they return to the part's
count linearly over a ramp along the part's surface. The outer wall and the
innermost wall stay continuous across the ramp, and walls end one at a time
inside the wall. A zone can thicken the walls or shells, or thin them, down to
none.

Each of walls, top shells and bottom shells has its own pair of options, set on
the modifier:

| Option | Meaning |
| --- | --- |
| `wall_loops_transition` | Hard or Ramp, for the change from the part's `wall_loops` to the modifier's. |
| `wall_ramp_taper_ratio` | rho: mm of ramp per mm of wall thickness changed. 0 keeps the hard step. |
| `top_shell_layers_transition`, `bottom_shell_layers_transition` | Hard or Ramp, for the change of the shell depth, which counts both the shell layers and the minimum shell thickness. |
| `top_shell_ramp_taper_ratio`, `bottom_shell_ramp_taper_ratio` | rho_s: mm of ramp per mm of shell depth changed. 0 keeps the hard step. |

The transitions are region options, but they act only on a modifier that sets
them itself. They are not inherited from the process, the object, a layer range
or an enclosing modifier, and they do nothing on a part. The settings tab shows
them only when a mesh modifier is selected. Wall and shell ramps are
independent: one modifier can ramp both, or different, overlapping modifiers
can ramp each.

Painted modifiers are not zone sources: a painted modifier shares its host's
mesh, so its zone would be the whole part. Slicing ignores the transitions on a
painted modifier: it gets no ramp key and slices like Hard, and validation does
not check them.

## The zone and the ramp

The zone of a modifier is the part of the part's surface that lies inside the
modifier mesh. A face the modifier shares with the part from inside belongs to
the zone; a face it rests on from outside does not. The signed distance d from
the zone's boundary, measured along the part's surface, is negative inside the
zone.

Walls (`WallThicknessField`):

    t(d) = w * (N1 - dN * clamp(d / L_r, 0, 1)),    dN = N1 - N0,    L_r = |dN| * rho * w

w is the bead width, N0 the part's wall count and N1 the modifier's.

Shells (`ShellDepthField`): the stock shell rule puts a layer k layers from a
top (bottom) surface, dz from it in Z, in the shell where `k < n` or
`dz < T - EPSILON`, with n the shell layer count and T the minimum shell
thickness. Inside a zone the rule is the modifier's (n1, T1); over the ramp
both terms go linearly back to the part's (n0, T0):

    L_s = |D1 - D0| * rho_s,    D = shell_depth(n, T, h)

D is the depth the stock projection reaches at the nominal layer height h.
Where n and T move in opposite directions, a layer that both ends put in the
shell stays in it across the ramp, so the shell never dips mid-ramp.

In both cases the ramp lies outside the zone, whether the zone thickens or
thins. Where zones and ramps overlap, the thicker result wins: the thickest
over the zones that thicken at a point, else the thinnest over those that thin.
Each zone carries the N0 (or n0, T0) of the part it lies on, so a modifier
spanning several parts gets one zone per part, each tapering to that part's
count.

## Why geodesic distance

The ramp follows the surface because the wall it grades lies on the surface.
Distance in XY on each layer would restart on every layer and would not see a
zone on the face above or below, so a ramp could not turn a corner or reach a
top face from a side face. Euclidean distance in 3D cuts through the material:
on a thin plate a zone on one face would reach the opposite face. Geodesic
distance along the part's surface does neither, and its iso-lines continue
across edges onto the next face, so each layer's slice crosses the ramp where
the surface does.

`SurfaceZoneDistance` computes it with the heat method on the part's surface,
remeshed around the zone's boundary: short edges at the boundary, growing to
longer ones where the distance is read, and only as far as the ramp reaches. It
saturates beyond the ramp. Where the solve does not reach, such as a second
shell of the part with no boundary of its own, the straight-line distance to the
boundary stands in. Whether a point is in the zone is decided by winding
numbers, so the part and modifier meshes have to be closed and consistently
oriented. The distance is accurate to a few percent; it is smoothed where
distances arriving by two routes meet, such as behind the corner of a hole.

The walls and the shells each build their own field from the same distance
machinery, with their own profile. Each field is built once per object and
then only queried, from any number of threads.

## The region merge

A modifier region with the Ramp wall transition must share the walls of its
part, or the stock code would wall its outline. `PrintApply.cpp` keys the
modifier's `PrintRegion` with a `RampKey` holding the counts of the region it
modifies. The key keeps two modifiers with the same config over parents with
different counts in separate regions, and through `wall_group_loops()` makes
`Layer::is_perimeter_compatible()` group the modifier with its part, so that
`Layer::make_perimeters()` runs one wall generator over their merged islands
and splits the fill area back per region afterwards. Only the wall part of the
key affects the grouping.

The merge is the reason the other wall settings of the modifier must match the
part's: regions whose wall settings differ cannot share walls. Where the merge
fails, the modifier keeps the hard step and a slicing warning says why. The
shell part of the key needs no merge: the shell depth field decides the depth
from the surface whatever region a point belongs to, and solid fills already
merge across regions with equal fill parameters.

The key is set only where the transition is active: for walls with the Arachne
wall generator and outside spiral vase mode (`wall_ramp_active()`), for shells
outside spiral vase mode (`shell_ramp_active()`). Elsewhere the modifier keeps
the Hard transition, and its region is the stock one, so a project with the
options off slices exactly as before.

## The wall band

On each layer, the band of an island is the region within the local wall
thickness t of its contour, measured inward along the contour normals
(`WallBandShape`). Its inner edge is the wall/infill boundary. The contour is
sampled every w/4; each sample is mapped to the part's surface at the layer's
height, offset slightly outward along its normal so that on a thin wall the
right face is found. Where t is below the thinnest wall the region allows, t is
taken as 0, which is where a ramp to no walls ends.

`PerimeterGenerator::process_arachne()` swaps the band in for the island only
where the field changes the thickness, so the stock wall path is untouched
elsewhere. The infill gets the island minus the band.

## Why Arachne on a band

A fixed wall count can only step. Arachne fills an arbitrary region with
variable-width beads and decides itself where a bead ends, so a band whose
width tapers gives a wall whose bead count tapers. Arachne treats both edges of
the band as outer walls, so the bead along the outer contour and the bead along
the band's inner edge run unbroken, and beads drop only in the interior of the
wall. The band call raises Arachne's wall count cap so it never cuts the band
short, sets the distribution count to 1 so only the middle bead flexes, and
raises the transition angle above the ramp's slope, atan(1 / rho), so that each
drop is one bead at a time even on a steep ramp.

Arachne numbers the band's beads from both edges. `wall_band_paths()`
renumbers them by their depth from the outer wall, as stock walls are numbered,
so wall order, seams and speeds behave as for stock walls.

Drop stations are shifted by delta = rho * w / 4 along the surface on alternate
layers (a brick bond), so bead ends do not stack vertically.

## The shell depth

The shell field is thresholded per layer into levels, which both shell
projections clip with: `discover_vertical_shells()` when Ensure vertical shell
thickness is All, `discover_horizontal_shells()` otherwise. Within the reach of
a layer's levels the levels decide which layers below a top surface (above a
bottom surface) are solid, so the shell runs past the stock range where a zone
thickens it and stops short where one thins it; outside that reach the stock
rule applies. Inside the reach the surfaces of every region are projected, so a
zone's shell carries on below a modifier shorter than it. A modifier region on
the Ramp is projected with the counts of the region it modifies where no zone
decides (`PrintRegion::shell_default_counts()`), so its depth only changes where
the field says so.

The solid edge moves by about rho_s * h per layer, so the inner face of the
shell is a planar staircase that follows the taper. Non-planar shells are not
built.

## Constraints

- **Default off.** Both transitions default to Hard, and nothing is keyed or
  built without a modifier set to Ramp, so existing projects and profiles slice
  unchanged.
- **Walls need Arachne.** With the Classic wall generator a modifier set to
  Ramp uses the Hard wall transition, and `Print::validate()` warns once per
  object. Shells work with both generators.
- **No spiral vase mode.** A Ramp modifier is an error there.
- **Only one wall on top surfaces** set on a Ramp wall modifier itself is an
  error, as the ramp keeps all its walls under top surfaces.
- **Consistent settings.** A modifier whose Ramp settings or wall
  compatibility differ between the object's height ranges gets no zone and
  keeps the hard step, with a warning. The same holds for shells.
- **Unsupported wall paths.** Islands that take Alternate extra wall, Only one
  wall on first layer, Extra perimeters on overhangs, or extra walls keep the
  stock walls on those layers, with a warning.
- **Per-part gradients.** A modifier whose region is kept apart from its
  part's by a per-part mixed-filament gradient keeps the hard wall step, with a
  warning.
- **Closed meshes.** Zone membership uses winding numbers; a zone narrower than
  about the fine edge of the remesh may be missed.

All of these fallbacks are reported as slicing warnings (`Print::validate()`
for configuration errors and the Classic generator, `active_step_add_warning()`
in `PrintObject::make_perimeters()` and `prepare_infill()` for the rest), which
the GUI shows as notifications.
