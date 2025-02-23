import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons

import numpy as np
from CR3BP import CR3BP

plt.style.use("dark_background")

obj = CR3BP()


# The parametrized function to be plotted
def f(tf, x0, y0, z0, vx0, vy0, vz0):
    _, states = obj.propagate_orbit([x0, y0, z0, vx0, vy0, vz0], tf)
    return states[:, :3]


state_vars = ["tf", "x0", "y0", "z0", "vx0", "vy0", "vz0"]
slider_vars = ["tf", "x0", "z0", "vy0"]
# value, slidermin, slidermax
# defaultvals = {
#     "tf": [2, 0, 5],
#     "x0": [0.825, -1, 1],
#     "y0": [0, -1, 1],
#     "z0": [0.02, -1, 1],
#     "vx0": [0, -0.5, 0.5],
#     "vy0": [0.125, -0.5, 0.5],
#     "vz0": [0, -0.5, 0.5],
# }

defaultvals = {
    "tf": [2 * np.pi, 2 * np.pi, 10 * np.pi],
    "x0": [0.93784941, 0.90, 1.05],
    "y0": [0, -0.05, 0.05],
    "z0": [0.03, -0.05, 0.05],
    "vx0": [0, -0.5, 0.5],
    "vy0": [0.5480972, 0.4, 0.6],
    "vz0": [0, -0.5, 0.5],
}

# Create the figure and the line that we will manipulate
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
xyz_axes = ax.quiver(
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    label=["x", "y", "z"],
    length=0.15,
    normalize=True,
    color="lightgrey",
    linewidth=0.5,
)

surfR, surfTheta = np.meshgrid(np.linspace(0, 1.5, 2), np.linspace(0, 2 * np.pi, 500))
xy_plane = ax.plot_surface(
    surfR * np.cos(surfTheta),
    surfR * np.sin(surfTheta),
    0 * surfR,
    linewidth=0,
    facecolor=[0.25, 0.25, 0.25, 0.5],
)
ringTheta = np.linspace(0, 2 * np.pi, 500)
ax_rings = [
    ax.plot(
        ringR * np.cos(ringTheta),
        ringR * np.sin(ringTheta),
        color=[0.75, 0.75, 0.75, 0.75],
        linewidth=0.5,
    )
    for ringR in [0.25, 0.5, 0.75, 1, 1.25, 1.5]
]
ax_spokes = [
    ax.plot(
        [0, 1.5 * np.cos(ringTheta)],
        [0, 1.5 * np.sin(ringTheta)],
        color=[0.75, 0.75, 0.75, 0.75],
        linewidth=0.5,
    )
    for ringTheta in np.arange(0, 2 * np.pi, np.pi / 3)
]
ax.set_xlabel("x [LU]")
ax.set_ylabel("y [LU]")
ax.set_zlabel("z [LU]")
for n, axis in enumerate([ax.yaxis, ax.zaxis, ax.xaxis]):
    axis.pane.set_facecolor([0.3 + n / 40, 0.3 + n / 40, 0.3 + n / 40, 0.5])
    # the +n/40 is to match the color differences in original xyz panes
    axis.pane.set_edgecolor([0.75, 0.75, 0.75, 0.5])
    axis.gridlines.set_alpha(0.25)
ax.set_axis_off()
bodies = ax.plot([-obj.mu, 1 - obj.mu], [0, 0], "co", markersize=4)
lagrange_points = ax.plot(
    obj.L_points[0, :], obj.L_points[1, :], "wo", markersize=2, visible=False
)
traj = ax.plot(*f(*[defaultvals[var][0] for var in state_vars]).T, "r", lw=1)
plt.axis("equal")

slider_objs = []
slider_axes = []
num_sliders = len(slider_vars)
slider_spacing = 0.05
margin = 0.05

for n, varname in enumerate(slider_vars):
    label_firstletter = varname[0]
    label_rest = varname[1:]

    slider_vals = defaultvals[varname]

    slider_ax = fig.add_axes(
        [1 - margin - (num_sliders - n - 1) * slider_spacing, 0.1, 0.01, 0.85]
    )
    slider_axes.append(slider_ax)
    slider_obj = Slider(
        ax=slider_axes[-1],
        label="$" + label_firstletter + "_{" + label_rest + "}$",
        valmin=slider_vals[1],
        valmax=slider_vals[2],
        valinit=slider_vals[0],
        orientation="vertical",
        track_color=[0.25, 0.25, 0.25],
    )
    slider_obj.poly.set_visible(False)
    slider_obj.hline.set_visible(False)
    slider_objs.append(slider_obj)

# buttons
ax_buttons_x = 1 - margin - (num_sliders + 0.75) * slider_spacing
label_ax = fig.add_axes([ax_buttons_x, 0.7, 0.06, 0.04])
label_ax.text(
    0.5, 0.5, "Sliders:", verticalalignment="center", horizontalalignment="center"
)
label_ax.set_facecolor([0.1, 0.1, 0.1, 0])
label_ax.set_xticks([])
label_ax.set_yticks([])

reset_ax = fig.add_axes([ax_buttons_x, 0.65, 0.06, 0.04])
reset_btn = Button(reset_ax, "Reset", hovercolor="0.975", color="0.25")

zoomout_ax = fig.add_axes([ax_buttons_x, 0.6, 0.06, 0.04])
zoomout_btn = Button(zoomout_ax, "Coarser", hovercolor="0.975", color="0.25")

zoomin_ax = fig.add_axes([ax_buttons_x, 0.55, 0.06, 0.04])
zoomin_btn = Button(zoomin_ax, "Finer", hovercolor="0.975", color="0.25")

center_ax = fig.add_axes([ax_buttons_x, 0.5, 0.06, 0.04])
center_btn = Button(center_ax, "Center", hovercolor="0.975", color="0.25")


label_ax = fig.add_axes([0.025, 0.7, 0.075, 0.04])
label_ax.text(
    0.5, 0.5, "Toggle:", verticalalignment="center", horizontalalignment="center"
)
label_ax.set_facecolor([0.1, 0.1, 0.1, 0])
label_ax.set_xticks([])
label_ax.set_yticks([])
lagrange_ax = fig.add_axes([0.025, 0.6, 0.075, 0.04])
lagrange_btn = Button(lagrange_ax, "Lagrange Pts", hovercolor="0.975", color="0.25")

axtoggle_ax = fig.add_axes([0.025, 0.65, 0.075, 0.04])
axtoggle_btn = Button(axtoggle_ax, "Axes Type", hovercolor="0.975", color="0.25")

# Optimizer
optmethod_ax = fig.add_axes([0.025, 0.35, 0.075, 0.1])
optmethod_btns = RadioButtons(
    ax=optmethod_ax,
    labels=["Planar", "X-Fixed", "Z-Fixed"],
    active=1,
    label_props={"color": "w"},
    radio_props={
        "facecolor": "gray",
        "edgecolor": "white",
    },
)

optimize_ax = fig.add_axes([0.025, 0.31, 0.075, 0.04])
optimize_btn = Button(optimize_ax, "Optimize", hovercolor="0.975", color="0.25")


print_ax = fig.add_axes([0.025, 0.14, 0.075, 0.04])
print_btn = Button(print_ax, "Print ICs", hovercolor="0.975", color="0.25")


def get_state():
    slidervals = [slider.val for slider in slider_objs]
    ic = [slidervals[0], slidervals[1], 0, slidervals[2], 0, slidervals[3], 0]
    return np.array(ic)


# The function to be called anytime a slider's value changes
def update(val):
    states = f(*get_state())
    traj[0].set_data_3d(states[:, 0], states[:, 1], states[:, 2])
    fig.canvas.draw_idle()


def update_sliders(zoom=None):
    for slider in slider_objs:
        varname = "".join(ch for ch in slider.label._text if ch.isalnum())

        slider_vals = defaultvals[varname]

        if (varname == "tf") and zoom is not None:
            if zoom < 0.99:
                zoom *= 5
            elif zoom > 1.01:
                zoom /= 5
            # time zoom should be much less sensative

        old_valmin = slider.valmin
        old_valmax = slider.valmax
        curr_val = slider.val
        val_range = old_valmax - old_valmin
        new_valmax = (
            (curr_val + zoom * val_range / 2) if zoom is not None else slider_vals[2]
        )
        new_valmin = (
            (curr_val - zoom * val_range / 2) if zoom is not None else slider_vals[1]
        )
        new_valinit = curr_val if zoom is not None else slider_vals[0]

        slider.ax.set_ylim(new_valmin, new_valmax)
        slider.valmin = new_valmin
        slider.valmax = new_valmax
        for slider2 in slider_objs:
            if slider2 != slider:
                slider2.eventson = False
        slider.set_val(new_valinit)
        for slider2 in slider_objs:
            if slider2 != slider:
                slider2.eventson = True
        fig.canvas.draw_idle()


def reset(event):
    update_sliders()
    fig.canvas.draw_idle()


def center(event):
    update_sliders(zoom=1)


def zoomin(event):
    update_sliders(zoom=0.02)


def zoomout(event):
    update_sliders(zoom=50)


def swap_axes(event):
    if xy_plane.get_visible():
        xy_plane.set_visible(False)
        for ring in ax_rings:
            ring[0].set_visible(False)
        for spoke in ax_spokes:
            spoke[0].set_visible(False)
        xyz_axes.set_visible(False)
        ax.set_axis_on()
    else:
        xy_plane.set_visible(True)
        for ring in ax_rings:
            ring[0].set_visible(True)
        for spoke in ax_spokes:
            spoke[0].set_visible(True)
        xyz_axes.set_visible(True)
        ax.set_axis_off()
    fig.canvas.draw_idle()


def toggle_Lpoints(event):
    lagrange_points[0].set_visible(not lagrange_points[0].get_visible())
    fig.canvas.draw_idle()


def optimize_ics(event):
    curr_state = get_state()
    opt_option = optmethod_btns.value_selected
    if opt_option == "Planar":
        curr_state[3] = 0
        corrector = obj.targetter_planar_xfixed
    elif opt_option == "X-Fixed":
        corrector = obj.targetter_xfixed
    elif opt_option == "Z-Fixed":
        corrector = obj.targetter_zfixed
    else:
        print("ERR")
        return 0
    out_trj, _, ics, _, tfs = corrector(curr_state[1:], 1e-10, 1e-9, 25)
    if out_trj is None:
        new_state = curr_state
    else:
        new_state = [tfs[-1] * 2, *out_trj]
    for n, slider in enumerate(slider_objs):
        # eventson = False so that there isn't an infinite loop
        for slider2 in slider_objs:
            if slider2 != slider:
                slider2.eventson = False
        name = slider_vars[n]
        idx = state_vars.index(name)
        slider.set_val(new_state[idx])
        for slider2 in slider_objs:
            if slider2 != slider:
                slider2.eventson = True
    center(None)


def print_ics(event):
    print("\n\n\tInitial Conditions:")
    for i in range(len(slider_vars)):
        print("%4s: %.16e" % (slider_vars[i], slider_objs[i].val))


# def set_opt_option(event):
#     opt_option = event


reset_btn.on_clicked(reset)
center_btn.on_clicked(center)
zoomin_btn.on_clicked(zoomin)
zoomout_btn.on_clicked(zoomout)
axtoggle_btn.on_clicked(swap_axes)
lagrange_btn.on_clicked(toggle_Lpoints)
optimize_btn.on_clicked(optimize_ics)
print_btn.on_clicked(print_ics)
# optmethod_btns.on_clicked(set_opt_option)
# objzero_btns.on_clicked(toggle_objzero)
# optvar_btns.on_clicked(toggle_optvar)
for slider in slider_objs:
    slider.on_changed(update)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(right=0.7, left=0.025, bottom=0.025, top=0.975)
plt.show()
