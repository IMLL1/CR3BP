import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import newton
from CR3BP import CR3BP

plt.style.use("dark_background")

obj = CR3BP()


# The parametrized function to be plotted
def f(tf, x0, y0, z0, vx0, vy0, vz0):
    t, states = obj.propagate_orbit([x0, y0, z0, vx0, vy0, vz0], tf)
    return states[:, :3]


# Define initial parameters
x0 = 0.75; y0 = 0; z0 = 0; vx0 = 0; vy0 = 0.5; vz0 = 0; tf = 5

# Create the figure and the line that we will manipulate
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
line = ax.plot(*f(tf, x0, y0, z0, vx0, vy0, vz0).T, "r", lw=1.5)
line = line[0]
ax.set_xlabel("x [LU]")
ax.set_ylabel("y [LU]")
ax.set_zlabel("z [LU]")
ax.plot([obj.mu, 1 - obj.mu], [0, 0], "bo", markersize=4)
ax.plot(obj.L_points[0, :], obj.L_points[1, :], "ko", markersize=2)
ax.set_title("Trajectory")
plt.grid(linestyle="dashed", c="gray", lw=1)
plt.axis("equal")


tf_axis = fig.add_axes([0.05, 0.1, 0.01, 0.85])
tf_slider = Slider(
    ax=tf_axis,
    label="$t_f$",
    valmin=0,
    valmax=15,
    valinit=tf,
    orientation="vertical",
    color=[.25,.25,.25],
)


sliderVars = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]
slider_objs = []
slider_axes = []
num_sliders = len(sliderVars)

for n, varName in enumerate(sliderVars):
    label_firstletter = varName[0]
    label_rest = varName[1:]
    
    sliderSpacing = 0.03
    margin = 0.025
    
    slider_ax = fig.add_axes([1-margin-(num_sliders-n-1)*sliderSpacing, 0.1, 0.01, 0.85])
    slider_axes.append(slider_ax)
    slider_obj = Slider(
        ax=slider_axes[-1],
        label="$"+label_firstletter+"_{"+label_rest+"}$",
        valmin=-1,
        valmax=1,
        valinit=globals()[varName],
        orientation="vertical",
        color=[.25,.25,.25],
    )
    slider_objs.append(slider_obj)


# The function to be called anytime a slider's value changes
def update(val):
    states = f(tf_slider.val, *[slider.val for slider in slider_objs])
    line.set_data_3d(states[:, 0], states[:, 1], states[:, 2])
    fig.canvas.draw_idle()


# register the update function with each slider
for slider in [*slider_objs, tf_slider]:
    slider.on_changed(update)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(right=0.75)
plt.show()
