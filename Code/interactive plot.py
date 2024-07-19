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
xyz_axes = ax.quiver([0,0,0], [0,0,0], [0,0,0], [1,0,0], [0,1,0], [0,0,1], label=["x","y","z"], length=0.15, normalize=True, color='lightgrey', linewidth=0.5)

surfR, surfTheta = np.meshgrid(np.linspace(0,1.5,2), np.linspace(0,2*np.pi,500))
xy_plane = ax.plot_surface(surfR*np.cos(surfTheta), surfR*np.sin(surfTheta), 0*surfR, linewidth=0, facecolor=[0.25,0.25,0.25,0.5])
ringTheta = np.linspace(0,2*np.pi,500)
ax_rings = [ax.plot(ringR*np.cos(ringTheta),ringR*np.sin(ringTheta), color=[.75,.75,.75,.75], linewidth=0.5) for ringR in [.25, .5, .75, 1, 1.25, 1.5]]
ax_spokes = [ax.plot([0,1.5*np.cos(ringTheta)],[0,1.5*np.sin(ringTheta)], color=[.75,.75,.75,.75], linewidth=0.5) for ringTheta in np.arange(0,2*np.pi,np.pi/3)]


bodies = ax.plot([obj.mu, 1 - obj.mu], [0, 0], "co", markersize=4)
lagrange_points = ax.plot(obj.L_points[0, :], obj.L_points[1, :], "wo", markersize=2)

traj = ax.plot(*f(tf, x0, y0, z0, vx0, vy0, vz0).T, "r", lw=1.5)
traj = traj[0]

plt.axis("equal")
# ax.set_xlabel("x [LU]")
# ax.set_ylabel("y [LU]")
# ax.set_zlabel("z [LU]")
ax.set_axis_off()


tf_axis = fig.add_axes([0.05, 0.1, 0.01, 0.85])
tf_slider = Slider(ax=tf_axis, label="$t_f$", valmin=0, valmax=15, valinit=tf, orientation="vertical", color=[.3,.3,.3], track_color=[.1,.1,.1])


slider_vars = ["x0", "y0", "z0", "vx0", "vy0", "vz0"]
slider_objs = []
slider_axes = []
num_sliders = len(slider_vars)

for n, varName in enumerate(slider_vars):
    label_firstletter = varName[0]
    label_rest = varName[1:]
    
    slider_spacing = 0.04
    margin = 0.025
    
    slider_ax = fig.add_axes([1-margin-(num_sliders-n-1)*slider_spacing, 0.1, 0.01, 0.85])
    slider_axes.append(slider_ax)
    slider_obj = Slider(ax=slider_axes[-1],
                        label="$"+label_firstletter+"_{"+label_rest+"}$",
                        valmin=-1.25, valmax=1.25, valinit=globals()[varName],
                        orientation="vertical", color=[.3,.3,.3], track_color=[.1,.1,.1])
    slider_objs.append(slider_obj)


# The function to be called anytime a slider's value changes
def update(val):
    states = f(tf_slider.val, *[slider.val for slider in slider_objs])
    traj.set_data_3d(states[:, 0], states[:, 1], states[:, 2])
    fig.canvas.draw_idle()

# register the update function with each slider
for slider in [*slider_objs, tf_slider]:
    slider.on_changed(update)

# buttons
center_ax = fig.add_axes([0.7, 0.4, 0.06, 0.04])
center_btn = Button(center_ax, 'Center', hovercolor='0.975', color='0.25')

zoomin_ax = fig.add_axes([0.7, 0.5, 0.06, 0.04])
zoomin_btn = Button(zoomin_ax, 'Finer', hovercolor='0.975', color='0.25')

zoomout_ax = fig.add_axes([0.7, 0.6, 0.06, 0.04])
zoomout_btn = Button(zoomout_ax, 'Coarser', hovercolor='0.975', color='0.25')

reset_ax = fig.add_axes([0.7, 0.7, 0.06, 0.04])
reset_btn = Button(reset_ax, 'Reset', hovercolor='0.975', color='0.25')

def make_sliders(zoom=None):
    for num in range(len(slider_objs)):
        old_valmin = slider_objs[num].valmin
        old_valmax = slider_objs[num].valmax
        curr_val = slider_objs[num].val
        val_range = old_valmax-old_valmin
        new_valmax = (curr_val + zoom*val_range) if zoom is not None else 1.25
        new_valmin = (curr_val - zoom*val_range) if zoom is not None else -1.25
        new_valinit = curr_val if zoom is not None else globals()[slider_vars[num]]
        
        slider_ax = fig.add_axes([1-margin-(num_sliders-num-1)*slider_spacing, 0.1, 0.01, 0.85])
        slider_objs[num].ax.remove()
        slider_obj = Slider(ax=slider_ax,
                            label=slider_objs[num].label._text,
                            valmin=new_valmin, valmax=new_valmax, valinit=new_valinit,
                            orientation=slider_objs[num].orientation,
                            track_color=slider_objs[num].track._facecolor,
                            color=slider_objs[num].poly._facecolor)
        slider_objs[num] = slider_obj
        for slider in [*slider_objs, tf_slider]:
            slider.on_changed(update)

def reset(event):
    make_sliders()
reset_btn.on_clicked(reset)

def center(event):
    make_sliders(zoom=1)
center_btn.on_clicked(center)

def zoomin(event):
    make_sliders(zoom=0.1)
zoomin_btn.on_clicked(zoomin)

def zoomout(event):
    make_sliders(zoom=10)
zoomout_btn.on_clicked(zoomout)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(right=.75, left=0.075, bottom=0.025, top=0.975)
plt.show()