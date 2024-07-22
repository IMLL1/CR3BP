import matplotlib.pyplot as plt

import numpy as np
from CR3BP import CR3BP
import pandas as pd
import matplotlib as mpl

# TODO: https://bluescarni.github.io/heyoka.py/notebooks/Periodic%20orbits%20in%20the%20CR3BP.html

plt.style.use("dark_background")

obj = CR3BP() 

# The parametrized function to be plotted
def propagate(init_state):
    _, states = obj.propagate_orbit(init_state[:-1], init_state[-1], rtol=1e-8, atol=1e-8)
    return states[:, :3]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("x [LU]")
ax.set_ylabel("y [LU]")
ax.set_zlabel("z [LU]")
ax.set_axis_off()

u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
sf = 1737.1/389703
xSphere = np.cos(u)*np.sin(v)*sf + 1 - obj.mu
ySphere = np.sin(u)*np.sin(v)*sf
zSphere = np.cos(v)*sf
sphere = ax.plot_surface(xSphere, ySphere, zSphere, color="white")

ics = pd.read_csv("periodic_orbits.csv", dtype=float, index_col=0).sort_index()
ics.columns = ics.columns.str.strip()
ics = ics[["x0 (LU)","y0 (LU)","z0 (LU)","vx0 (LU/TU)","vy0 (LU/TU)","vz0 (LU/TU)","Period (TU)"]]
ics.index = ics.index.astype(int)
trajs = []

plot_index=[]
id = max(ics.index)
while id>1:
    plot_index.append(id)
    id -= (1 + (max(ics.index)-id)//10)
# plot_index = ics.index
    
for n, id in enumerate(plot_index):
    state0 = np.array(ics.loc[id])
    col = mpl.colors.hsv_to_rgb([n/len(plot_index), 1, 1])
    traj = ax.plot(*propagate(state0).T, color=col, lw=0.5)
    trajs.append(traj)

plt.axis("equal")
plt.tight_layout()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.show()