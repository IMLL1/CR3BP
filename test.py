from CR3BP import *
import colorsys

plt.style.use("dark_background")


new_CR3BP = CR3BP()

# fig, ax = plt.subplots(layout="constrained")
# plt.plot([new_CR3BP.mu, 1 - new_CR3BP.mu], [0, 0], "o")
# plt.xlabel("x [LU]")
# plt.ylabel("y [LU]")
# plt.title("Trajectory (10 time units, start @x=0.8)")
# plt.grid(linestyle="dashed", c="gray", lw=1)
# x_list = np.round(np.arange(0.5, 0.7, 0.01), 2)
# for x0 in x_list:
#     num = (x0 - min(x_list)) / (max(x_list) - min(x_list))
#     color = colorsys.hsv_to_rgb(num, 1, 1)
#     t, states = new_CR3BP.propagate_orbit([0.8, 0, 0, 0, x0, 0], 10)
#     plt.plot(states[:, 0], states[:, 1], lw=0.5, color=color)
# plt.axis("equal")
# import matplotlib as mpl

# fig.colorbar(
#     mpl.cm.ScalarMappable(
#         norm=mpl.colors.Normalize(
#             min(x_list), (max(x_list) - min(x_list)) * 1 / 0.8 + min(x_list)
#         ),
#         cmap="hsv",
#     ),
#     ax=ax,
#     orientation="vertical",
#     label="Starting Velocity",
# )

vy = 0.5

fig, ax = plt.subplots(layout="constrained")
plt.plot([new_CR3BP.mu, 1 - new_CR3BP.mu], [0, 0], "o")
plt.xlabel("x [LU]")
plt.ylabel("y [LU]")
plt.title(f"Trajectory (10 time units, vy = {vy})")
plt.grid(linestyle="dashed", c="gray", lw=1)
x_list = np.round(np.arange(0.8, 0.83, 0.0001), 4)
for x0 in x_list:
    num = (x0 - min(x_list)) / (max(x_list) - min(x_list))
    color = colorsys.hsv_to_rgb(num, 1, 1)
    t, states = new_CR3BP.propagate_orbit([x0, 0, 0, 0, vy, 0], 10)
    plt.plot(states[:, 0], states[:, 1], lw=0.5, color=color)
plt.axis("equal")
import matplotlib as mpl

fig.colorbar(
    mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(
            min(x_list), (max(x_list) - min(x_list)) * 1 / 0.8 + min(x_list)
        ),
        cmap="hsv",
    ),
    ax=ax,
    orientation="vertical",
    label="Starting Position",
)
plt.show()
