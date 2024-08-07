"""
Adapted from Alfonso Gonzalez's class at https://github.com/alfonsogonzalez/AWP
Some inspiration taken from NASA Johnson Space Center's Copernicus: https://www.nasa.gov/general/copernicus/
"""

from scipy.integrate import solve_ivp
from scipy.optimize import newton, minimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CR3BP:
    def __init__(self, mu=1.215058560962404e-2):
        if mu < 0.5:
            self.mu = mu
        else:
            print("mu should be <0.5. Setting mu=1-mu")
            self.mu = 1 - mu

        self.L_points = self.lagranges()

    def lagranges(self):
        def optFunc(x):
            zero = (
                -(x + self.mu) * (1 - self.mu) / (np.abs(x + self.mu) ** 3)
                - (x - 1 + self.mu) * self.mu / (np.abs(x - 1 + self.mu) ** 3)
                + x
            )
            return zero

        L1 = [newton(optFunc, (1 - self.mu) / 2), 0]
        L2 = [newton(optFunc, (2 - self.mu) / 2), 0]
        L3 = [newton(optFunc, -1), 0]
        L4 = [1 / 2 - self.mu, np.sqrt(3) / 2]
        L5 = [1 / 2 - self.mu, -np.sqrt(3) / 2]

        return np.array([L1, L2, L3, L4, L5]).T

    def pseudopotential(self, x, y, z):
        r1mag = np.sqrt((x + self.mu) ** 2 + y**2 + z**2)
        r2mag = np.sqrt((x - 1 + self.mu) ** 2 + y**2 + z**2)
        Ugrav = -((1 - self.mu) / r1mag + self.mu / r2mag)
        Ucent = -0.5 * (x**2 + y**2)

        return Ugrav + Ucent

    def eom(self, t, state):
        x, y, z, vx, vy, vz = state
        xyz = state[:3]
        r1vec = xyz + np.array([self.mu, 0, 0])
        r2vec = xyz + np.array([self.mu - 1, 0, 0])
        r1mag = np.linalg.norm(r1vec)
        r2mag = np.linalg.norm(r2vec)

        ddxyz = (
            -(1 - self.mu) * r1vec / r1mag**3
            - self.mu * r2vec / r2mag**3
            + np.array([2 * vy + x, -2 * vx + y, 0])
        )

        dstate = np.zeros(6)
        dstate[:3] = state[3:]
        dstate[3:] = ddxyz
        return dstate

    def propagate_orbit(
        self,
        state0,
        tspan,
        propagator="LSODA",
        rtol=1e-9,
        atol=1e-9,
        dense_output=False,
    ):
        self.solution = solve_ivp(
            fun=self.eom,
            t_span=(0, tspan),
            y0=np.array(state0),
            method=propagator,
            atol=atol,
            rtol=rtol,
            dense_output=dense_output,
        )

        self.states = self.solution.y.T
        self.ts = self.solution.t

        return self.ts, self.states

    def plot_2d(self):
        plt.figure()
        plt.plot([self.mu, 1 - self.mu], [0, 0], "o")
        plt.plot(self.states[:, 0], self.states[:, 1], lw=1)
        plt.axis("equal")
        plt.xlabel("x [LU]")
        plt.ylabel("y [LU]")
        plt.title("Trajectory (2D)")
        plt.grid(linestyle="dashed", lw=0.5, c="gray")
        plt.show()

    def plot_3d(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot([self.mu, 1 - self.mu], [0, 0], "o")
        ax.plot(self.states[:, 0], self.states[:, 1], self.states[:, 2], lw=1)
        plt.axis("equal")
        ax.set_xlabel("x [LU]")
        ax.set_ylabel("y [LU]")
        ax.set_zlabel("z [LU]")

        plt.title("Trajectory (3D)")
        plt.grid(linestyle="dashed", lw=0.5, c="gray")
        plt.show()

    def plot_pseudopotential(self):
        base = self.pseudopotential(0.85 * self.L_points[0, 0], 0, 0)
        L1 = self.pseudopotential(self.L_points[0, 0], 0, 0)
        L2 = self.pseudopotential(self.L_points[0, 1], 0, 0)
        L3 = self.pseudopotential(self.L_points[0, 2], 0, 0)
        L23mean = (L2 + L3) / 2
        L45 = self.pseudopotential(self.L_points[0, 3], self.L_points[1, 3], 0)
        L451 = (L3 + 2 * L45) / 3
        level_list = [base, L1, L2, L23mean, L3, L451, L45]

        X, Y = np.meshgrid(np.linspace(-1.5, 1.5, 250), np.linspace(-1.5, 1.5, 250))
        Z = self.pseudopotential(X, Y, 0 * X)
        plt.figure()
        plt.plot([self.mu, 1 - self.mu], [0, 0], "ok", ms=2.5)
        plt.contourf(
            X,
            Y,
            Z,
            levels=level_list,
            colors=plt.cm.turbo(np.linspace(0, 1, len(level_list))),
        )
        plt.axis("equal")
        plt.xlabel("x [LU]")
        plt.ylabel("y [LU]")
        plt.title("Pseudopotential")
        plt.grid(linestyle="dashed", lw=0.5, c="gray")
        plt.show()

    # TODO: Tolerance specification, geq and leq constraints, multiple fixed ICs?
    def find_periodic_orbit(self,
                            opt_vars=["tf", "z", "vy"],
                            obj_zero=["vx", "y"],
                            init_guess=[2.77, 0.82285, 0, 0.05, 0, 0.17, 0],
                            tol=None):
        func_inputs = pd.Series({"tf":0, "x": 1, "y": 2, "z": 3, "vx": 4, "vy": 5, "vz": 6})
        fixed_vars = list(func_inputs.drop(index=opt_vars).keys())
        init_guess = np.array(init_guess)
        opt_paramnums = list(func_inputs[opt_vars].values)
        def minFunc(inputs):
            states_in = np.zeros(7)
            
            # insert non-optimization variables
            states_in[func_inputs[fixed_vars]] = init_guess[func_inputs[fixed_vars]]
            states_in[func_inputs[opt_vars]] = inputs
            # prevent time from going to zero (bad optimization)
            if states_in[0] < 0.5*init_guess[0]: states_in[0] = 0.5*init_guess[0]
            
            _, states = self.propagate_orbit(states_in[1:], states_in[0])
            state_fin = states[-1,:]
            
            # get objective states and their norm
            obj_states = np.array(state_fin[func_inputs[obj_zero]-1])
            obj_func = np.linalg.norm(obj_states)
            return obj_func

        # init input is init guess for non-set variables
        init_input = init_guess[opt_paramnums]
        min_object = minimize(minFunc, init_input, method = 'Nelder-Mead', tol=tol)
        minimizing_guess=min_object.x
        optimal_state = np.zeros(7)
        optimal_state[func_inputs[fixed_vars]] = init_guess[func_inputs[fixed_vars]]
        optimal_state[func_inputs[opt_vars]] = minimizing_guess
        return optimal_state