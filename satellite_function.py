import math

import numpy as np
from scipy.integrate import solve_ivp


class Clohessy_Wiltshire:
    """Relative-orbit propagation with the Clohessy-Wiltshire state matrix."""

    def __init__(self, R0_c=None, V0_c=None, R0_t=None, V0_t=None):
        self.R0_c = np.asarray(R0_c, dtype=np.float64)
        self.V0_c = np.asarray(V0_c, dtype=np.float64)
        self.R0_t = np.asarray(R0_t, dtype=np.float64)
        self.V0_t = np.asarray(V0_t, dtype=np.float64)
        self.u = 3.986e5

    def State_transition_matrix(self, t):
        r = 42164.0
        omega = math.sqrt(self.u / (r**3))
        tau = omega * t
        s = np.sin(tau)
        c = np.cos(tau)
        matrix = np.array(
            [
                [4 - 3 * c, 0, 0, s / omega, 2 * (1 - c) / omega, 0],
                [6 * (s - tau), 1, 0, -2 * (1 - c) / omega, 4 * s / omega - 3 * tau, 0],
                [0, 0, c, 0, 0, s / omega],
                [3 * omega * s, 0, 0, c, 2 * s, 0],
                [6 * omega * (c - 1), 0, 0, -2 * s, 4 * c - 3, 0],
                [0, 0, -omega * s, 0, 0, c],
            ],
            dtype=np.float64,
        )
        state_c = np.concatenate((self.R0_c, self.V0_c))
        state_t = np.concatenate((self.R0_t, self.V0_t))
        return matrix @ state_c, matrix @ state_t


class Numerical_calculation_method:
    """Fallback numerical propagation for experiments."""

    def __init__(self, R0_c=None, V0_c=None, R0_t=None, V0_t=None):
        self.R0_c = np.asarray(R0_c, dtype=np.float64)
        self.V0_c = np.asarray(V0_c, dtype=np.float64)
        self.R0_t = np.asarray(R0_t, dtype=np.float64)
        self.V0_t = np.asarray(V0_t, dtype=np.float64)
        self.pursuer_initial_state = np.concatenate((self.R0_c, self.V0_c))
        self.escaper_initial_state = np.concatenate((self.R0_t, self.V0_t))

    @staticmethod
    def orbit_ode(_t, x, thrust, direction):
        mu = 398600.0
        orbit_radius = 35786.0
        mass = 300.0
        omega = math.sqrt(mu / (orbit_radius**3))
        a_tr = thrust * math.cos(direction[0]) * math.cos(direction[1]) / mass
        a_tn = thrust * math.cos(direction[0]) * math.sin(direction[1]) / mass
        a_tt = thrust * math.sin(direction[0]) / mass
        return [
            x[3],
            x[4],
            x[5],
            2 * omega * x[4] + 3 * omega**2 * x[0] + a_tr,
            -2 * omega * x[3] + a_tn,
            -omega**2 * x[2] + a_tt,
        ]

    def numerical_calculation(self, t):
        params = (0.0, [1.0, 1.0])
        t_eval = np.linspace(0, t, max(int(t // 50) + 2, 2))
        solution1 = solve_ivp(self.orbit_ode, (0, t), self.pursuer_initial_state, args=params, t_eval=t_eval)
        solution2 = solve_ivp(self.orbit_ode, (0, t), self.escaper_initial_state, args=params, t_eval=t_eval)
        return solution1.y[:, -1], solution2.y[:, -1]


Time_window_of_danger_zone = Clohessy_Wiltshire
