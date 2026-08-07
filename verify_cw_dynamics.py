import argparse
import math
from pathlib import Path

import numpy as np

from satellite_function import Clohessy_Wiltshire, EARTH_MU, GEO_ORBIT_RADIUS


def mean_motion():
    return math.sqrt(EARTH_MU / (GEO_ORBIT_RADIUS**3))


def propagate_from_initial(r0, v0, times):
    cw = Clohessy_Wiltshire(
        R0_c=np.asarray(r0, dtype=np.float64),
        V0_c=np.asarray(v0, dtype=np.float64),
        R0_t=np.zeros(3, dtype=np.float64),
        V0_t=np.zeros(3, dtype=np.float64),
    )
    return np.asarray([cw.propagate_relative(t) for t in times], dtype=np.float64)


def propagate_iteratively(r0, v0, dt, steps):
    state = np.concatenate((np.asarray(r0, dtype=np.float64), np.asarray(v0, dtype=np.float64)))
    states = [state.copy()]
    for _ in range(steps):
        state = Clohessy_Wiltshire(
            R0_c=state[:3],
            V0_c=state[3:],
            R0_t=np.zeros(3, dtype=np.float64),
            V0_t=np.zeros(3, dtype=np.float64),
        ).propagate_relative(dt)
        states.append(state.copy())
    return np.asarray(states, dtype=np.float64)


def expected_bounded_solution(x0, times, n):
    tau = n * times
    states = np.zeros((len(times), 6), dtype=np.float64)
    states[:, 0] = x0 * np.cos(tau)
    states[:, 1] = -2.0 * x0 * np.sin(tau)
    states[:, 3] = -x0 * n * np.sin(tau)
    states[:, 4] = -2.0 * x0 * n * np.cos(tau)
    return states


def summarize(name, states, reference=None):
    pos = states[:, :3]
    vel = states[:, 3:]
    dist = np.linalg.norm(pos, axis=1)
    speed = np.linalg.norm(vel, axis=1)

    print(f"\n{name}")
    print(f"  distance km: min={dist.min() / 1000:.6f}, max={dist.max() / 1000:.6f}, final={dist[-1] / 1000:.6f}")
    print(f"  speed m/s:   min={speed.min():.9f}, max={speed.max():.9f}, final={speed[-1]:.9f}")
    print(f"  final state: r={pos[-1]}, v={vel[-1]}")
    if reference is not None:
        err = np.max(np.abs(states - reference), axis=0)
        print(f"  max abs error vs reference: r={err[:3]} m, v={err[3:]} m/s")


def save_csv(path, times, direct, iterative, analytic):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "time_s,"
        "direct_x_m,direct_y_m,direct_z_m,direct_vx_mps,direct_vy_mps,direct_vz_mps,"
        "iter_x_m,iter_y_m,iter_z_m,iter_vx_mps,iter_vy_mps,iter_vz_mps,"
        "analytic_x_m,analytic_y_m,analytic_z_m,analytic_vx_mps,analytic_vy_mps,analytic_vz_mps"
    )
    data = np.column_stack((times, direct, iterative, analytic))
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main():
    parser = argparse.ArgumentParser(description="Verify current Clohessy-Wiltshire relative dynamics.")
    parser.add_argument("--initial-distance", type=float, default=20000.0, help="Initial radial separation x0 in meters.")
    parser.add_argument("--dt", type=float, default=100.0, help="Propagation interval in seconds.")
    parser.add_argument("--orbits", type=float, default=1.0, help="Number of GEO reference orbits to simulate.")
    parser.add_argument("--csv", default="outputs/cw_dynamics_check.csv", help="CSV output path.")
    args = parser.parse_args()

    n = mean_motion()
    period = 2.0 * math.pi / n
    steps = int(math.ceil(args.orbits * period / args.dt))
    times = np.arange(steps + 1, dtype=np.float64) * args.dt

    r0 = np.array([args.initial_distance, 0.0, 0.0], dtype=np.float64)
    v0 = np.array([0.0, -2.0 * n * args.initial_distance, 0.0], dtype=np.float64)

    direct = propagate_from_initial(r0, v0, times)
    iterative = propagate_iteratively(r0, v0, args.dt, steps)
    analytic = expected_bounded_solution(args.initial_distance, times, n)

    print("CW dynamics verification")
    print(f"  GEO mean motion n: {n:.12e} rad/s")
    print(f"  GEO period: {period:.3f} s ({period / 3600:.3f} h)")
    print(f"  initial r: {r0} m")
    print(f"  initial v: {v0} m/s")
    print("  expected bounded solution for this initial condition:")
    print("    x(t)=x0*cos(n*t), y(t)=-2*x0*sin(n*t)")
    print(f"    distance should stay between {args.initial_distance / 1000:.3f} km and {2 * args.initial_distance / 1000:.3f} km")

    summarize("Direct STM from initial state", direct, analytic)
    summarize("Iterative env-style STM, zero action each step", iterative, direct)

    save_csv(Path(args.csv), times, direct, iterative, analytic)
    print(f"\nCSV saved to: {args.csv}")


if __name__ == "__main__":
    main()
