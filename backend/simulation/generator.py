import numpy as np

class Config:
    X, Y, Z = 20, 20, 20

    AMBIENT = 20.0
    DIFFUSION_RATE = 0.065
    COOLING_RATE   = 0.0035
    HEATING_RATE   = 24.0
    THERMAL_INERTIA = 0.88

    # --- Event Logic ---
    SPAWN_CHANCE = 0.028
    MIN_DURATION = 40
    MAX_DURATION = 200

    INPUT_WINDOW = 50

def generate_test_sequence(seed=None, return_viz=True):
    if seed is not None:
        np.random.seed(seed)

    SIM_LENGTH = Config.INPUT_WINDOW

    grid = np.ones((1, Config.X, Config.Y, Config.Z), dtype=np.float32) * Config.AMBIENT
    clusters = []

    history = np.zeros((SIM_LENGTH, Config.X, Config.Y, Config.Z), dtype=np.float32)

    max_temp_log = []
    cluster_count_log = []

    for t in range(SIM_LENGTH):

        # Spawn clusters
        if np.random.rand() < Config.SPAWN_CHANCE:
            lifespan = np.random.randint(Config.MIN_DURATION, Config.MAX_DURATION)
            clusters.append({
                'pos': (
                    np.random.randint(2, 18),
                    np.random.randint(2, 18),
                    np.random.randint(2, 18)
                ),
                'end': t + lifespan
            })

        # Diffusion
        padded = np.pad(grid, ((0,0),(1,1),(1,1),(1,1)), mode='edge')
        neighbor_sum = (
            padded[:, 2:, 1:-1, 1:-1] + padded[:, :-2, 1:-1, 1:-1] +
            padded[:, 1:-1, 2:, 1:-1] + padded[:, 1:-1, :-2, 1:-1] +
            padded[:, 1:-1, 1:-1, 2:] + padded[:, 1:-1, 1:-1, :-2]
        )

        prev_grid = grid.copy()
        grid = grid + Config.DIFFUSION_RATE * (neighbor_sum - 6 * grid)

        # Cooling
        grid = grid - Config.COOLING_RATE * (grid - Config.AMBIENT)

        # Heating
        for c in clusters:
            if t < c['end']:
                cx, cy, cz = c['pos']
                if grid[0, cx, cy, cz] < 350.0:
                    grid[0, cx-1:cx+2, cy-1:cy+2, cz-1:cz+2] += Config.HEATING_RATE

        # Thermal inertia
        grid = (
            Config.THERMAL_INERTIA * prev_grid +
            (1.0 - Config.THERMAL_INERTIA) * grid
        ).astype(np.float32)

        history[t] = grid[0]

        max_temp_log.append(float(np.max(grid)))
        cluster_count_log.append(sum(t < c['end'] for c in clusters))

    # Normalize
    norm_history = (history - Config.AMBIENT) / (350.0 - Config.AMBIENT)
    X_test = norm_history[np.newaxis, ..., np.newaxis]

    if not return_viz:
        return X_test

    return X_test, {
        "frames": history,
        "max_temp_log": max_temp_log,
        "cluster_count_log": cluster_count_log
    }
