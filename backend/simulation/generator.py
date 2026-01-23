import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # --- Dimensions ---
    X, Y, Z = 20, 20, 20

    # --- Physics Parameters (Your Selection) ---
    AMBIENT = 20.0
    DIFFUSION_RATE = 0.05      # Low enough to keep hotspots distinct
    COOLING_RATE   = 0.005     # Very low (Grain is an insulator)
    HEATING_RATE   = 20.0       # Net growth ~3C per hour
    THERMAL_INERTIA = 0.85

    # --- Event Logic ---
    SPAWN_CHANCE = 0.0159999
    MIN_DURATION = 25
    MAX_DURATION = 144

    # --- Data Generation Strategy (Case C Modified) ---
    # SIM_LENGTH = 50           # Full physics lifecycle
    INPUT_WINDOW = 50          # Model input size
    # PREDICT_HORIZON = 24       # Label: "Will it explode in next 24h?"

    # --- Batching ---
    # SIMS_PER_STEP = 2          # How many physics sims to run in parallel (CPU)
    # WINDOWS_PER_SIM = 10       # How many random slices to take from each sim
    # Effective Batch Size = 2 * 10 = 20 samples per training step

    # --- Training Scale ---
    # TOTAL_SIMULATIONS = 10000   # Total distinct physics worlds to simulate
    # # Total Samples = 2000 * 10 = 20,000 samples (Good for initial prototype)
    # EPOCHS = 12 # total sample = 4,00,000samples

def generate_test_sequence(seed=None, return_viz=True):
    if seed is not None:
        np.random.seed(seed)

    SIM_LENGTH = Config.INPUT_WINDOW

    grid = np.ones((1, Config.X, Config.Y, Config.Z), dtype=np.float32) * Config.AMBIENT
    clusters = []

    history = np.zeros((SIM_LENGTH, Config.X, Config.Y, Config.Z), dtype=np.float32)

    # --- NEW: metrics ---
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

        # --- metrics ---
        max_temp_log.append(float(np.max(grid)))
        cluster_count_log.append(sum(t < c['end'] for c in clusters))

    # Normalize for model (UNCHANGED)
    norm_history = (history - Config.AMBIENT) / (350.0 - Config.AMBIENT)
    X_test = norm_history[np.newaxis, ..., np.newaxis]

    if not return_viz:
        return X_test

    return X_test, {
        "frames": history,  # RAW temperatures (for Plotly)
        "max_temp_log": max_temp_log,
        "cluster_count_log": cluster_count_log
    }
