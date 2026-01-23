import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, mixed_precision
from tensorflow.keras.models import load_model
import os
mixed_precision.set_global_policy('mixed_float16')

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    # Dimensions
    X, Y, Z = 20, 20, 20

    # Physics Parameters
    AMBIENT = 20.0
    DIFFUSION_RATE = 0.05      # Low enough to keep hotspots distinct
    COOLING_RATE   = 0.005     # Very low
    HEATING_RATE   = 20.0 
    THERMAL_INERTIA = 0.85

    # Event Logic
    SPAWN_CHANCE = 0.0159999
    MIN_DURATION = 25
    MAX_DURATION = 144

    # Data Generation
    SIM_LENGTH = 200           # Full physics lifecycle
    INPUT_WINDOW = 50          # Model input size
    PREDICT_HORIZON = 24       # Label

    # Batching
    SIMS_PER_STEP = 2          # How many physics sims to run in parallel (CPU)
    WINDOWS_PER_SIM = 10       # How many random slices to take from each sim
    # Effective Batch Size = 2 * 10 = 20 samples per training step

    # --- Training Scale ---
    TOTAL_SIMULATIONS = 10000   # Total distinct physics worlds to simulate
    # Total Samples = 2000 * 10 = 20,000 samples (Good for initial prototype)
    EPOCHS = 12 # total sample = 4,00,000samples

# ==========================================
# 2. THE DATA GENERATOR
# ==========================================
def generate_data():
    """
    Generator that yields batches of (X, y)
    - Normalized inputs (0.0 to 1.0)
    - Fixed Batch Size (Crucial for stable training)
    """
    # Target batch size based on your config
    TARGET_BATCH_SIZE = Config.SIMS_PER_STEP * Config.WINDOWS_PER_SIM

    while True:
        X_batch = []
        y_batch = []

        # Keep running sims until we have enough valid samples for a full batch
        while len(X_batch) < TARGET_BATCH_SIZE:

            # --- A. Run Parallel Physics (Same as before) ---
            grid = np.ones((Config.SIMS_PER_STEP, Config.X, Config.Y, Config.Z)) * Config.AMBIENT
            batch_clusters = [[] for _ in range(Config.SIMS_PER_STEP)]

            # Buffer: (200, 2, 20, 20, 20)
            history_buffer = np.zeros(
                (Config.SIM_LENGTH, Config.SIMS_PER_STEP, Config.X, Config.Y, Config.Z),
                dtype=np.float32
            )

            # Physics Loop
            for t in range(Config.SIM_LENGTH):
                # 1. Spawn
                for i in range(Config.SIMS_PER_STEP):
                    if np.random.rand() < Config.SPAWN_CHANCE:
                        lifespan = np.random.randint(Config.MIN_DURATION, Config.MAX_DURATION)
                        batch_clusters[i].append({
                            'pos': (np.random.randint(2,18), np.random.randint(2,18), np.random.randint(2,18)),
                            'end': t + lifespan
                        })

                # 2. Diffusion
                padded = np.pad(grid, pad_width=((0,0),(1,1),(1,1),(1,1)), mode='edge')
                neighbor_sum = (
                    padded[:, 2:, 1:-1, 1:-1] + padded[:, :-2, 1:-1, 1:-1] +
                    padded[:, 1:-1, 2:, 1:-1] + padded[:, 1:-1, :-2, 1:-1] +
                    padded[:, 1:-1, 1:-1, 2:] + padded[:, 1:-1, 1:-1, :-2]
                )
                prev_grid = grid
                grid = grid + Config.DIFFUSION_RATE * (neighbor_sum - 6 * grid)

                # 3. Cooling
                grid = grid - Config.COOLING_RATE * (grid - Config.AMBIENT)

                # 4. Heating
                for i in range(Config.SIMS_PER_STEP):
                    for c in batch_clusters[i]:
                        if t < c['end']:
                            cx, cy, cz = c['pos']
                            if grid[i, cx, cy, cz] < 350.0:
                                grid[i, cx-1:cx+2, cy-1:cy+2, cz-1:cz+2] += Config.HEATING_RATE

                # Thermal Inertia
                grid = (Config.THERMAL_INERTIA * prev_grid + (1.0 - Config.THERMAL_INERTIA) * grid).astype(np.float32)
                history_buffer[t] = grid.copy()

            # --- B. Slice Random Windows (With Normalization) ---
            max_start = Config.SIM_LENGTH - Config.INPUT_WINDOW - Config.PREDICT_HORIZON

            for i in range(Config.SIMS_PER_STEP):
                start_times = np.random.randint(0, max_start, size=Config.WINDOWS_PER_SIM)

                for t_start in start_times:
                    t_end = t_start + Config.INPUT_WINDOW

                    # 1. Grab Window
                    window = history_buffer[t_start:t_end, i, :, :, :]

                    window += np.random.normal(loc=0.0, scale=0.5, size=window.shape)

                    # *** CRITICAL FIX: NORMALIZATION ***
                    # Scale from approx 20-350 down to 0-1
                    # (Input - Min) / (Max - Min)
                    window = (window - Config.AMBIENT) / (350.0 - Config.AMBIENT)

                    # 2. Determine Label
                    future = history_buffer[t_end : t_end + Config.PREDICT_HORIZON, i, :, :, :]
                    max_future_temp = np.max(future)

                    if max_future_temp <= 30.0: is_explosion = 0.0
                    elif max_future_temp <= 60.0: is_explosion = 0.1
                    elif max_future_temp <= 100.0: is_explosion = 0.3
                    elif max_future_temp <= 150.0: is_explosion = 0.5
                    elif max_future_temp <= 200.0: is_explosion = 0.8
                    else: is_explosion = 1.0

                    # 3. Filter (Bias training)
                    if max_future_temp > 100 or np.random.rand() < 0.5:
                        X_batch.append(window)
                        y_batch.append(is_explosion)

        # --- C. Yield Fixed Batch Size ---
        # Trim to target size to avoid shape errors
        X_out = np.array(X_batch[:TARGET_BATCH_SIZE])[..., np.newaxis]
        y_out = np.array(y_batch[:TARGET_BATCH_SIZE])

        yield X_out, y_out

# ==========================================
# 3. THE MODEL (3D CNN + LSTM)
# ==========================================
def build_3d_model():
    input_shape = (Config.INPUT_WINDOW, Config.X, Config.Y, Config.Z, 1)

    model = models.Sequential(name="test_model")

    # --- 1. SPATIAL FEATURE EXTRACTION (TimeDistributed 3D CNN) ---
    # Processes each frame independently but shares weights across time

    model.add(layers.TimeDistributed(
        layers.Conv3D(16, (3,3,3), activation='relu', padding='same'),
        input_shape=input_shape
    ))
    model.add(layers.TimeDistributed(layers.MaxPooling3D((2,2,2))))

    model.add(layers.TimeDistributed(
        layers.Conv3D(32, (3,3,3), activation='relu', padding='same')
    ))
    model.add(layers.TimeDistributed(layers.MaxPooling3D((2,2,2))))

    model.add(layers.TimeDistributed(layers.Flatten()))

    # --- 2. TEMPORAL DYNAMICS (LSTM) ---
    # Learns the "Velocity" of the heat change
    model.add(layers.LSTM(96, return_sequences=False, dropout=0.25))

    # --- 3. DECISION HEAD ---
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer='adam',
        loss='MSE',
        # metrics=['accuracy']
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")]
    )

    return model

# ==========================================
# 4. MAIN EXECUTION ROUTINE
# ==========================================

print("--- SYSTEM CHECK ---")
# Check GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ GPU Detected: {gpus[0]}")
else:
    print("⚠️  NO GPU DETECTED. Training will be extremely slow.")
    print("Action: Go to Runtime -> Change runtime type -> T4 GPU")

# Instantiate
# model = build_3d_model()
model = tf.keras.models.load_model("silo_model_v4_1.keras", compile=True)
model.summary()

# Calculate Steps
# If we want 2000 total sims, and we do 2 sims per step:
# Steps per Epoch = 2000 / 2 = 1000 steps
steps_per_epoch = Config.TOTAL_SIMULATIONS // Config.SIMS_PER_STEP

print(f"\n--- TRAINING CONFIG ---")
print(f"Simulations per Epoch: {Config.TOTAL_SIMULATIONS}")
print(f"Samples per Epoch:     {Config.TOTAL_SIMULATIONS * Config.WINDOWS_PER_SIM}")
print(f"Batch Size (Effective):{Config.SIMS_PER_STEP * Config.WINDOWS_PER_SIM}")
print(f"Steps per Epoch:       {steps_per_epoch}")


# Create Generator
train_gen = generate_data()

# Optional: Mount Drive to save model
# from google.colab import drive
# drive.mount('/content/drive')
model.optimizer.learning_rate.assign(0.0002)
print("\n--- STARTING TRAINING ---")
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss',
    factor=0.5,      # Halving is good
    patience=4,      # Wait 4 epochs to confirm it's actually stuck, not just noise
    min_lr=1e-6,     # Allow it to go much lower for fine-tuning physics
    verbose=1
)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=Config.EPOCHS,
    initial_epoch=8,
    callbacks=[lr_schedule],
    verbose=1
)

# Save fine-tuned model
model.save("silo_model_v4_2.keras")
print("✅ Fine-tuning complete. Model saved as 'silo_model_v4_2.keras'")