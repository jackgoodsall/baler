import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer

# --------------------------
# Utility functions
# --------------------------
def log_minmax(arr):
    eps = 10e-12
    arr_clipped = np.clip(arr, eps, None) 
    log_arr = np.log10(arr_clipped)
    log_min = np.log10(arr_clipped.min())
    log_max = np.log10(arr_clipped.max())
    return (
        (log_arr - log_min) / (log_max - log_min) if log_max != log_min else 0,
        (log_min.astype(np.float32), log_max.astype(np.float32))
    )

def inverse_log_minmax(norm_arr, *log):
    norm_arr = np.array(norm_arr)
    log_arr = norm_arr * (log[1] - log[0]) + log[0]
    return np.power(10, log_arr, dtype=np.float64)

# --------------------------
# Config
# --------------------------
UNTRANSFORM = False

RAW_DATA_DIR = "HEPMCArea/DATA/"
RAW_DATA_FILE_NAME = "MomentumOrdered100-000Events50ParticlesFlattened.npz"
SAVE_DATA_FILE_TO_DIR = "HEPMCArea/processedData/"
SAVE_DATA_FILE_TO_NAME = "100-000Events50ParticlesScaled.npz"
UNTRANSFORMED_DATA_FILE = "ScaledQuantileTransformed.npz"
UNTRANSFORMED_SAVE_FILE = "UnscaledQuantileTransformed.npz"

# --------------------------
# Load data
# --------------------------
input_file_object = np.load(RAW_DATA_DIR + RAW_DATA_FILE_NAME)
variable_names = input_file_object["names"]
input_data = input_file_object["data"]  # shape: [N_samples, N_particles * N_features]

# Convert MeV → GeV if needed
input_data /= 1000.0

N_samples, total_features = input_data.shape
N_features = 4  # px, py, pz, e
N_particles = total_features // N_features

print(f"Loaded: {N_samples} samples, {N_particles} particles, {N_features} features per particle")

# --------------------------
# Forward Transform
# --------------------------
if not UNTRANSFORM:
    quantile_transformers = []
    standard_scalers = []
    energy_scales = []

    transformed_columns = []

    # Loop through particles
    for i in range(N_particles):
        base_idx = i * N_features
        px, py, pz, e = (
            input_data[:, base_idx],
            input_data[:, base_idx+1],
            input_data[:, base_idx+2],
            input_data[:, base_idx+3]
        )

        # Quantile transform for pz
        qtf = QuantileTransformer(output_distribution="normal")
        pz_trans = qtf.fit_transform(pz.reshape(-1, 1)).reshape(-1)
        quantile_transformers.append(qtf)

        # Standard scale px, py, transformed pz
        scalers_for_particle = []
        new_px = StandardScaler().fit_transform(px.reshape(-1, 1)).reshape(-1)
        scalers_for_particle.append(StandardScaler().fit(px.reshape(-1, 1)))

        new_py = StandardScaler().fit_transform(py.reshape(-1, 1)).reshape(-1)
        scalers_for_particle.append(StandardScaler().fit(py.reshape(-1, 1)))

        new_pz = StandardScaler().fit_transform(pz_trans.reshape(-1, 1)).reshape(-1)
        scalers_for_particle.append(StandardScaler().fit(pz_trans.reshape(-1, 1)))

        # Log-MinMax for energy
        new_e, scale = log_minmax(e)
        energy_scales.append(scale)

        # Append transformed features in px, py, pz, e order
        transformed_columns.extend([new_px, new_py, new_pz, new_e])
        standard_scalers.append(scalers_for_particle)

    # Stack back into (N_samples, N_particles * N_features)
    transformed_data = np.vstack(transformed_columns).T
    print(transformed_data.shape)
    np.savez(SAVE_DATA_FILE_TO_DIR + SAVE_DATA_FILE_TO_NAME,
             data=transformed_data, names=variable_names)

# --------------------------
# Inverse Transform
# --------------------------
if UNTRANSFORM:
    saved_obj = np.load(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_DATA_FILE)
    transformed_data = saved_obj["data"]

    restored_columns = []
    idx = 0
    for i in range(N_particles):
        # Extract transformed px, py, pz, e
        px_t = transformed_data[:, idx]; idx += 1
        py_t = transformed_data[:, idx]; idx += 1
        pz_t = transformed_data[:, idx]; idx += 1
        e_t  = transformed_data[:, idx]; idx += 1

        # Inverse standard scaling for px, py, pz
        px_inv = standard_scalers[i][0].inverse_transform(px_t.reshape(-1, 1)).reshape(-1)
        py_inv = standard_scalers[i][1].inverse_transform(py_t.reshape(-1, 1)).reshape(-1)
        pz_norm = standard_scalers[i][2].inverse_transform(pz_t.reshape(-1, 1)).reshape(-1)

        # Inverse quantile transform for pz
        pz_inv = quantile_transformers[i].inverse_transform(pz_norm.reshape(-1, 1)).reshape(-1)

        # Inverse log-minmax for energy
        e_inv = inverse_log_minmax(e_t, *energy_scales[i])

        # Append restored features
        restored_columns.extend([px_inv, py_inv, pz_inv, e_inv])

    unscaled_data = np.vstack(restored_columns).T * 1000.0  # Convert back to MeV if needed

    np.savez(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_SAVE_FILE,
             data=unscaled_data, names=variable_names)
