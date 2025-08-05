import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import os
# --------------------------
# Config
# --------------------------
UNTRANSFORM = True

RAW_DATA_DIR = "HEPMCArea/DATA/"
RAW_DATA_FILE_NAME = "MomentumOrdered100-000Events20ParticlesFlattened.npz"
SAVE_DATA_FILE_TO_DIR = "HEPMCArea/processedData/20EventData/"
SAVE_DATA_FILE_TO_NAME = "100-000Events20ParticlesNormalScaled.npz"
UNTRANSFORMED_DATA_FILE = "decompressed.npz"
UNTRANSFORMED_SAVE_FILE = "BiggerAE.npz"
print(os.getcwd())
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


quantile_transformers_pz = []   # Quantile transformers for pz
quantile_transformers_e = []    # Quantile transformers for e
standard_scalers = []           # Standard scalers for px and py

transformed_columns = []


for i in range(N_particles):
    base_idx = i * N_features
    px, py, pz, e = (
        input_data[:, base_idx],
        input_data[:, base_idx+1],
        input_data[:, base_idx+2],
        input_data[:, base_idx+3]
    )


    scaler_px = StandardScaler().fit(px.reshape(-1, 1))
    new_px = scaler_px.transform(px.reshape(-1, 1)).reshape(-1)

    scaler_py = StandardScaler().fit(py.reshape(-1, 1))
    new_py = scaler_py.transform(py.reshape(-1, 1)).reshape(-1)


    qtf_pz = QuantileTransformer(output_distribution="normal")
    new_pz = qtf_pz.fit_transform(pz.reshape(-1, 1)).reshape(-1)

    qtf_e = QuantileTransformer(output_distribution="normal")
    new_e = qtf_e.fit_transform(e.reshape(-1, 1)).reshape(-1)

    # Store transformers
    standard_scalers.append((scaler_px, scaler_py))
    quantile_transformers_pz.append(qtf_pz)
    quantile_transformers_e.append(qtf_e)

    # Append transformed features in px, py, pz, e order
    transformed_columns.extend([new_px, new_py, new_pz, new_e])

# Stack back into (N_samples, N_particles * N_features)
transformed_data = np.vstack(transformed_columns).T
print("Transformed data shape:", transformed_data.shape)
#np.savez(SAVE_DATA_FILE_TO_DIR + SAVE_DATA_FILE_TO_NAME, data=transformed_data, names=variable_names)


if UNTRANSFORM:
    saved_obj = np.load(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_DATA_FILE)
    transformed_data = saved_obj["data"]

    restored_columns = []
    idx = 0
    for i in range(N_particles):
        px_t = transformed_data[:, idx]; idx += 1
        py_t = transformed_data[:, idx]; idx += 1
        pz_t = transformed_data[:, idx]; idx += 1
        e_t  = transformed_data[:, idx]; idx += 1


        px_inv = standard_scalers[i][0].inverse_transform(px_t.reshape(-1, 1)).reshape(-1)
        py_inv = standard_scalers[i][1].inverse_transform(py_t.reshape(-1, 1)).reshape(-1)

        pz_inv = quantile_transformers_pz[i].inverse_transform(pz_t.reshape(-1, 1)).reshape(-1)


        e_inv = quantile_transformers_e[i].inverse_transform(e_t.reshape(-1, 1)).reshape(-1)

        restored_columns.extend([px_inv, py_inv, pz_inv, e_inv])


    unscaled_data = np.vstack(restored_columns).T * 1_000  
    print("Sample restored row:", unscaled_data[0])
    np.savez(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_SAVE_FILE,
             data=unscaled_data, names=variable_names)
