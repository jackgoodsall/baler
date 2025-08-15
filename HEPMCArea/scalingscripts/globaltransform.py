import numpy as np
from sklearn.preprocessing import QuantileTransformer
import os


UNTRANSFORM = True


RAW_DATA_DIR = "HEPMCArea/DATA/"
RAW_DATA_FILE_NAME = "50ParticlesFlattened.npz"
SAVE_DATA_FILE_TO_DIR = "/eos/user/j/jgoodsal/baler/HEPMCArea/balercompressionexperiments/random/"
SAVE_DATA_FILE_TO_NAME = "..npz"
UNTRANSFORMED_DATA_FILE = "decompressed.npz"
UNTRANSFORMED_SAVE_FILE = "GlobaltransformAE.npz"

print(os.getcwd())


input_file_object = np.load(RAW_DATA_DIR + RAW_DATA_FILE_NAME)
variable_names = input_file_object["names"]
input_data = input_file_object["data"]  # shape: [N_samples, N_particles * N_features]

# Convert MeV → GeV if needed
input_data /= 1000.0  

N_samples, total_features = input_data.shape
print(f"Loaded: {N_samples} samples, {total_features // 4} particles, 4 features per particle")


qtf_global = QuantileTransformer(output_distribution="normal", random_state=42)
transformed_data = qtf_global.fit_transform(input_data)

print("Transformed data shape:", transformed_data.shape)
np.savez(SAVE_DATA_FILE_TO_DIR + SAVE_DATA_FILE_TO_NAME, data = transformed_data, names = variable_names)

if UNTRANSFORM:
    saved_obj = np.load(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_DATA_FILE)
    transformed_data = saved_obj["data"]

    # Inverse transform back to original space
    unscaled_data = qtf_global.inverse_transform(transformed_data) * 1000.0  # back to MeV

    print("Sample restored row:", unscaled_data[0])
    np.savez(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_SAVE_FILE,
             data=unscaled_data, names=variable_names)
