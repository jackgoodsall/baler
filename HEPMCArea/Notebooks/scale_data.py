"""
File to transform given data using defined transformations, and to be able to untransform them (scale them back) into the original range.


"""

import numpy as np
from sklearn.preprocessing import StandardScaler


# Oscars slog min max function https://github.com/OscarrrFuentes/baler/tree/higgs_rediscovery/V4_results
# Rewritten by chatgpt because the original was an eye abomination
def log_minmax(arr):
    arr = np.array(arr)  # Ensure input is a NumPy array
    log_arr = np.log10(arr)
    log_min = np.log10(arr.min())
    log_max = np.log10(arr.max())
    return (
        (log_arr - log_min) / (log_max - log_min),
        (log_min.astype(np.float32), log_max.astype(np.float32))
    )

def inverse_log_minmax(norm_arr, *log):
    norm_arr = np.array(norm_arr)  # Ensure input is a NumPy array
    log_arr = norm_arr * (log[1] - log[0]) + log[0]
    return np.power(10, log_arr,  dtype=np.float64)


# Untransform
UNTRANSFORM = False

# I/o Configss
RAW_DATA_DIR = "HEPMCArea/DATA/"
RAW_DATA_FILE_NAME = "MomentumOrdered100-000Events.npz" 
SAVE_DATA_FILE_TO_DIR = "HEPMCArea/processedData/"
SAVE_DATA_FILE_TO_NAME = "MomentumOrderedScaled100-000Events.npz"
UNTRANSFORMED_DATA_FILE = "Unscaledreconstructed3.npz"

UNTRANSFORMED_SAVE_FILE = "Scaledreconstructed3.npz"

# Load data store names and data
input_file_object = np.load(RAW_DATA_DIR + RAW_DATA_FILE_NAME)
variable_names = input_file_object["names"]
input_data = input_file_object["data"]

input_data /= 1000

energy0, px0, py0, pz0 = input_data[:,3], input_data[:,0], input_data[:,1], input_data[:,2]
energy1, px1, py1, pz1 =  input_data[:,7], input_data[:,4], input_data[:,5], input_data[:,6]

# Fit standard scalar to 
new_data = []
transformers = []
for column in [px0, py0, pz0, px1, py1, pz1]:
    transformer = StandardScaler()
    new_column = transformer.fit_transform(column.reshape(-1 ,1 ))
    new_data.append(new_column.reshape(-1))
    transformers.append(transformer)

new_energy0, (scale_features0) = log_minmax(energy0)
new_energy1, (scale_features1) = log_minmax(energy1)


new_data.insert(3, new_energy0)
new_data.append(new_energy1)


np.savez(SAVE_DATA_FILE_TO_DIR + SAVE_DATA_FILE_TO_NAME
         , data = np.array(new_data, dtype  =float).T, names = variable_names ) 


if UNTRANSFORM:
    untransformed_data_ojb = np.load(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_DATA_FILE)
    untransformed_data = untransformed_data_ojb["data"]


    energy01, px0, py0, pz0 = untransformed_data[:,3], untransformed_data[:,0], untransformed_data[:,1], untransformed_data[:,2]
    energy11, px1, py1, pz1 =  untransformed_data[:,7], untransformed_data[:,4], untransformed_data[:,5], untransformed_data[:,6]

    new_data = []
    for column, transformer in zip([px0, py0, pz0, px1, py1, pz1], transformers):
        new_column = transformer.inverse_transform(column.reshape(-1 ,1 ))
        new_data.append(new_column.reshape(-1))
    print(len(new_data))

    new_energy0 = inverse_log_minmax(energy01, *scale_features0)
    new_energy1 = inverse_log_minmax(energy11, *scale_features1)

    new_data.insert(3, new_energy0)
    new_data.append(new_energy1)

    data = np.array(new_data , dtype  =float).T
    data *= 1000
    
    np.savez(SAVE_DATA_FILE_TO_DIR + UNTRANSFORMED_SAVE_FILE
         , data = data , names = variable_names ) 




