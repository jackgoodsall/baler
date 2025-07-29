"""
File to transform given data using defined transformations, and to be able to untransform them (scale them back) into the original range.


"""

import numpy as np
from sklearn.proprocessing import StandardScalar

# I/o Configs
RAW_DATA_DIR = "../DATA/"
RAW_DATA_FILE_NAME = "100-000Events.npz" 
SAVE_DATA_FILE_TO_DIR = "../processedData/"
SAVE_DATA_FILE_TO_NAMe = "Scaled100-000Events.npz"

# Load data store names and data
input_file_object = np.load(RAW_DATA_DIR + RAW_DATA_FILE_NAME)
variable_names = input_file_object["names"]
input_data = input_file_object["data"]

# Unpack data
energy0, px0, py0, pz0 = input_data[0 :4]
energy1, px1, py1, pz1 = input_data[4 :]

new_data = []
for column in [px0, py0, pz0, px1, py1, pz1]:
    transformers = StandardScalar()
    new_column = transformers.fit_transform(column)
    new_data.append(new_column)

