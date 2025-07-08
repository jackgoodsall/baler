# Copyright 2022 Baler Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


def min_max_norm(data):
    """Min max scalar to the range [0, 1]
    
    """
    true_min = np.min(data)
    true_max = np.max(data)
    feature_range = true_max - true_min
    data = [((i - true_min) / feature_range) for i in data]
    data = np.array(data)
    return data

def log_min_max(data):
    """
    The log scaling function from  Rediscovery of the Higgs Boson Using
    Baler: A Compression-Based Machine Learning Tool Keerath Dhariwal
    """
    true_min = np.min(data)
    true_max = np.max(data)
    feature_range = np.log10(true_max) - np.log10(true_min)
    data = [((i - np.log10(true_min)) / feature_range) for i in data]
    data = np.array(data)
    return data
