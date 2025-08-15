import numpy as np
import vector
import os

# — Config paths (same as before) —
SAVE_DATA_FILE_TO_DIR = "HEPMCArea/DATA/"
SAVE_DATA_FILE_TO_NAME = "50ParticlesFlattened.npz"

# 0) Load the transformed data back from .npz
saved_obj = np.load(os.path.join(SAVE_DATA_FILE_TO_DIR, SAVE_DATA_FILE_TO_NAME))
data = saved_obj["data"]        # shape: (N_samples, N_particles*4)
variable_names = saved_obj["names"]

N_samples, total = data.shape
N_particles = total // 4


flat = data.reshape(-1, 4)
print(data.shape)
# build one vector.obj per 4-tuple
vectors = [
    vector.obj(
        px=float(px),
        py=float(py),
        pz=float(pz),
        E =float(E),
    )
    for px, py, pz, E in flat
]


E_all   = np.array([v.E   for v in vectors]).reshape(N_samples, N_particles)
pt_all  = np.array([v.pt  for v in vectors]).reshape(N_samples, N_particles)
eta_all = np.array([v.eta for v in vectors]).reshape(N_samples, N_particles)
phi_all = np.array([v.phi for v in vectors]).reshape(N_samples, N_particles)

# interleave and write to CSV just like before
out = np.empty((N_samples, N_particles * 4), float)
for i in range(N_particles):
    out[:, 4*i:4*i+4] = np.stack([
        E_all[:, i],
        pt_all[:, i],
        eta_all[:, i],
        phi_all[:, i],
    ], axis=1)

cols = []
for i in range(N_particles):
    cols += [f"p{i}_E", f"p{i}_pt", f"p{i}_eta", f"p{i}_phi"]

np.savez("lorentz4vector.npz", data = out, names = cols)