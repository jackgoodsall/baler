#!/usr/bin/env python3
import os
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from joblib import dump


IN_DIR  = "HEPMCArea/DATA/"
IN_NAME = "50ParticlesFlattened.npz"   # expects px,py,pz,E per particle
OUT_DIR = "HEPMCArea/processedData/50partsCylGeV/"

OUT_DATA_NAME = "Eptetaphi_qt_final.npz"            # quantile-transformed features
QTF_PKL_NAME  = "Eptetaphi_qtf.pkl"           # fitted QuantileTransformer
META_NAME     = "Eptetaphi_qt_meta.npz"       # indices + names

os.makedirs(OUT_DIR, exist_ok=True)

in_path   = os.path.join(IN_DIR, IN_NAME)
out_path  = os.path.join(OUT_DIR, OUT_DATA_NAME)
qtf_path  = os.path.join(OUT_DIR, QTF_PKL_NAME)
meta_path = os.path.join(OUT_DIR, META_NAME)


obj = np.load(in_path, allow_pickle=True)
Xcart = obj["data"].astype(np.float64)      # shape (N, P*4)
N, T = Xcart.shape
assert T % 4 == 0, "Total features must be multiple of 4"
P = T // 4
print(f"Loaded {N} samples, {P} particles × 4 (px,py,pz,E)")


px = Xcart[:, 0::4] / 1000
py = Xcart[:, 1::4] / 1000
pz = Xcart[:, 2::4] / 1000
E  = Xcart[:, 3::4] / 1000
 
pt  = np.hypot(px, py)            # stable sqrt(px^2+py^2)
phi = np.arctan2(py, px)               # [-pi, pi)
eta = np.arcsinh(pz / np.clip(pt, 1e-12, None))  # asinh(pz/pt)

# Interleave -> [E, pt, eta, phi] per particle
X_eptef = np.empty_like(Xcart, dtype=np.float64)
for i in range(P):
    base = 4*i
    X_eptef[:, base+0] = E[:,  i].ravel()
    X_eptef[:, base+1] = pt[:, i].ravel()
    X_eptef[:, base+2] = eta[:,i].ravel()
    X_eptef[:, base+3] = phi[:,i].ravel()

names = np.array([f"p{i}_{c}" for i in range(P) for c in ("E","pt","eta","phi")], dtype=np.str_)

# Index slices
E_idx   = np.arange(0, T, 4)
pt_idx  = np.arange(1, T, 4)
eta_idx = np.arange(2, T, 4)
phi_idx = np.arange(3, T, 4)


X_pre = X_eptef.copy()
X_pre[:, E_idx]  = np.log1p(np.maximum(X_pre[:, E_idx], 0.0))
X_pre[:, pt_idx] = np.log1p(np.maximum(X_pre[:, pt_idx], 0.0))
# put angles in a sane central range before QTF
X_pre[:, phi_idx] = np.arctan2(np.sin(X_pre[:, phi_idx]), np.cos(X_pre[:, phi_idx]))

qtf = QuantileTransformer(output_distribution="normal", random_state=42, copy=True)
Z = qtf.fit_transform(X_pre)


np.savez(out_path, data=Z, names=names)
dump(qtf, qtf_path)
np.savez(meta_path, names=names, E_idx=E_idx, pt_idx=pt_idx, eta_idx=eta_idx, phi_idx=phi_idx)

print(f"Saved QT features: {out_path}  shape={Z.shape}")
print(f"Saved QTF: {qtf_path}")
print(f"Saved meta: {meta_path}")
