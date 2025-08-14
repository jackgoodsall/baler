#!/usr/bin/env python3
import os
import numpy as np
from joblib import load

# --------------------------
# Config
# --------------------------
IN_DIR   = "/eos/user/j/jgoodsal/baler/HEPMCArea/balercompressionexperiments/50partsTransformerGeV/"
IN_NAME  = "decompressed.npz"             # should contain Z in (E,pt,eta,phi) order
QTF_PKL  = "Eptetaphi_qtf.pkl"            # fitted QuantileTransformer from transform step
META_NAME = "Eptetaphi_qt_meta.npz"

OUT_DIR  = IN_DIR
OUT_NAME = "pxpypzE_from_qt copy.npz"

in_path   = os.path.join(IN_DIR, IN_NAME)
qtf_path  = os.path.join(IN_DIR, QTF_PKL)
meta_path = os.path.join(IN_DIR, META_NAME)
out_path  = os.path.join(OUT_DIR, OUT_NAME)


obj = np.load(in_path, allow_pickle=True)
Z = obj["data"].astype(np.float64)
N, T = Z.shape
assert T % 4 == 0, "Total features must be multiple of 4"
P = T // 4

meta = np.load(meta_path, allow_pickle=True)
names = np.asarray(meta["names"]).astype(np.str_)
E_idx   = meta["E_idx"]; pt_idx  = meta["pt_idx"]
eta_idx = meta["eta_idx"]; phi_idx = meta["phi_idx"]


qtf = load(qtf_path)
X_pre = qtf.inverse_transform(Z)

# invert the log1p on E and pt
X_pre[:, E_idx]  = np.expm1(X_pre[:, E_idx]) * 1000
X_pre[:, pt_idx] = np.expm1(X_pre[:, pt_idx]) * 1000
# keep phi in [-pi, pi) for geometric stability
X_pre[:, phi_idx] = np.arctan2(np.sin(X_pre[:, phi_idx]), np.cos(X_pre[:, phi_idx]))


E   = X_pre[:, E_idx] 
pt  = np.maximum(X_pre[:, pt_idx], 0.0)  # guard tiny negatives
eta = X_pre[:, eta_idx]
phi = X_pre[:, phi_idx]

px = pt * np.cos(phi)
py = pt * np.sin(phi)
pz = pt * np.sinh(eta)


print(E[0])

# interleave to [px, py, pz, E]
X_cart = np.empty_like(X_pre)
for i in range(P):
    base = 4*i
    X_cart[:, base+0] = px[:, i]
    X_cart[:, base+1] = py[:, i]
    X_cart[:, base+2] = pz[:, i]
    X_cart[:, base+3] = E[:,  i] 

out_names = np.array([f"p{i}_{c}" for i in range(P) for c in ("px","py","pz","E")], dtype=np.str_)

-
os.makedirs(OUT_DIR, exist_ok=True)
np.savez(out_path, data=X_cart, names=out_names)
print(f"Saved: {out_path}  shape={X_cart.shape}")

# quick sanity
p_mag = np.sqrt(X_cart[:, 0::4]**2 + X_cart[:, 1::4]**2 + X_cart[:, 2::4]**2)
m2 = (X_cart[:, 3::4]**2) - (p_mag**2)
print(f"Mean |p|: {p_mag.mean():.4g}, fraction m^2<0: {np.mean((m2<0).ravel()):.2%}")

I_WANT = True

if I_WANT:
    np.savez("testtesttest.npz",
         data=X_pre,          # same shape as input, now unscaled
         names=names)