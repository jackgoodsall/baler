# Scaling scripts

Readme to explain what scaling logic is applied in each of the scaling scripts. These scripts are questionably written so would be worth rewritting them.

## lorentz4vector.py

Converts the $(px, py, pz, E)$ vector into a $(E, pt, eta, phi)$ vector, this should allow for easier reconstruction as it is in the detectors more natural coordinate system.

## normal_data_transforms.py

Assumes data is in format of (px, py, pz, E) * N_particles

Applies standarad scalar transforms to the px, py variables and quatile scaling into a normal distribution for the E and pz variables.

Returns the momenta variables in the same order as the original.


## Global transform

Fits a quantile transform to normal distribution on the whole of the momenta block.

## 50partquantreverse and 50partquartest

Scales 4 momenta to GeV from MeV.

Convert to $(E, pt, eta, phi)$ notation.

Apply $log(1 + x)$ scaling to $E$ and $p_t$ to lessen effect of outliers.

Quantile scaling to normal distribution and save new 

$(E, pt, eta, phi)$ file.

The reverse scaling version, saves both the untransformed $(E, pt, eta, phi)$ version and $(px, py, pz, E)$.


