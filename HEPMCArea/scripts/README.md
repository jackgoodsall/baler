# Scripts

The scripts in this folder are the set of functions used for making the plots in the notebook [final_state_plots](../Notebooks/final_state_plots.ipynb). 


## event_cuts.py

This file includes the classes and methods used to apply event_cuts based on the kinematics of the final state dilepton pair.

The main ones include

etacut - For Atlas this includes that neither lepton is in the region $1.37 \gt abs(\eta) \lt 1.52$


ptcut - For our data the standard cuts were used, such that  pt[0] > 25 Gev and Pt[1] > 20 GeV.

## hepmc_configs.py

Contains dataclasses for configurables for the final state histrogram, ratio and scatter plots.


## calc_variables.py

File containing functions to calculate or get specific variables from the 4 momenta lists.

Add things here for creating plots of any variables you want, always assumed that the [0] outer list is the reconstructed and the [1] outer list is the original.

