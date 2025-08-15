# Notebooks

## extract_final_state_momentum

This notebook contains the functionaltiy to extract the final state momentum block of dilepton, lepton and final state particles from HEPMC files.

The number of particles to be extracted can be specified. With the following logic used.

    * 1. Momentum ordered Z boson decay dileptons
    * 2. Momentum ordered final state electrons
    * 3. Momentum ordered final state particles

To ensure no duplicate particles are selected (as 1 is a subset of 2 is a subset 3) they are first placed in a dictionary with the key as paritcle id and then taken back in insertion order.

These are then saved as an npz as this is the native file type used in the Baler software.

## smear_momenta 

Notebook that was used for the Rivet test of energy/mass conservation, which was taken by smearing the 4 momenta blocks of particles within events in HEPMC. 

## write_final_state_momenta

Basically decrepit.

Write a block of 4 momentas back into their associated HEPMC files.

## final_state_plots

Notebook for creating plots of variables from the final state momenta blocks, all logic for there are held in ../scripts/ .

Can take data from either HEPMC files or from NPZ. (The npz method is more up to date and functional and also faster).

Creating a plot requires 2 lists containining lists of 4 momenta for each event, the functionality to use 1 list only is not coded.

Plots can be created via final_state_plots([reconstructed, original], calc_variables.{variable}, Histroconfig, Cutconfig).

