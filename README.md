# Investigating the use of machine learning for lossy data compression on HEPMC data with Baler

## Overview

This repo and set of README's is set up to show my contribution to the investigation on the use of lossy data compression for HEPMC files, the instructions on how to set up Baler can be found [here](BALERSETUP.md), but also can and should be ran on the main branch of the forked repository for the main project [here](https://github.com/baler-collaboration/baler). The Baler scripts in this repo have been edited to allow for easier debugging and sanity checks in early development of my investigation to ensure everything was working as intended, these are not needed and running on the main branch is advisable.

## HEPMC

All the writeup's, result and step's taken in the goal of trying to get a working lossy compression of HEPMC files can be found in set of README's found in the [HEPMCarea Folder](/HEPMCArea/) with the one being found [here](/HEPMCArea/README.md).

## What I learnt / personal reflection

After every project I do, whether a personal project or university/internship related such as this one I liked to write a short personal reflection of what I learnt, both of how to proceed and in terms of programme/coding structure.

Starting with how to approach research the main thing I learnt is to take small steps even if they don't look like they are working towards the end goal in any way, this both helped mentally to feel I was accomplishing something every few days and to help feel like the end goal wasn't a marathon away.

In terms of programming/coding the main thing I learnt is to take some time at the beginning to set up a good experiment tracker early, something like wandb or Mlflow. After trying like 5/6 differnet sets of hyperparamaters/preprocessing techniques it got hard to track which belonged to which and exactly what configs were used for each, and made me wished I had used some time early on to set up something like Mlflow.




