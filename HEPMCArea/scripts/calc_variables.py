"""
Script for calculating and getting different values or components for a particles 4 momenta.
"""
from typing import List
import numpy as np


def calc_invariant_mass(particle_lists: List[List[float]]) -> List[List[float]]:
    """Calculates the invariant mass and returns as lists with each entry a distribution"""
    invariant_masses = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            # Extract (px, py, pz, E) for both particles
            p1 = np.array([pair[0], pair[1], pair[2], pair[3]])
            p2 = np.array([pair[4], pair[5], pair[6], pair[7]])
            combined = p1 + p2
            inv_mass = np.sqrt(combined[3]**2 - np.sum(combined[0:3]**2))
            l.append(inv_mass)
        invariant_masses.append(l)
    invariant_masses = np.array(invariant_masses, dtype=float)
    return invariant_masses

def calc_transverse_momentum(particle_lists: List[List[float]]) -> List[List[float]]:
    """Calculates transverse momentum and returns as lists with each entry a distribution"""
    transverse_momentums  = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            pt1 = np.sqrt(np.square(pair[0]) + np.square(pair[1]))
            pt2 = np.sqrt(np.square(pair[4]) + np.square(pair[5]))
            l.append(pt1)
            l.append(pt2)
        transverse_momentums.append(l)
    transverse_momentums = np.array(transverse_momentums, dtype = float)
    print(transverse_momentums.shape)
    return transverse_momentums

def calc_energy(particle_lists: List[List[float]]) -> List[List[float]]:
    """Calculates energies and returns as lists with each entry a distribution"""
    energies = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            energy1 = pair[3]  # Energy component of particle 1
            energy2 = pair[7]  # Energy component of particle 2
            l.append(energy1)
            l.append(energy2)
        energies.append(l)
    energies = np.array(energies, dtype=float)
    print(energies.shape)
    return energies

def calc_px(particle_lists: List[List[float]]) -> List[List[float]]:
    
    """Calculates(retrieves) momentum in x direction and returns as lists with each entry a distribution"""
    pxs = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            l.append(pair[0])
            l.append(pair[4])
        pxs.append(l)
    pxs = np.array(pxs, dtype=float)
    return pxs

def get_e_0(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the e[0] for each particle in each list."""
    return np.array(
        [[particle[3] for particle in particles if len(particle) > 8] 
         for particles in particle_lists],
        dtype=float
    )

def get_px_0(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the px[0] for each particle in each list."""
    return np.array(
        [[particle[0] for particle in particles if len(particle) > 8] 
            for particles in particle_lists],
        dtype=float
    )

def get_py_0(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the py[0]  for each particle in each list."""
    return np.array(
        [[particle[1] for particle in particles if len(particle) > 8] 
            for particles in particle_lists],
        dtype=float
    )

def get_pz_0(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the pz[0] for each particle in each list."""
    return np.array(
        [[particle[2] for particle in particles if len(particle) > 8] 
            for particles in particle_lists],
        dtype=float
    )

def get_e_1(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the e[1] for each particle in each list."""
    return np.array(
        [[particle[7] for particle in particles if len(particle) > 8] 
         for particles in particle_lists],
        dtype=float
    )

def get_px_1(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the px[0] for each particle in each list."""
    return np.array(
        [[particle[4] for particle in particles if len(particle) > 8] 
            for particles in particle_lists],
        dtype=float
    )

def get_py_1(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the py[1] for each particle in each list."""
    return np.array(
        [[particle[5] for particle in particles if len(particle) > 8] 
            for particles in particle_lists],
        dtype=float
    )

def get_pz_1(particle_lists: List[List[float]]) -> np.ndarray:
    """Extracts the pz[1] for each particle in each list."""
    return np.array(
        [[particle[6] for particle in particles if len(particle) > 8] 
            for particles in particle_lists],
        dtype=float
    )

def _calc_eta(particle: List[float]) -> np.ndarray:
    """Calculates the eta given a """
    p = np.sqrt(np.sum(particle[:3]**2))
    eta = 0.5 * np.log((p + particle[2])/(p-particle[2])) if p != abs(particle[2]) else np.inf
    return eta

def calc_eta_0(particle_lists: List[List[float]]) -> np.ndarray:
    """Calculates the eta for each particle in each list"""
    return np.array([[[_calc_eta(events[0 : 4])] for events in list] for list in particle_lists] )

def calc_eta_1(particle_lists: List[List[float]]) -> np.ndarray:
    """Calculates the eta for each particle in each list"""
    return np.array([[[_calc_eta(events[4: ])] for events in list] for list in particle_lists] )


def calc_phi_0(particle_lists: List[List[float]]) -> np.ndarray:
    """Calcualtes the phi component of the 4 momenta"""
    return np.array([[[np.arctan2(events[0], events[1])] for events in lists] for lists in particle_lists])

def calc_phi_1(particle_lists: List[List[float]]) -> np.ndarray:
    """Calcualtes the phi component of the 4 momenta"""
    return np.array([[[np.arctan2(events[4], events[5])] for events in lists] for lists in particle_lists])