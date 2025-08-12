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
            combined /= 1000
            inv_mass =  np.sqrt(max(combined[3]**2 - combined[0]**2 -combined[1]**2 - combined[2]**2,0 ))
            inv_mass *= 1_000
            l.append(inv_mass)
        invariant_masses.append(l)
    invariant_masses = np.array(invariant_masses, dtype=float)
    return invariant_masses

def calc_invariant_mass_0(particle_lists: List[List[float]]) -> List[List[float]]:
    """Calculates the invariant mass and returns as lists with each entry a distribution"""
    invariant_masses = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            # Extract (px, py, pz, E) for both particles
            p1 = np.array([pair[0], pair[1], pair[2], pair[3]])
            combined = p1
            inv_mass = combined[3]**2 - np.sum(combined[0:3]**2)
            l.append(inv_mass)
        invariant_masses.append(l)
    invariant_masses = np.array(invariant_masses, dtype=float)
    return invariant_masses

def calc_invariant_mass_1(particle_lists: List[List[float]]) -> List[List[float]]:
    """Calculates the invariant mass and returns as lists with each entry a distribution"""
    invariant_masses = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            # Extract (px, py, pz, E) for both particles
            p1 = np.array([pair[4], pair[5], pair[6], pair[7]])
            combined = p1
            inv_mass = combined[3]**2 - np.sum(combined[0:3]**2)
            l.append(inv_mass)
        invariant_masses.append(l)
    invariant_masses = np.array(invariant_masses, dtype=float)
    return invariant_masses

def calc_invariant_mass_cylindrical(particle_lists: List[List[List[float]]]) -> np.ndarray:
    """
    List 1 (index 0): each entry is [pt1, eta1, phi1, pt2, eta2, phi2] (cylindrical, massless approx).
                      Uses m^2 = 2 pT1 pT2 (cosh(Δη) - cos(Δφ)).
    List 2 (index 1): each entry is [px, py, pz, E] (Cartesian). Uses m^2 = E^2 - |p|^2.

    Any further lists are treated like list 2.
    Returns an array of per-list invariant-mass distributions.
    """
    out = []

    for i, dist in enumerate(particle_lists):
        masses = []
        for entry in dist:
            if i == 0:
                # --- First list: detector coords, massless two-body invariant mass ---
                if len(entry) < 6:
                    continue
                _,pt1, eta1, phi1, _,pt2, eta2, phi2 = entry[:8]
                d_eta = eta1 - eta2
                d_phi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]
                m2 = 2.0 * pt1 * pt2 * (np.cosh(d_eta) - np.cos(d_phi))
                masses.append(np.sqrt(m2) if m2 >= 0 else 0.0)
            else:
                # --- Second list (and beyond): native Cartesian four-vector ---
                if len(entry) < 4:
                    continue
                px, py, pz, E, px0, py0, pz0, E0 = entry[:8]
                m2 = (E + E0)**2 - (px0 + px)**2 - (py0 + py)**2 - (pz0 + pz)**2
                masses.append(np.sqrt(m2) if m2 >= 0 else 0.0)

        out.append(masses)

    return np.array(out, dtype=float)

def calc_invariant_mass_single_particle_cyl_cart(particle_lists: List[List[List[float]]]) -> np.ndarray:
    """
    Per-list SINGLE-particle invariant masses, returned as list-of-lists -> np.ndarray.

    List 0 (index 0): cylindrical single particle.
        Accepts [pt, eta, phi, X] or [*, pt, eta, phi, X].
        Uses p = pt * cosh(eta).
          - If X >= p  : treat X as energy E  -> m = sqrt(E^2 - p^2).
          - Else       : treat X as rest mass -> m = |X|.
        (phi is unused for the mass itself, kept for schema consistency.)

    List 1+ (index >= 1): Cartesian single four-vector.
        [px, py, pz, E]
        m = sqrt(E^2 - (px^2 + py^2 + pz^2)) with negatives clipped to 0.

    Returns:
        np.ndarray where out[i] is the mass distribution for list i.
    """
    out = []

    for i, dist in enumerate(particle_lists):
        masses = []
        for entry in dist:
            if i == 0:
                E, pt, eta, phi = entry[0],entry[1], entry[2], entry[3]

                p = pt * np.cosh(eta)

                m2 = E*E - p*p
                masses.append(np.sqrt(m2) if m2 >= 0 else 0.0)

            else:
                # --- Cartesian single four-vector ---
                if len(entry) < 4:
                    continue
                px, py, pz, E = entry[:4]
                m2 = E*E - (px*px + py*py + pz*pz)
                masses.append(np.sqrt(m2) if m2 >= 0 else 0.0)

        out.append(masses)

    return np.array(out, dtype=float)

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

def calc_transverse_momentum_0(particle_lists: List[List[float]]) -> List[List[float]]:
    """Calculates transverse momentum and returns as lists with each entry a distribution"""
    transverse_momentums  = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            pt1 = np.sqrt(np.square(pair[0]) + np.square(pair[1]))
            l.append(pt1)
        transverse_momentums.append(l)
    transverse_momentums = np.array(transverse_momentums, dtype = float)
    print(transverse_momentums.shape)
    return transverse_momentums

def calc_transverse_momentum_1(particle_lists: List[List[float]]) -> List[List[float]]:
    """Calculates transverse momentum and returns as lists with each entry a distribution"""
    transverse_momentums  = []
    for lists in particle_lists:
        l = []
        for pair in lists:
            if len(pair) < 8:
                continue
            pt2 = np.sqrt(np.square(pair[4]) + np.square(pair[5]))
            l.append(pt2)
        transverse_momentums.append(l)
    transverse_momentums = np.array(transverse_momentums, dtype = float)
    print(transverse_momentums.shape)
    return transverse_momentums

def calc_transverse_momentum_cyl_cart(particle_lists: List[List[List[float]]]) -> np.ndarray:
    """Calculates transverse momentum and returns as lists with each entry a distribution.
       - List 0 (cylindrical): take pt from entry[1].
       - Lists 1+ (Cartesian-style in your data): pt = sqrt(px^2 + py^2) using entry[4], entry[5].
    """
    transverse_momentums = []

    for i, lists in enumerate(particle_lists):
        l = []
        for entry in lists:
            if i == 0:
                pt = entry[1]
                l.append(float(pt))
            else:
                pt = np.sqrt(np.square(entry[0]) + np.square(entry[1]))
                l.append(float(pt))
        transverse_momentums.append(l)

    transverse_momentums = np.array(transverse_momentums, dtype=float)
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


def calc_phi_0_cyl_cart(particle_lists: List[List[List[float]]]) -> np.ndarray:
    """Calculates transverse momentum and returns as lists with each entry a distribution.
       - List 0 (cylindrical): take pt from entry[1].
       - Lists 1+ (Cartesian-style in your data): pt = sqrt(px^2 + py^2) using entry[4], entry[5].
    """
    transverse_momentums = []

    for i, lists in enumerate(particle_lists):
        l = []
        for entry in lists:
            if i == 0:
                l.append(float(entry[3]))
            else:
                l.append(np.arctan2(entry[0], entry[1]))

        transverse_momentums.append(l)

    transverse_momentums = np.array(transverse_momentums, dtype=float)
    print(transverse_momentums.shape)
    return transverse_momentums