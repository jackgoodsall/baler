"""
File for the classes and config class for the HEPMC distribution plotter script
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any

import numpy as np

@dataclass
class CutConfigurations:
    """Dataclass to store configurables of Cuts, key is a callable cut function defined as a 
    EventCuts polymorphism and the value is a dictionary of the parameters for the specific 
    cut (upper bound, lower_bound"""
    cuts: Dict[Callable, Dict[str, Any]] = field(default_factory = dict)

    def add_cut(self, cut_func: Callable, **params):
        self.cuts[cut_func] = params
    

class EventCuts:
    def __init__(self, **kwargs):
        self.cut_function = self.make_cuts
        self.cut_indexes = None
        self.cut_config = kwargs["cut_config"] 
        
    def fit_cuts(self, events):
        mask = np.zeros(len(events), dtype=bool)
        for cut_func, params in self.cut_config.cuts.items():
            cutter = cut_func(**params)  
            mask |= cutter.make_cuts(events) 
        self.cut_indexes = mask
    
    def make_cuts(self, events):
        return events[:, ~self.cut_indexes]
    

class TransverseMomentumCut(EventCuts):
    """Class for TransverseMomentumCut that inherits from EventCuts"""
    def __init__(self, **cut_paramaters):
        """Set up init with default paramaters, all other paramaters are ignored"""
        self.cut_indexes = None
        self.pt0_lower_bound = cut_paramaters.get("pt0_lower", 0)
        self.pt0_upper_bound = cut_paramaters.get("pt0_upper", np.inf)
        self.pt1_lower_bound = cut_paramaters.get("pt1_lower", 0)
        self.pt1_upper_bound = cut_paramaters.get("pt1_upper", np.inf)

    def make_cuts(self, events):
        ## Make cuts, vectorise momentum_cuts 
        cut_func = np.vectorize(self._momentum_cuts, signature='(n)->()')
        # Save indexes and return them
        self.cut_indexes = cut_func(events).astype(bool)
        return self.cut_indexes
    
    def _momentum_cuts(self, event):
        pt0, pt1 = self._calculate_transverse_momentum(event)
        if pt0 < self.pt0_lower_bound or pt0 > self.pt0_upper_bound:
            return 1
        if pt1 < self.pt1_lower_bound or pt1 > self.pt1_upper_bound:
            return 1
        return 0

    def _calculate_transverse_momentum(self, event):
        pt0 = np.sqrt(np.sum(event[0:2] **2 ))
        pt1 = np.sqrt(np.sum(event[4:6] **2 ))
        return pt0, pt1


class EtaCut(EventCuts):
    def __init__(self, **cut_paramaters):
        self.cut_indexes = None
        self.eta_min = cut_paramaters.get("eta_min", 1.37)
        self.eta_max = cut_paramaters.get("eta_max", 1.52)
    def make_cuts(self, events):
        cut_func = np.vectorize(self._cut_eta, signature="(n)->()")
        self.cut_indexes = cut_func(events).astype(bool)
        return self.cut_indexes
    def _cut_eta(self, event):
        eta0, eta1 = self._calculate_eta(event)

        if abs(eta0) > self.eta_min and abs(eta0) < self.eta_max:
            return 1
        if abs(eta1) > self.eta_min and abs(eta1) < self.eta_max:
            return 1
        return 0
       

    def _calculate_eta(self, event):
        px0, py0, pz0 = event[0], event[1], event[2]
        p0 = np.sqrt(px0**2 + py0**2 + pz0**2)
        eta0 = 0.5 * np.log((p0 + pz0) / (p0 - pz0)) if p0 != abs(pz0) else np.sign(pz0) * np.inf

        # Extract momentum components (px, py, pz) for particle 2
        px1, py1, pz1 = event[4], event[5], event[6]
        p1 = np.sqrt(px1**2 + py1**2 + pz1**2)
        eta1 = 0.5 * np.log((p1 + pz1) / (p1 - pz1)) if p1 != abs(pz1) else np.sign(pz1) * np.inf
        return eta0, eta1
    