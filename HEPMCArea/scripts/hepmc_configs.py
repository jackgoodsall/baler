"""
File for the configurables used for plotting in the HEPMC distribution plotter script.
"""
from dataclasses import dataclass
from typing import Tuple, List, Union, Callable, Dict, Any


@dataclass
class PlotConfiguration:
    """Standard configurables for a plot in matplotlib
    """
    # Titles and axis labels
    xlabel: str
    ylabel: str
    title: str
    range: tuple[int, int]

    legend: bool = True
  
    label: str = "Plot"
    # Figure configurables
    figsize: tuple[int, int] = (8, 6)

    # Axis configurables
    x_scale: str = "linear"
    y_scale: str = "log"


@dataclass
class HistConfigurations(PlotConfiguration):
    """Configurable specific to the histogram plot used here.
    """

    bins: int = 100
    use_range_over_bin_y_label: bool = True
    
    # Second label
    label_two: str = "Plot 2"
    
    # Hist plotting types
    hist_type: str = "step"

    ## Configs for the ratio plot on the bottom
    height_ratios: tuple[int, int] = (9, 2)
    ratio_y_label: str = "ratio"
    ratio_y_rotation: int = 0
    ratio_y_size: int = 15
    ratio_y_pad: int = 20