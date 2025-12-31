import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ..preprocesser import DataGroup
from .calculator import CICalculator

class Plotter:
    """
    Class for plotting distribution
    """
    def __init__(self, groups: list[DataGroup]):
        self.groups = groups
        self.cicalc = CICalculator(groups)
        
    def plot_dist(self, figsize, x_min = None, x_max = None
                  , plot_type = "kde", bins = 100, show_ci = False, resamples = None, ci_level = None):
        """
        Method for plotting distribution (bar, kde)
        """
        plt.figure(figsize = figsize)
        #min value of x axis - would be either specified in input or minimal from data
        x_lim = []
        labels = []
        if show_ci == True:
            ci = self.cicalc.bootstrap_ci(resamples = resamples, ci_level = ci_level)
        
        for group in self.groups:
            mean = float(np.mean(group.data))
            median = float(np.median(group.data))
            # vertical mean line
            plt.axvline(mean, label = f'{group.label} mean: {mean:.2f}', color = group.color, linestyle = '--', linewidth = 3)
            # vertical median line
            plt.axvline(median, label = f'{group.label} median: {median:.2f}', color = group.color, linestyle = ':', linewidth = 3)            
            x_lim.append(np.min(group.data))
            labels.append(group.label)
            if show_ci == True:
                plt.axvline(ci[group.label]["ci_left"]
                            , label = f'{group.label} ci left: {ci[group.label]["ci_left"]:.2f}'
                            , color = group.color, linestyle = ':', linewidth = 2)
                plt.axvline(ci[group.label]["ci_right"]
                            , label = f'{group.label} ci righ: {ci[group.label]["ci_right"]:.2f}'
                            , color = group.color, linestyle = ':', linewidth = 2)

            if plot_type == "kde":
                sns.kdeplot(group.data, label = group.label
                            , color = group.color, bw_adjust = 1.0)
            elif plot_type == "bar":
                plt.hist(group.data, label = group.label
                         , color = group.color, edgecolor = "black"
                         , density = True, alpha = 0.5, bins = bins)
            else:
                raise ValueError("plot_type must be 'kde' or 'bar'")                

        #fig attributes    
        if x_min is not None:
            plt.xlim(left = x_min)
        else:
            plt.xlim(left = min(x_lim))
        if x_max is not None:
            plt.xlim(right = x_max)
        
        plt.title(f"Distribution of {labels} - {plot_type}")
        plt.xlabel("Metric Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()
