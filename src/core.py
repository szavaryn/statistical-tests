from . import config as cfg
from .preprocesser import DataGroup, preprocess_groups
from .logic.assumptor import NormalityChecker, VarianceChecker
from .logic.calculator import CICalculator, StatCalculator
from .logic.plotter import Plotter
from .logic.tester import Tester

class SampleDescriber:
    def __init__(self, groups: list[DataGroup]):
        self.groups = preprocess_groups(groups)
        
        self.cicalc = CICalculator(self.groups)
        self.stats = StatCalculator(self.groups)
        self.norm = NormalityChecker(self.groups)
        self.plot = Plotter(self.groups)
    
    def key_stats(self, ci_resamples = cfg.CI_RESAMPLES, ci_level = cfg.CI_LEVEL):
        return self.stats.key_stats(ci_resamples = ci_resamples, ci_level = ci_level)
        
    def plot_dist(self, x_min = None, x_max = None, figsize = cfg.FIGSIZE
                  , plot_type = "kde", bins = 100, show_ci = False, resamples = cfg.CI_RESAMPLES, ci_level = cfg.CI_LEVEL):
        return self.plot.plot_dist(x_min = x_min, x_max = x_max, figsize = figsize
                  , plot_type = plot_type, bins = bins, show_ci = show_ci, resamples = resamples, ci_level = ci_level)
        
    def shapiro_wilk_test(self, p_min = cfg.ALPHA):
        return self.norm.shapiro_wilk_test(p_min = p_min)

    def standard_moments(self):
        return self.norm.standard_moments()

    def qq_plot(self, figsize = cfg.FIGSIZE):
        return self.norm.qq_plot(figsize = figsize)

class SampleComparer:
    def __init__(self, groups: list[DataGroup]):
        self.groups = preprocess_groups(groups)
        self.vars = VarianceChecker(self.groups)
        self.test = Tester(self.groups)

    def levene_test(self, center = "mean"):
        return self.vars.levene_test(center = center)
        
    def mannwhitney_test(self, alpha = cfg.ALPHA, beta = cfg.BETA, alternative = "two-sided"):
        return self.test.mannwhitney_test(alpha = alpha, beta = beta, alternative = alternative)
