import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene, shapiro, probplot
from ..preprocesser import DataGroup

class VarianceChecker:
    """
    Checking equality of variances    
    """
    def __init__(self, groups: list[DataGroup]):
        self.groups = groups
        self.samples = [group.data for group in self.groups]

    def levene_test(self, center = "mean"):
        if len(self.samples) < 2:
            raise ValueError("At least two samples are required.")
        H0 = "All groups have equal variances."
        H1 = "At least one group has a different variance."
        results = {}
        stat, p = levene(*self.samples, center = center)
        res = f"Reject H0" if p < 0.05 else f"Fail to reject H0"
        results = {
            "H0": H0
            , "H1": H1
            , "p": p
            , "res": res
        }
        return results

class NormalityChecker:
    """
    Checking normality of distribution
    """
    def __init__(self, groups: list[DataGroup]):
        self.groups = groups

    def shapiro_wilk_test(
        self
        , p_min: float
    ):
        results = {}
        H0 = "The sample data is normally distributed."
        H1 = "The sample data doesn't have normal distribution"
        for group in self.groups:
            w, p = shapiro(group.data)
            res = f"{group.label}: reject H0" if p < p_min else f"{group.label}: fail to reject H0."
            results[group.label] = {
                "H0": H0
                , "H1": H1
                , "w": float(w)
                , "p": float(p)
                , "res": res
            }
        return results
            
    def standard_moments(self):
        results = {}
        for group in self.groups:
            n = len(group.data)
            mean = np.mean(group.data)
            s = np.std(group.data, ddof=1)
            z = (group.data - mean) / s
            #3rd standard moment - skewness
            g1 = float(np.mean(z**3))
            Zg1 = float(g1 / (np.sqrt(6/n)))
            #4th standard moment - kurtosis
            g2 = float(np.mean(z**4) - 3)
            Zg2 = float(g2 / (np.sqrt(24/n)))
            #D'Agostino-Pearson omnibus test
            X2 = Zg1**2 + Zg2**2
            sk_res = f"{group.label} data has no skewness" if (g1>-0.5 and g1<0.5) else f"{group.label} data has skewness"
            kur_res = f"{group.label} data has no kurtosis" if (g2>-2 and g2<2) else f"{group.label} data has kurtosis"
            res = f"{group.label} data is normally distributed" if X2 < 6 else f"{group.label} data is NOT normally distributed"
            desc = f"""
{sk_res} - as g1 should be in (-0.5:0.5) under normality assumption, while g1 = {g1:.3f}
{kur_res} - as g2 should be in (-2:2) under normality assumption, while g2 = {g2:.3f}
According to D'Agostino-Pearson test if X2 = Zg1^2 + Zg2^2 > 6, we reject the null hypothesis of normality (as χ²cdf(2, 5.991464546) = 0.95), while X2 = {X2:.3f}
"""
            results[group.label] = {
                "g1": g1
                , "Zg1": Zg1
                , "g2": g2
                , "Zg2": Zg2
                , "X2": X2
                , "res": res
                , "desc": desc
            }
        results["required"] = {
                "g1": "in (-0.5:0.5)"
                , "Zg1": None
                , "g2": "in (-2:2)"
                , "Zg2": None
                , "X2": "<6"
                , "res": None
                , "desc": None            
        }
        return results
        
    def qq_plot(self, figsize):
        """
        Q-Q plot
        """
        for group in self.groups:
            fig, ax = plt.subplots(figsize = figsize)
            (osm, osr), (slope, intercept, r) = probplot(group.data, dist = "norm", fit = True)
            n = len(osr)
            ranks = np.argsort(np.argsort(osr))
            colors = ranks / (n - 1) if n > 1 else np.zeros_like(ranks)
            sc = ax.scatter(
                osm, osr, c = colors, cmap = "viridis", s = 12,
                marker = "o", edgecolor = "k", linewidths = 0.4, alpha = 0.9
            )
            fig.colorbar(sc, ax = ax).set_label("Quantile rank (low → high)")
            
            x_line = np.array([osm.min(), osm.max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linestyle = "--", color = "C2",linewidth = 1.2, zorder = 2)
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel("Sample Quantiles")
            ax.set_title(f'Q-Q plot: {group.label}')
            ax.grid(True, linestyle = ":", linewidth = 0.6, alpha = 0.7)
            fig.show()
