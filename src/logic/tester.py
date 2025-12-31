import numpy as np
from scipy.stats import norm, mannwhitneyu
from ..preprocesser import DataGroup

class Tester:
    """
    Performing statistical tests
    """
    def __init__(self, groups: list[DataGroup]):
        self.groups = groups
        self.samples = [group.data for group in self.groups]
    
    def mannwhitney_test(self, alpha, beta, alternative = "two-sided"):
        if len(self.samples) != 2:
            raise ValueError("Mann-Whitney U test requires two samples.")
        H0 = "The two samples come from the same distribution. Equal medians if samples have the same shape (variance, etc.)"
        H1 = "Distributions differ. One of groups tends to have greater / less value than another."
        results = {}
        n1 = len(self.samples[0])
        n2 = len(self.samples[1])

        mean_diff = np.mean(self.samples[1]) - np.mean(self.samples[0])
        median_diff = np.median(self.samples[1]) - np.median(self.samples[0])
        
        U, p = mannwhitneyu(*self.samples, alternative = alternative)
        U_exp = n1 * n2 / 2
        var_u = n1 * n2 * (n1 + n2 + 1) / 12
        
        z = (U - U_exp) / np.sqrt(var_u)
        z_cr = norm.ppf(1 - alpha / 2) if alternative == "two-sided" else norm.ppf(1 - alpha)
        p_val = 2 * (1 - norm.cdf(abs(z))) if alternative == "two-sided" else 1 - norm.cdf(abs(z))
        power = norm.cdf(abs(z - z_cr)) + 1 - norm.cdf(abs(z + z_cr))
        
        r = z / np.sqrt(n1 + n2)
        vda = U / (n1 * n2)
        rg = 1 - ((2 * U) / (n1 * n2) )

        res = f"Reject H0" if p_val < alpha else f"Fail to reject H0"
        
        results["required"] = {
            "H0": None
            , "H1": None
            , "mean_diff": None
            , "median_diff": None
            , "U_obs": None
            , "U_exp": None
            , "Variance U": None           
            , "Z": f"> |{abs(z_cr):.3f}|"
            , "p": f"< {alpha}"
            , "power": f"> {1 - beta}"
            , "r": "small: > 0.1, medium: > 0.3, large > 0.5"
            , "vda": "small: > 0.56, medium: > 0.64, large > 0.71"
            , "rg": "small: > 0.11, medium: > 0.28, large > 0.43"
            , "res": None
        }
        
        results["observed"] = {
            "H0": H0
            , "H1": H1
            , "mean_diff": float(mean_diff)
            , "median_diff": float(median_diff)
            , "U_obs": float(U)
            , "U_exp": float(U_exp)
            , "Variance U": float(var_u)
            , "Z": float(z)
            , "p": float(p_val)
            , "power": float(power)
            , "r": float(r)
            , "vda": float(vda)
            , "rg": float(abs(rg))
            , "res": res
        }
        return results
