import numpy as np
from scipy.stats import bootstrap
from ..preprocesser import DataGroup

class CICalculator:
    """
    Class for calculating confident intervals for mean values of not normally distributed data
    """
    def __init__(self, groups: list[DataGroup]):
        self.groups = groups
        #as it could be called several times inside different classes, i'd like to cache it
        self._cache: dict[tuple, dict] = {}

    def bootstrap_ci(
        self
        , resamples: int
        , ci_level: float
        , method: str = "percentile"
        , random_state: int = 42
    ) -> dict[str, dict[str, float]]:
        """
        Method for using resampling of existing sample
        """
        #returning values of ci_intervals if it was calculated previously
        key = (resamples, ci_level, method, random_state)
        if key in self._cache:
            return self._cache[key]
            
        #calculate for the first time with corresponding input
        results = {}
        for group in self.groups:    
            ci = bootstrap((group.data,), np.mean, confidence_level = ci_level
                           , n_resamples = resamples, method = method, random_state = random_state)
            results[group.label] = {
                "ci_left": float(ci.confidence_interval[0])
                , "ci_right": float(ci.confidence_interval[1])
            }
        self._cache[key] = results
        return results

class StatCalculator:
    """
    Calculating sample stats
    """
    def __init__(self, groups: list[DataGroup]):
        self.groups = groups
        self.cicalc = CICalculator(groups)

    def key_stats(self, ci_resamples, ci_level):
        ci = self.cicalc.bootstrap_ci(resamples = ci_resamples, ci_level = ci_level)
        results = {}
        for group in self.groups:
            #sample size
            size = len(group.data)
            #sample amount
            amount = float(np.sum(group.data))
            #sample median
            median = float(np.median(group.data))
            #sample mean
            mean = float(np.mean(group.data))
            #confidence intervals
            ci_ints = (ci[group.label]["ci_left"], ci[group.label]["ci_right"])
            #sample variance
            var = float(np.var(group.data))
            #sample standard deviation
            std = float(np.std(group.data))
            #sample standard error
            ste = float(std / np.sqrt(size))
            results[group.label] = {
                "Size": size
                , "Total amount": amount
                , "Mean": mean
                , "Confidence intervals": ci_ints
                , "Median": median
                , "Variation": var
                , "Standard deviation": std
                , "Standart error": ste
            }
        return results
