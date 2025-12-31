import numpy as np
from dataclasses import dataclass
from itertools import cycle
from . import config as cfg

#frozen == True for protection from smth like group.label = "B"
#but i'd like to manually assign something
@dataclass(frozen = False)
class DataGroup:
    """
    Structure for all listed classes
    """
    data: np.ndarray
    label: str | None = None
    color: str | None = None

def preprocess_groups(groups: list[DataGroup]) -> list[DataGroup]:
    """
    UDF for providing standardized preprocessing of DataGroup inputs
    """
    if not groups:
        raise ValueError("At least one group is required.")
    #iterating colors - next is used if not specified at input
    color_cycle = cycle(cfg.DEFAULT_COLORS)
    #output
    preprocessed = []
    for i, group in enumerate(groups, start = 1):
        #data -> drop null values, convert to np.array
        arr = np.asarray(group.data, dtype = float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            raise ValueError(f"{group.label or f"sample_{i}"} contains no valid data.")

        preprocessed.append(
            DataGroup(
                data = arr
                , label = group.label or f"sample_{i}"
                , color = group.color or next(color_cycle)
            )
        )
    return preprocessed