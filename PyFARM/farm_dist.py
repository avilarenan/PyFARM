import numpy as np
from typing import Union, Optional

# FARM distance between two vectors
def farm_dist(dyr: Union[float, np.ndarray], dyq: Optional[Union[float, np.ndarray]] = None, metric_space: bool = False) -> float:
    """
    FARM distance between two vectors. The distance is based on the angle between them.

    Parameters:
    dyr (float or np.ndarray): dy reference normalized to dx = 1
    dyq (float or np.ndarray, optional): dy query normalized to dx = 1. Defaults to None if dyr comprises query delta values.
    metric_space (bool): If True, the distance relies only on the sine function. Defaults to False.

    Returns:
    float: distance value
    """
    # Create complex numbers for vectors
    v1c = complex(0, dyr[0]) if dyq is None else complex(1, dyr)
    v2c = complex(0, dyr[1]) if dyq is None else complex(1, dyq)

    # Calculate the angles of the complex numbers
    phi_v1 = np.angle(v1c)
    phi_v2 = np.angle(v2c)

    # Calculate the absolute angle difference
    phi_d = abs(phi_v1 - phi_v2)

    if metric_space:
        # Return sin(phi) if 0 <= phi <= pi/2 else 1
        dist = np.sin(phi_d) if phi_d <= np.pi / 2 else 1
    else:
        # Adjust for contrasting vectors
        align = np.sign(phi_v1) * np.sign(phi_v2)
        if align == 0:
            align = 1  # If any phi is 0°
        # Calculate distance with combined sine and exponential approach
        dist = np.sin(phi_d) if align >= 0 else max(np.sin(phi_d), (1 - np.exp(-phi_d * 5)))

    return dist


# FARM Mean Product Quality Measure
def farm_mean_p(rts: Union[list, np.ndarray], qts: Union[list, np.ndarray]) -> float:
    """
    Mean absolute product quality measure for feature alignment.

    Parameters:
    rts (list or np.ndarray): Reference time series
    qts (list or np.ndarray): Query time series

    Returns:
    float: Quality measure (larger means better)
    """
    rts = np.array(rts)
    qts = np.array(qts)

    if len(rts) != len(qts):
        raise ValueError("Reference and query time series must have the same length.")
    return np.mean(np.abs(rts * qts))


# FARM Slope Mean Factorial Quality Measure
def farm_slm_f(rts: Union[list, np.ndarray], qts: Union[list, np.ndarray], l_orig: int = 1, l_adj: int = 1) -> float:
    """
    Slope-mean-factorial quality measure for feature alignment.

    Parameters:
    rts (list or np.ndarray): Reference time series
    qts (list or np.ndarray): Query time series
    l_orig (int): Length of the original time series
    l_adj (int): Length of the adjusted time series

    Returns:
    float: Quality measure (larger means better)
    """
    rts = np.array(rts)
    qts = np.array(qts)

    if len(rts) != len(qts):
        raise ValueError("Reference and query time series must have the same length.")

    # Calculate the product of the differences (slopes)
    slope_product = np.sum(np.diff(rts) * np.diff(qts), where=(~np.isnan(np.diff(rts))) & (~np.isnan(np.diff(qts))), axis=0)

    # Return the normalized sum of slopes
    return (1 / len(rts)) * slope_product / (1 + abs((l_orig - l_adj) / l_orig))


# Example usage
if __name__ == "__main__":
    # Example data
    ref_ts = [1, 2, 1.5, 2.1, 5, 2.32, 1, 0.2]
    qry_ts = [0.2, 0.4, 0.4, 0.5, 1, 0.92, 0.5, 0.2]

    # Calculate the FARM distance
    dist = farm_dist([0.23, -2.85])
    print(f"FARM Distance: {dist}")

    # Calculate mean product quality
    mean_p = farm_mean_p(ref_ts, qry_ts)
    print(f"Mean Product Quality: {mean_p}")

    # Calculate slope mean factorial quality measure
    slm_f = farm_slm_f(ref_ts, qry_ts)
    print(f"Slope Mean Factorial Quality: {slm_f}")
