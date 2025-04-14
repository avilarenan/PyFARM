import numpy as np
import pandas as pd

def tw_decomp(rts: np.ndarray, qts: np.ndarray, twp: pd.DataFrame, mode: str = 'uni', abs: bool = False) -> dict:
    """
    Time series decomposition based on a time warping path.

    This function performs time series decomposition using a time warping path (twp), 
    either adjusting only the query time series ('uni') or both the query and reference time series ('dual').

    Parameters:
    -----------
    rts : np.ndarray
        Reference time series. Must be a numpy array of shape (n,).
        
    qts : np.ndarray
        Query time series. Must be a numpy array of shape (n,).
        
    twp : pd.DataFrame
        Time warping path, typically generated from dynamic time warping. The DataFrame should have 
        two columns 'x' and 'y' representing the indices in the query and reference time series.
        
    mode : str, optional, default: 'uni'
        The mode of decomposition. 'uni' adjusts only the query time series, 
        'dual' adjusts both the query and reference time series.
        
    abs : bool, optional, default: False
        If True, the warping will rely on absolute values (DTW-style). If False, 
        it will rely on differences between data points for interpolation.

    Returns:
    --------
    dict
        A dictionary with the following keys:
        - 'rts': Decomposed reference time series.
        - 'qts': Decomposed query time series.

    Raises:
    -------
    ValueError
        If any of the input conditions are violated, such as mode being invalid, 
        or series lengths not aligning with the warping path.
    """
    # Validate inputs
    if mode not in ['uni', 'dual']:
        raise ValueError("Mode must be either 'uni' or 'dual'")
    
    if abs:
        if not (max(len(rts), len(qts)) > max(twp['y'].max(), twp['x'].max())):
            raise ValueError('If the largest indices equals the number of series data points, only de-warping on absolute values is possible.')
    else:
        if not (len(rts) >= twp['y'].max() and len(qts) >= twp['x'].max()):
            raise ValueError('Length of series must be equal for abs=True or +1 the largest index of de-warping indices.')

    if abs:  # Warping relying on data points like in DTW
        if mode == 'uni':  # Adjust query time series only
            dec_qts = qts[twp['x'].values]  # Insert samples in query ts
            # Deletions in query ts by averaging
            dec_qts = np.array([np.mean(dec_qts[twp['y'] == y]) for y in np.unique(twp['y'].values)])
            dec_rts = rts  # Keep reference ts unchanged
        elif mode == 'dual':  # Adjust query and reference time series
            dec_qts = qts[twp['x'].values]  # Apply query warping
            dec_rts = rts[twp['y'].values]  # Apply reference warping
    else:  # Warping relying on differences between data points
        # Insert samples in query ts (linear interpolation)
        dec_qts = [qts[0]] + [np.linspace(qts[int(x)], qts[int(x)+1], num=len(twp[twp['x'] == x]))[1:] for x in np.unique(twp['x'].values)]
        dec_qts = np.concatenate(dec_qts)
        
        if mode == 'uni':  # Adjust query time series only
            dec_qts = dec_qts[~np.concatenate(([True], np.diff(twp['y'].values) == 0))]  # Remove duplicates in query ts
            dec_rts = rts  # Keep reference ts untouched
        elif mode == 'dual':  # Adjust query and reference time series
            # Insert samples in reference ts (linear interpolation)
            dec_rts = [rts[0]] + [np.linspace(rts[int(y)], rts[int(y)+1], num=len(twp[twp['y'] == y]))[1:] for y in np.unique(twp['y'].values)]
            dec_rts = np.concatenate(dec_rts)

    return {'rts': dec_rts, 'qts': dec_qts}
