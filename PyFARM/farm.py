import numpy as np
import pandas as pd
from scipy.stats import zscore
from .farm_dist import farm_dist, farm_mean_p, farm_slm_f
from .tw_decomp import tw_decomp

def farm(
    refTS: np.ndarray, 
    qryTS: np.ndarray, 
    lcwin: int = 5, 
    rel_th: int = 15, 
    ff_align: bool = True, 
    reshape_fnc: callable = None, 
    fuzzyc: np.ndarray = None, 
    metric_space: bool = True, 
    reshape_mode: str = 'uni'
) -> dict:
    """
    Align and compare two time series using feature alignment, time series decomposition, 
    and various quality measures.

    Parameters:
    - refTS (np.ndarray): The reference time series.
    - qryTS (np.ndarray): The query time series.
    - lcwin (int, optional): Local correlation window size, must be odd (default is 5).
    - rel_th (int, optional): Relevance threshold for global relevance calculation (default is 15).
    - ff_align (bool, optional): Whether to perform feature alignment (default is True).
    - reshape_fnc (callable, optional): Function to reshape the time series (default is None).
    - fuzzyc (np.ndarray, optional): Coefficients for fuzzification (default is None).
    - metric_space (bool, optional): Whether to use metric space for distance calculation (default is True).
    - reshape_mode (str, optional): Mode for reshaping the time series, either 'uni' or 'dual' (default is 'uni').

    Returns:
    - dict: A dictionary containing the following keys:
        - 'path': DataFrame representing the warped path.
        - 'rts_decomp': Decomposed reference time series.
        - 'qts_decomp': Decomposed query time series.
        - 'qts_shaped': Shaped query time series after applying local relevance fuzzification.
        - 'rel_local': Local relevance scores.
        - 'rel_local_end': local relevance values (not fuzzyfied) with correlation coefficient spot to the end of the lcwin.
        - 'rel_local_fuzz': Fuzzified local relevance scores.
        - 'rel_global': Global relevance score.
        - 'qmeas_mean_p': Quality measure based on the mean product of time series.
        - 'qmeas_slm_f': Quality measure based on the slope-mean factorial of time series.
    """
    if reshape_fnc is None:
        reshape_fnc = lambda x: x
    if fuzzyc is None:
        fuzzyc = np.array([2, 6, 10, 6, 2]) / 10

    # Delta of time series
    dref = np.diff(refTS)
    dqry = np.diff(qryTS)
    lref = len(dref)
    lqry = len(dqry)

    # Initialize the warped path
    path = pd.DataFrame({'y': [0], 'x': [0]})
    path_last = path.iloc[-1]
    dist_root = farm_dist(dref[0], dqry[0], metric_space)

    # Ensure conditions
    assert lcwin <= min(len(refTS), len(qryTS)), 'Local correlation window exceeds time series length.'
    assert isinstance(refTS, np.ndarray), 'refTS must be of type np.array.'
    assert isinstance(qryTS, np.ndarray), 'qryTS must be of type np.array.'
    assert lcwin % 2 == 1, 'Local correlation window must be odd.'
    assert 1 <= rel_th <= 100, 'Relevance threshold must be between 1 and 100.'
    assert reshape_mode in ['uni', 'dual'], "Reshape mode must be either 'uni' or 'dual'."
    assert len(fuzzyc) % 2 == 1, "Fuzzyc must have an odd number of coefficients."
    assert len(fuzzyc) < 2 * min(len(refTS), len(qryTS)), "Number of fuzzy coefficients must be < 2 * min(len(refTS), len(qryTS))."

    missal_y = 0

    # Feature alignment (if enabled)
    if ff_align:
        while path_last['x'] + 1 < lqry or path_last['y'] + 1 < lref:
            # Calculate diagonal, dx+2y+1 and dx+1y+2 distances
            dist_xy = farm_dist(dref[path_last['y'] + 1], dqry[path_last['x'] + 1], metric_space) if path_last['y'] + 1 < lref and path_last['x'] + 1 < lqry else np.nan
            dist_xyy = farm_dist(dref[path_last['y'] + 2], dqry[path_last['x'] + 1], metric_space) + missal_y if path_last['y'] + 2 < lref and path_last['x'] + 1 < lqry else np.nan
            dist_xxy = (np.inf if path_last['y'] == path_last['x'] else farm_dist(dref[path_last['y'] + 1], dqry[path_last['x'] + 2], metric_space)) if (path_last['y'] + 1 < lref and path_last[
            'x'] + 2 < lqry) else np.nan

            # dist_xy = farm_dist(dref[path_last['y']], dqry[path_last['x']], metric_space) if path_last['y'] + 1 <= lref and path_last['x'] + 1 <= lqry else np.nan
            # dist_xyy = farm_dist(dref[path_last['y'] + 1], dqry[path_last['x']], metric_space) + missal_y if path_last['y'] + 2 <= lref and path_last['x'] + 1 <= lqry else np.nan
            # dist_xxy = farm_dist(dref[path_last['y']], dqry[path_last['x'] + 1], metric_space) if path_last['y'] + 1 <= lref and path_last['x'] + 2 <= lqry and path_last['y'] != path_last['x'] else np.nan

            # Choose the next step
            next_step = {
                0: {'root': dist_xy + dist_root, 'path': pd.DataFrame({'y': [path_last['y'] + 1], 'x': [path_last['x'] + 1]})},
                1: {'root': dist_xxy + dist_root, 'path': pd.DataFrame({'y': [path_last['y'], path_last['y'] + 1], 'x': [path_last['x'] + 1, path_last['x'] + 2]})},
                2: {'root': dist_xyy + dist_root, 'path': pd.DataFrame({'y': [path_last['y'] + 1, path_last['y'] + 2], 'x': [path_last['x'], path_last['x'] + 1]})},
                3: {'root': dist_root, 'path': pd.DataFrame({'y': [path_last['y'] + 1] if path_last['y'] < lref -1 else [path_last['y']],
                                                             'x': [path_last['x'] + 1] if path_last['x'] < lqry -1 else [path_last['x']]})},
                # 3: {'root': dist_root, 'path': pd.DataFrame({'y': [min(path_last['y'] + 1, lref)], 'x': [min(path_last['x'] + 1, lqry)]})}
            }

            # Select path with minimum distance
            min_index = np.nanargmin([dist_xy, dist_xxy, dist_xyy, 1])
            next_step = next_step[min_index]
            dist_root = next_step['root']
            path_last = next_step['path'].iloc[-1]
            path = pd.concat([path, next_step['path']], ignore_index=True)

        # Time series decomposition
        ts_dec = tw_decomp(refTS, qryTS, path, reshape_mode)
    else:
        ts_dec = {'rts': refTS, 'qts': qryTS}

    # Quality measure calculations
    qmeas_mean_p = farm_mean_p(ts_dec['rts'], ts_dec['qts'])
    qmeas_slm_f = farm_slm_f(ts_dec['rts'], ts_dec['qts'], len(refTS), len(ts_dec['qts']))

    # Local relevance calculation
    rel_rts = zscore(ts_dec['rts'])
    rel_qts = zscore(ts_dec['qts'])

    # Padding at the ends to ensure equal length for correlation
    rel_rts = np.concatenate([np.zeros(lcwin -1), rel_rts, np.zeros(lcwin // 2)])
    rel_qts = np.concatenate([np.zeros(lcwin -1), rel_qts, np.zeros(lcwin // 2)])

    # calculate farm.dist for all element pairings (used for last spot correlation weighting)
    w_rq_farm_dist = np.array([farm_dist(rt, qt) for rt, qt in zip(rel_rts, rel_qts)])

    # Create rolling windows for local correlation
    rel_rts = pd.DataFrame(np.lib.stride_tricks.sliding_window_view(rel_rts, lcwin))
    rel_qts = pd.DataFrame(np.lib.stride_tricks.sliding_window_view(rel_qts, lcwin))
    w_rq_farm_dist < - pd.DataFrame(np.lib.stride_tricks.sliding_window_view(w_rq_farm_dist, lcwin))

    # Calculate local correlation
    rel_local = np.array([1 - np.sqrt(0.5 * (1 - np.corrcoef(x, y)[0, 1])) for x, y in zip(rel_rts.values, rel_qts.values)])
    rel_local[np.isnan(rel_local)] = 0

    # Assuming rel_rts is a numpy array
    sum_weights = np.linspace(0, 1, num=rel_rts.shape[1]) ** 3
    sum_weights = sum_weights / np.sum(sum_weights)

    # get sum of weighted farm similarity for all windows
    w_rq_farm_dist = np.array(np.sum((1 - w_rq_farm_dist) * (np.ones((w_rq_farm_dist.shape[0], 1)) @ sum_weights.reshape(1, -1)), axis=1))

    # weight local relevance for last spot and
    # cut tail by floor(win/2) to set spot to the end of sliding window
    rel_local_end = (rel_local * w_rq_farm_dist)[:-np.floor(lcwin / 2).astype(int)]

    # Cut head by floor(win/2) to set spot centric of sliding window
    rel_local = rel_local[np.floor(lcwin / 2).astype(int):]

    # Fuzzification
    fuzzy_l = len(fuzzyc)
    rel_local_fuzz = np.concatenate([np.zeros(fuzzy_l // 2), rel_local, np.zeros(fuzzy_l // 2)])
    rel_local_fuzz = np.dot(np.lib.stride_tricks.sliding_window_view(rel_local_fuzz, fuzzy_l)[:, ::-1], fuzzyc)
    rel_local_fuzz /= np.sum(fuzzyc)

    # Global relevance calculation
    rel_global = np.sqrt((np.mean(np.sort(rel_local_fuzz)[-int(rel_th / 100 * len(rel_local_fuzz)):]) ** 2 + np.mean(rel_local_fuzz) ** 2) / 2)

    # Shaped query time series
    qts_shaped = ts_dec['qts'] * np.array([reshape_fnc(x) for x in rel_local_fuzz])

    return {
        'path': path,
        'rts_decomp': ts_dec['rts'],
        'qts_decomp': ts_dec['qts'],
        'qts_shaped': qts_shaped,
        'rel_local': rel_local,
        'rel_local_end': rel_local_end,
        'rel_local_fuzz': rel_local_fuzz,
        'rel_global': rel_global,
        'qmeas_mean_p': qmeas_mean_p,
        'qmeas_slm_f': qmeas_slm_f
    }
