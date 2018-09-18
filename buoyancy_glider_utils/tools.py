#!/usr/bin/env python
from __future__ import print_function
import numpy as _np
import pandas as _pd


def smooth(y, window_len=11, window_func=_np.ones):
    """
    Smooth the pd.Series using a window with requested size.
    The function is taken from the CTD package.
    This function is adapted from:
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """

    y = _np.array(y).copy()
    wl = window_len
    s = _np.r_[2 * y[0] - y[wl:1:-1], y, 2 * y[-1] - y[-1:-wl:-1]]

    w = window_func(window_len)

    y = _np.convolve(w / w.sum(), s, mode='same')
    y = y[wl - 1:-wl + 1]
    return y


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = linspace(-4, 4, 500)
    y = exp( -t**2 ) + random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial
    from numpy import array
    y = array(y)

    try:
        window_size = _np.abs(int(window_size))
        order = _np.abs(int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomial order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = _np.mat([[k**i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = _np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = _np.concatenate((firstvals, y, lastvals))

    return _np.convolve(m[::-1], y, mode='valid')


def bin_depths(y, dives, depth, bins=None, how='mean', interp_lim=6, as_xarray=False):
    """
    This function bins the variable to set depths.
    If no depth is specified it defaults to:
        start : step : end
        ------------------------
           0  :  1.  : maxdepth
        ------------------------
    This is the sampling method typically used by the CSIR.
    This can be easily changed by specifying an array of increasing values.

    The function used to bin the data can also be set, where the default
    is the mean. This can be changed to 'median', 'std', 'count', etc...

    Lastly, the data is stored as SeaGliderVariable.gridded for future
    easy access. If you would like to regrid the data to another grid
    use SeaGliderVariable.bin_depths().
    """
    from pandas import cut
    from numpy import arange, array, c_, nanmax

    y = array(y)
    depth = array(depth)
    dives = array(dives)

    bins = arange(nanmax(depth) + 1) if bins is None else bins

    labels = c_[bins[:-1], bins[1:]].mean(axis=1)
    bins = cut(depth, bins, labels=labels)

    grp = _pd.Series(y).groupby([dives, bins])
    grp_agg = getattr(grp, how)()
    gridded = grp_agg.unstack(level=0)
    gridded = gridded.reindex(labels.astype(float))
    gridded.index.name = 'depth'
    gridded.columns.name = 'dives'

    if interp_lim > 0:
        gridded = gridded.interpolate(limit=interp_lim).bfill(limit=interp_lim)

    # gridded = gridded.reindex(index=labels)

    if as_xarray:
        return gridded.stack().to_xarray()
    return gridded


def neighbourhood_iqr(dives, depth, var, dives_window=40, depth_window=50, iqr_multiplier=2.5):
    """
    A filter that filters in a neighbourhood of dives (#) and depth (m).
    The size of the neighbourhood is set with kwargs.
    The multiplier for the iqr is also set with the kwarg.

    Returns a mask containing the outliers.
    """

    from numpy import arange, array, nanpercentile, nanmin, nanmax
    from itertools import product as ittprod

    def iqr_lims(y, iqr_multiplier=2):
        q1, q3 = nanpercentile(y, [25, 75])
        iqr = (q3 - q1) * iqr_multiplier
        iqr_lower = q1 - iqr
        iqr_upper = q3 + iqr
        return iqr_lower, iqr_upper

    x = array(dives)
    y = array(depth)
    z = array(var)

    x_window = dives_window
    y_window = depth_window

    xr = arange(nanmin(x), nanmax(x), x_window)
    yr = arange(nanmin(y), nanmax(y), y_window)

    index_windows = ittprod(xr, yr)

    outliers = []
    for c, (xi, yi) in enumerate(index_windows):
        x0 = xi - x_window
        x1 = xi + x_window

        y0 = yi - y_window
        y1 = yi + y_window

        i = (y > y0) & (y < y1) & (x > x0) & (x < x1)
        if i.sum() == 0:
            break

        l0, l1 = iqr_lims(z[i], iqr_multiplier)

        outliers += i & ((z < l0) | (z > l1)),

    outliers = array(outliers).sum(0)
    if outliers.max() > 0:
        return outliers >= outliers.max()
    else:
        return outliers


def rolling_window(y, func, window):
    """
    A rolling window function that is nan-resiliant
    It applies a given aggregating function to the window.
    """
    from numpy import ndarray, array, nan, r_
    n = window
    y_mat = ndarray([n, len(y) - n]) * nan
    for i in range(n):
        y_mat[i, :] = y[i: i - n]
    y_filt = func(y_mat, axis=0)

    i0 = n // 2
    i1 = n - i0
    seg0 = array([func(y[: i + 1]) for i in range(i0)])
    seg1 = array([func(y[-i - 1:]) for i in range(i1)])
    y = r_[seg0, y_filt, seg1]

    return y


def calc_dive_time_avg(time, dives, time_units='seconds since 1970-01-01 00:00:00'):
    from pandas import Series
    from numpy import datetime64, array
    from xarray.coding.times import decode_cf_datetime

    time = array(time)
    if isinstance(time[0], datetime64):
        t = time.astype(float) / 1e9
    else:
        t = time

    t_num = Series(t).groupby(dives).mean()
    t_d64 = decode_cf_datetime(t_num, time_units)
    t_ser = Series(t_d64, index=t_num.index.values)
    t_ful = t_ser.reindex(index=dives).values

    return t_ful


def mask_to_depth(var, depth, dives):
    i = _np.r_[False, _np.diff(var)].astype(bool)
    idx_depth = _pd.Series(_np.array(depth)[i], index=_np.array(dives)[i])

    return idx_depth


if __name__ == '__main__':

    pass
