#!/usr/bin/env python
"""
@package seaglider.optics
@file seaglider/optics.py
@author Luke Gregor
@brief Module containing the processing functions for optics sensors (seaglider or slocum)
"""
from __future__ import print_function
from .tools import rolling_window, mask_to_depth


def find_bad_profiles(dives, depth, var, ref_depth=200):
    from numpy import array, c_
    from pandas import DataFrame

    dives = array(dives)

    df = DataFrame(c_[depth, dives, var], columns=['depth', 'dives', 'dat'])

    if not ref_depth:
        # reference depth is found by finding the average maximum
        # depth of the variable. The max depth is multiplied by 3
        # this reference depth can be set
        ref_depth = df.depth[df.dat.groupby(df.dives).idxmax().values].mean() * 3

    # find the median of bb/chl below the reference depth
    deep_avg = df[df.depth < ref_depth].groupby('dives').dat.median()

    # if the deep_avg is larger than the avg + 3 * std then bad dive
    bad_dive = deep_avg > (deep_avg.mean() + deep_avg.std() * 4)
    bad_dive = bad_dive.index.values[bad_dive]

    bad_dive_idx = array([(dives == d) for d in bad_dive]).any(0)

    return bad_dive_idx


def seperate_spikes(y, window_size, method='minmax'):
    """
    This script is copied from Nathan Briggs' MATLAB script.
    It returns the baseline of the data and the residuals of
    [measurements - baseline]

    INPUT ARGUMENTS
        y            a pandas.Series or numpy.ndarray
        window_size  the length of the rolling window size
        method       a string with `minmax` or `median`

    OUTPUT
        baseline     the baseline from which noise limits are determined
        spikes       spikes are the residual of [measurements - baseline ]

    NOTES
        method='minmax' first applies a rolling minimum to the dataset
            thereafter a rolling maximum is applied. This forms the baseline
        method='median' is a rolling median applied to the dataset

    """
    from numpy import array, ndarray, isnan, nanmedian, nanmin, nanmax, nan

    y = array(y)
    baseline = ndarray(y.shape) * nan
    mask = ~isnan(y)

    if method.startswith('min'):
        base_min = rolling_window(y[mask], nanmin, window_size)
        base = rolling_window(base_min, nanmax, window_size)
    else:
        base = rolling_window(y[mask], nanmedian, window_size)

    baseline[mask] = base

    spikes = y - baseline

    return baseline, spikes


def despiking_report(dives, depth, raw, baseline, spikes, bad_profile_idx, name='Backscatter', pcolor_kwargs={}):
    from matplotlib.pyplot import subplots, cm
    from numpy import array, unique, nanpercentile, ma
    from . plot import pcolormesh

    x = array(dives)
    y = array(depth)
    z = [array(raw),
         ma.masked_array(baseline, mask=bad_profile_idx),
         array(spikes)]

    bad = unique(x[bad_profile_idx])

    fig, ax = subplots(3, 1, figsize=[10, 11])
    title = '{}\nDespiking with Briggs et al. (2011)'.format(name)

    bmin, bmax = nanpercentile(z[1].data, [2, 98])
    smin, smax = nanpercentile(z[2].data, [2, 98])
    im = []
    props = dict(cmap=cm.Spectral_r)
    props.update(pcolor_kwargs)

    im += pcolormesh(x, y, z[0], ax=ax[0], vmin=bmin, vmax=bmax, **props),
    im += pcolormesh(x, y, z[1], ax=ax[1], vmin=bmin, vmax=bmax, **props),
    im += pcolormesh(x, y, z[2], ax=ax[2], vmin=smin, vmax=smax, **props),

    for i in range(0, 3):
        ax[i].set_ylim(400, 0)
        ax[i].set_xlim(x.min(), x.max())
        ax[i].set_ylabel('Depth (m)')
        if i != 2:
            ax[i].set_xticklabels([])
        else:
            ax[i].set_xlabel('Dive number')

        ax[i].cb.set_label('Relative Units')

    ax[0].set_title('Original')
    ax[1].set_title('Despiked')
    ax[2].set_title('Spikes with masked observations in profiles ($n={}$)'.format(bad.size))

    fig.tight_layout()
    fig.text(0.47, 1.02, title, va='center', ha='center', size=14)

    p0 = ax[0].get_position()
    p1 = ax[1].get_position()
    ax[0].set_position([p0.x0, p0.y0, p1.width, p0.height])

    return fig


def par_dark_count(par, dives, depth, time):
    from numpy import array, ma, nanmedian, isnan, nanpercentile

    par = array(par)
    dives = array(dives)
    depth = array(depth)
    time = array(time)
    # factory callibrations etc

    # DARK CORRECTION FOR PAR
    hrs = time.astype('datetime64[h]') - time.astype('datetime64[D]')
    xi = ma.masked_inside(hrs.astype(int), 22, 2)  # find 23:01 hours
    yi = ma.masked_outside(depth, *nanpercentile(depth[~isnan(par)], [90, 100]))  # 90th pctl of depth
    i = ~(xi.mask | yi.mask)
    dark = nanmedian(par[i])
    par_dark = par - dark
    par_dark[par_dark < 0] = 0

    return par_dark


def backscatter_dark_count(bbp, depth):
    from numpy import nanpercentile, isnan

    mask = (depth > 200) & (depth < 400)
    if (~isnan(bbp[mask])).sum() == 0:
        raise UserWarning(
            "\nThere are no backscatter measurements between 200 "
            "and 400 metres.\nThe dark count correction cannot be "
            "made and backscatter data can't be processed.")
    dark_pctl5 = nanpercentile(bbp[mask], 5)

    bbp -= dark_pctl5
    bbp[bbp < 0] = 0

    return bbp


def fluorescence_dark_count(flr, depth):
    from numpy import nanpercentile, isnan

    mask = (depth > 300) & (depth < 400)

    if (~isnan(flr[mask])).sum() == 0:
        raise UserWarning(
            "\nThere are no fluorescence measurements between "
            "300 and 400 metres.\nThe dark count correction "
            "cannot be made and fluorescence data can't be processed.")
    dark_pctl5 = nanpercentile(flr[mask], 5)

    flr -= dark_pctl5
    flr[flr < 0] = 0

    return flr


def par_scaling(par_uV, scale_factor_wet_uEm2s, sensor_output_mV):
    """
    Do a scaling correction for par with factory calibration coefficients.
    The factory calibrations are unique for each deployment and should be
    taken from the calibration file for that deployment.

    INPUT:  par - a pd.Series of PAR with units uV
            scale_factor_wet_uEm2s - float; unit uE/m2/sec; cal-sheet
            sensor_output_mV - float; unit mV; cal-sheet
    OUPUT:  par - pd.Series with units uE/m2/sec

    """
    sensor_output_uV = sensor_output_mV / 1000.

    par_uEm2s = (par_uV - sensor_output_uV) / scale_factor_wet_uEm2s

    return par_uEm2s


def par_fill_surface(par, dives, depth, replace_surface_depth=5, curve_max_depth=80):
    """
    Use an exponential fit to replace the surface values of PAR
    and fill missing values with equation:
        y(x) = a * exp(b * x)

    INPUT:  df - a dp.DataFrame indexed by depth
               - can also be group item grouped by dive, indexed by depth
            replace_surface_depth [5] - from replace_surface_depth to surface is replaced
            curve_depth [100] - the depth from which the curve is fit
    OUPUT:  a pd.Series with the top values replaced.

    Note that this function is a wrapper around a function
    that is applied to the gridded dataframe or groups
    """
    from scipy.optimize import curve_fit
    from pandas import DataFrame, Series
    from numpy import exp, c_, concatenate

    def fit_exp_curve(df, replace_surface_depth, curve_max_depth):

        def exp_func(x, a, b):
            """
            outputs a, b according to Equation: y(x) = a * exp(b * x)
            """
            return a * exp(b * x)

        #    no nans in data          limit to a set depth           remove surface data
        i = (df.notna().all(1)) & (df.depth > replace_surface_depth) & (df.depth < curve_max_depth)
        j = (df.par.isna()) | (df.depth < replace_surface_depth) | (df.depth > curve_max_depth)

        x = df.loc[i].depth.values
        y = df.loc[i].par.values
        if (x.size * y.size) == 0:
            y_hat = df.par.values.copy()
            return y_hat

        try:
            [a, b], _ = curve_fit(exp_func, x, y, p0=(500, -0.03), maxfev=1000)

            y_hat = a * exp(b * df.depth.values)
            par_filled = df.par.values.copy()
            par_filled[j] = y_hat[j]
            return y_hat
        except RuntimeError:
            print('Couldnt fit PAR, returned original PAR for dive')
            return df.par.values

    df = DataFrame(c_[par, dives, depth], columns=['par', 'dives', 'depth'])
    grp = df.groupby('dives')
    fitted = grp.apply(fit_exp_curve, replace_surface_depth, curve_max_depth)
    theoretical_par = Series(concatenate([l for l in fitted])).values

    return theoretical_par


def photic_depth(par, dives, depth, ref_percentage=1, max_depth=100, min_light=20, return_mask=False, return_slopes=True):
    """
    Calculates the photic depth from PAR.

    par  -  photosynthectically available radiation uE (microEinsteins)
    dives  -  ungridded
    depth  -  ungridded
    ref_precentage [1]   to find depth of
    max_depth [100]      limits the maximum euphotic depth
    min_light [20]       sets the minimumn amount of light to be considered a daytime dive
    return_mask [False]  if True returns a mask to be applied to dive data
    return_slopes [True] returns photic_depth and slopes of `ln(par)`
    """

    from pandas import DataFrame
    from numpy import c_, abs, exp, log, nan, concatenate, isnan
    from scipy.stats import linregress

    def calc_photic_depth(par, depth, return_slope):
        # returns nans/False if the par is < ~20 [default]
        if par.max() < min_light:
            if return_slope:
                return nan
            else:
                return (par * False) if return_mask else nan

        # natural log of par
        lnpar = log(par)
        # slope
        i = isnan(lnpar) | isnan(depth)
        slope = linregress(depth[~i], lnpar[~i]).slope
        if return_slope:
            return slope

        # Precentage light depth
        ld = exp((depth * -1)/(-1 / slope)) * 100.
        # finding the closest match to reference depth
        ind = abs(ld - ref_percentage).argmin()
        euph_depth = depth[ind]
        # if depth deeper than measurement depth make nan
        if euph_depth > max_depth:
            euph_depth = nan

        if return_mask:
            return depth < euph_depth
        else:
            return euph_depth

    df = DataFrame(c_[par, dives, depth], columns=['par', 'dives', 'depth'])
    grp = df.groupby('dives')

    slopes = grp.apply(lambda x: calc_photic_depth(
        x.par.values, x.depth.values, True))
    photic_depth = grp.apply(lambda x: calc_photic_depth(
        x.par.values, x.depth.values, False))

    if return_mask:
        photic_depth = concatenate(photic_depth.values).astype(bool)

    if return_slopes:
        return photic_depth, slopes
    else:
        return photic_depth


def sunset_sunrise(time, lat, lon):
    """
    Uses the Astral package to find sunset and sunrise times.
    The times are returned rather than day or night indicies.
    More flexible for quenching corrections.
    """
    from astral import Astral
    from pandas import DataFrame
    ast = Astral()

    df = DataFrame.from_dict(dict([
        ('time', time),
        ('lat', lat),
        ('lon', lon)]))

    # set days as index
    df = df.set_index(df.time.values.astype('datetime64[D]'))

    # groupby days and find sunrise for unique days
    grp_avg = df.groupby(df.index).mean()
    date = grp_avg.index.to_pydatetime()

    grp_avg['sunrise'] = list(map(ast.sunrise_utc, date, df.lat, df.lon))
    grp_avg['sunset'] = list(map(ast.sunset_utc, date, df.lat, df.lon))

    # reindex days to original dataframe as night
    df_reidx = grp_avg.reindex(df.index).astype('datetime64[ns]')
    sunrise, sunset = df_reidx[['sunrise', 'sunset']].values.T

    return sunrise, sunset


def quenching_correction(flr, bbp, dives, depth, time, lat, lon, photic_layer=None, quenching_layer=None, night_day_group=True, surface_layer=5, sunrise_sunset_offset=1):
    """
    Calculates the quenching depth and performs the quenching correction
    based on backscatter.

    INPUT:
    All inputs must be np.ndarray with all dives stacked
        flr - fluorescence, despiked
        bbp - backscatter, despiked
        dives, depth, time, lat, lon
        night_day_group = True: quenching corrected with preceding night
                          False: quenching corrected with following night
        surface_layer  - surface depth that is omitted from chlorophyll (metres)
        photic_layer - calculated using photic_depth function or a float/int that sets constant depth
        sunrise_sunset_offset - the delayed onset and recovery of quenching in hours [1] (assumes symmetrical) .

    OUTPUT:
        corrected fluorescence
        quenching layer - boolean mask of quenching depth

    METHOD:
        Correct for difference between night and daytime fluorescence.

        QUENCHING DEPTH
        ===============
        The default setting is for the preceding night to be used to
        correct the following day's quenching (`night_day_group=True`).
        This can be changed so that the following night is used to
        correct the preceding day. The quenching depth is then found
        from the differnece between the night and daytime fluorescence.
        We use the steepest gradient of the {5 minimum differences and
        the points the differnece changes sign (+ve/-ve)}.

        BACKSCATTER / CHLOROPHYLL RATIO
        ===============================
        1. Get the ratio between quenching depth and fluorescence
        2. Find the mean nighttime ratio for each night
        3. Get the ratio between nighttime and daytime quenching
        4. Apply the day/night ratio to the fluorescence
        5. If the corrected value is less than raw return to raw
    """

    import numpy as np
    import pandas as pd
    from scipy.interpolate import Rbf

    def grad_min(depth, fluor_diff, surface_layer=5):
        """
        Quenching depth for a day/night fluorescence difference

        INPUT:   depth and fluorescence as pd.Series or np.ndarray
                 surface_layer [5] is the depth to search for the
                     reference in the gradient
        OUPUT:   Quenching layer as a boolean mask
        """
        if depth.size <= surface_layer:
            return np.zeros(depth.size).astype(bool)

        x = np.array(depth)
        y = rolling_window(np.array(fluor_diff), np.mean, 5)
        s = x < surface_layer  # surface data to the top 5 metres
        mask = np.zeros(depth.size).astype(bool)

        # get the smallest 5 points and where the difference crosses 0
        small5 = np.argsort(np.abs(y))[:5]
        cross0 = np.where(np.r_[False, np.diff((y) > 0)])[0]
        # combine the indicies
        i = np.unique(np.r_[small5, cross0])
        # the max in the surface as a reference
        j = y[s].argmax()

        # calculate the gradient of the selected points to the reference
        grad = (y[s][j] - y[i]) / (x[s][j] - x[i])
        # If there are only nans in the gradient return only nans
        if np.isnan(grad).all():
            return mask
        # get the index of the steepest gradient (min)
        grad_min_i = i[np.nanargmin(grad)]

        # fill the mask with True values above the quenching depth
        mask[0:grad_min_i] = True
        # on up dives the array is backwards so reverse the mask
        if x[-1] < x[0]:
            mask = ~mask
        # If the majority of the points in the selected region are
        # negative (night < day) then return an empty mask
        return mask

    if (quenching_layer is None) & (photic_layer is None):
        raise UserWarning(
            'You need to supply either the photic_layer as a mask '
            'or the quenching_layer as a mask. '
            'The photic_layer will use the full method described in '
            'Thomalla et al. (2017).'
            'The quenching_layer can be used when PAR is not available.'
            'A manual method should then be used to define quenching_depth.'
            )
    elif (quenching_layer is not None) & (photic_layer is not None):
        raise UserWarning(
            'Only quenching_layer OR photic_layer should be given. '
            'Not both.'
            )

    # ############################ #
    #  GENERATE DAY/NIGHT BATCHES  #
    # ############################ #
    sunrise, sunset = sunset_sunrise(time, lat, lon)
    offset = np.timedelta64(sunrise_sunset_offset, 'h')
    # creating quenching correction batches, where a batch is a night and the following day
    day = (time > (sunrise + offset)) & (time < (sunset + offset))
    # find day and night transitions
    daynight_transitions = np.abs(np.diff(day.astype(int)))
    # get the cumulative sum of daynight to generate separate batches for day and night
    daynight_batches = daynight_transitions.cumsum()
    # now get the batches with padded 0 to account for the diff
    # also add a bool that makes night_day or day_night batches
    batch = np.r_[0, (daynight_batches + night_day_group) // 2]

    # ######################## #
    #  GET NIGHTTIME AVERAGES  #
    # ######################## #
    # blank arrays to be filled
    flr_night, bbp_night = flr.copy(), bbp.copy()

    # create a dataframe with fluorescence and backscatter
    df = pd.DataFrame(np.c_[flr, bbp], columns=['flr', 'bbp'])
    # get the binned averages for each batch and select the night
    night_ave = df.groupby([day, batch, np.around(depth)]).mean()
    night_ave = night_ave.dropna().loc[False]
    # A second group where only batches are grouped
    grp_batch = df.groupby(batch)

    # GETTING NIGHTTIME AVERAGE FOR NONGRIDDED DATA - USE RBF INTERPOLATION
    for b in np.unique(night_ave.index.labels[0]):
        i = grp_batch.groups[b].values  # batch index
        j = i[~np.isnan(flr[i]) & (depth[i] < 400)]  # index without nans
        x = night_ave.loc[b].index.values  # batch depth
        y = night_ave.loc[b]  # batch flr and bbp

        if y.flr.isna().all() | y.bbp.isna().all():
            continue
        # radial basis functions with a smoothing factor
        f1 = Rbf(x, y.flr.values, function='linear', smooth=10)
        f2 = Rbf(x, y.bbp.values, function='linear', smooth=10)
        # interpolation function is used to find flr and bbp for all
        # nighttime fluorescence
        flr_night[j] = f1(depth[j])
        bbp_night[j] = f2(depth[j])

    # calculate the difference between average nighttime - and fluorescence
    fluor_diff = flr_night - flr

    # ################################ #
    #  FIND THE QUENCHING DEPTH LAYER  #
    # ################################ #
    # blank array to be filled
    if quenching_layer is None:
        quenching_layer = np.zeros(depth.size).astype(bool)
        # create a grouped dataset by dives to find the depth of quenching
        cols = np.c_[depth, fluor_diff, dives][photic_layer]
        grp = pd.DataFrame(cols, columns=['depth', 'flr_dif', 'dives'])
        grp = grp.groupby('dives')
        # apply the minimum gradient algorithm to each dive
        quench_mask = grp.apply(lambda df: grad_min(df.depth, df.flr_dif))
        # fill the quench_layer subscripted to the photic layer
        quenching_layer[photic_layer] = np.concatenate([l for l in quench_mask])

    # ################################### #
    #  DO THE QUENCHING CORRECTION MAGIC  #
    # ################################### #
    # a copy of fluorescence to be filled with quenching corrected data
    flr_corrected = flr.copy()
    # nighttime backscatter to fluorescence ratio
    flr_bb_night = bbp_night / flr_night
    # quenching ratio for nighttime
    quench_ratio = flr_bb_night * flr / bbp
    # apply the quenching ratio to the fluorescence
    quench_corrected = flr / quench_ratio
    # fill the array with queching corrected data in the quenching layer only
    flr_corrected[quenching_layer] = quench_corrected[quenching_layer]

    return flr_corrected, quenching_layer


def quenching_report(flr, flr_corrected, quenching_layer, dives, depth, scatter_kwargs={}):
    from matplotlib.pyplot import subplots, cm, colorbar
    from numpy import array, nanpercentile

    y = array(depth)
    i = y < 183
    y = y[i]
    x = array(dives)[i]
    z = [array(flr)[i],
         array(flr_corrected)[i],
         array(quenching_layer)[i]]

    fig, ax = subplots(3, 1, figsize=[10, 11])
    title = 'Quenching correction with Thomalla et al. (2017)'

    bmin, bmax = nanpercentile(z[1], [2, 98])
    smin, smax = nanpercentile(z[2], [2, 98])
    im = []
    props = dict(rasterized=True, cmap=cm.YlGnBu_r)
    props.update(scatter_kwargs)
    im += ax[0].scatter(x, y, 6, z[0], vmin=bmin, vmax=bmax, **props),
    im += ax[1].scatter(x, y, 6, z[1], vmin=bmin, vmax=bmax, **props),
    im += ax[2].scatter(x, y, 6, z[2], vmin=smin, vmax=smax, **props),

    cb = []
    for i in range(0, 3):
        ax[i].set_ylim(180, 0)
        ax[i].set_xlim(x.min(), x.max())
        ax[i].set_ylabel('Depth (m)')

        cb = colorbar(mappable=im[i], ax=ax[i], pad=0.01, fraction=0.05)
        if i != 2:
            ax[i].set_xticklabels([])
            cb.set_label('Relative Units')
        else:
            ax[i].set_xlabel('Dive number')
            cb.set_label('Boolean mask')

    ax[0].set_title('Original fluorescence')
    ax[1].set_title('Quenching corrected fluorescence')
    ax[2].set_title('Quenching layer')

    fig.tight_layout()
    fig.text(0.47, 1.02, title, va='center', ha='center', size=14)

    return fig


def ismember(a, b):
    """
    Loop through a to find values that belong to b.
    Returns a boolean index of b where True shows members.
    """
    from numpy import array
    members = [(v == b) for v in a]
    return array(members).any(0)


if __name__ == '__main__':
    pass
