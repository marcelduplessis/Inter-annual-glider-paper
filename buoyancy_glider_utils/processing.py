from __future__ import print_function


def calc_physics(variable, dives, depth,
                 neighbourhood_size_dives=40, neighbourhood_size_depth=40, iqr_multiplier=3,
                 savitzkey_golay_window=11, savitzkey_golay_order=2,
                 verbose=True, name='Physics Variable'):
    """
    This function is a standard setup for processing physics
    variables (temperature and salinity).

    The function applies a neighbourhood interquartile range (IQR)
    outlier filter followed by a Savitzky-Golay smoothing function.

    For more information about the neighbourhood IQR outlier filter
    see the docs for buoyancy_glider_utils.tools.neighbourhood_iqr.

    The Savitzky-Golay filter is demonstrated well on wikipedia:
    https://en.wikipedia.org/wiki/Savitzky-Golay_filter
    """

    from pandas import Series
    from numpy import array, ma, NaN, inf
    from .tools import savitzky_golay, neighbourhood_iqr

    x = array(dives)
    y = array(depth)
    # an interpolation step is added so that no nans are created.
    # Note that this interpolates on the flattened series
    if isinstance(variable, Series):
        name = variable.name
    z = Series(variable).interpolate(limit=savitzkey_golay_window).bfill(limit=savitzkey_golay_window).values

    if neighbourhood_iqr != inf:
        if verbose:
            print(
                '\n' + '=' * 50 + "\n{}:\n"
                "\tMask bad data:\n\t\t"
                "neighbourhood (dives={}, depth={}m) "
                "interquartile range (IQR={})"
                " ".format(name, neighbourhood_size_dives, neighbourhood_size_depth, iqr_multiplier))
        outliers = neighbourhood_iqr(x, y, z, neighbourhood_size_dives, neighbourhood_size_depth, iqr_multiplier)
        z[outliers] = NaN
        z = ma.masked_where(outliers, z)

    if savitzkey_golay_window:
        if verbose:
            print("\tSmoothing with Savitzky-Golay filter (window={}, order={})".format(savitzkey_golay_window, savitzkey_golay_order))
        z = savitzky_golay(z, savitzkey_golay_window, savitzkey_golay_order)

    return z


def calc_bb(bb_raw, wavelength, tempC, salt, dives, depth,
            dark_count, scale_factor,
            briggs_spike_window=7,
            neighbour_iqr_multiplier=1.6,
            neighbour_depth_window=40, neighbour_dives_window=40,
            return_figure=False, verbose=True):
    """
    INORMATION
    ----------
    Process the raw backscattering data from channel 2 at wavelength 700 nm.
    NOTE: This uses the same processing steps as calc_bb1.

    Use standard values for the CSIR SeaGliders to process backscatter.
        wavelength = 700nm
        theta angle of sensors = 124deg
        xfactor for theta 124 = 1.076
    This function also makes use of the flo_functions toolkit (Zhang et al. 2009)
    to calculate total backscatter.

    The following steps are applied in this sequence:
    1. flo_scale_and_offset (factory scale and offset)
    2. flo_bback_total  (total backscatter based on Zhang et al. 2009)
    3. backscatter_dark_count  (based on Briggs et al. 2011)
    4. find_bad_profiles  (high values below 300 m are counted as bad profiles)
    5. seperate_spikes  (using Briggs et al. 2011 - rolling min--max)
    6. neighbourhood_iqr  (see buoyancy_glider_utils.tools.neighbourhood_iqr)

    INPUT
    -----
    All inputs must be ungridded np.ndarray or pd.Series data
    bb_raw      raw output from backscatter 470 nm
    tempC       QC'd temperature in degC
    salt        QC'd salinity in PSU
    dives       the dive count (round is down dives, 0.5 up dives)
    depth       in metres

    wavelength    e.g. 700 nm / 470 nm
    dark_count    factory values from the cal sheet
    scale_factor  factory values from the cal sheet
    briggs_spike_window  the window size over which to run the despiking

    return_figure return a figure object that shows the before and after
    verbose       will print the progress of the processing

    OUTPUT
    ------
    baseline    an np.ma.masked_array with the mask denoting the filtered values
                the baseline of the backscatter as defined by Briggs et al. 2011
    spikes      the spikes of backscatter from Briggs et al. 2011
    fig         a figure object if return_figure=True else returns None
    """
    from numpy import array, nan
    from . import optics as op
    from . import flo_functions as ff
    from .tools import neighbourhood_iqr

    bb_raw = array(bb_raw)
    dives = array(dives)
    depth = array(depth)
    tempC = array(tempC)
    salt = array(salt)

    theta = 124  # factory set angle of optical sensors
    xfactor = 1.076  # for theta 124

    iqr_multiplier = neighbour_iqr_multiplier
    depth_window = neighbour_depth_window
    dives_window = neighbour_dives_window

    if verbose:
        print('\n' + '=' * 50 + '\nBackscatter ({:.0f}nm)\n\tZhang et al. (2009) correction'.format(wavelength))
    beta = ff.flo_scale_and_offset(bb_raw, dark_count, scale_factor)
    bbp = ff.flo_bback_total(beta, tempC, salt, theta, wavelength, xfactor)

    # This is from .Briggs et al. (2011)
    if verbose:
        print('\tDark count correction')
    bbp = op.backscatter_dark_count(bbp, depth)

    if verbose:
        print(
            "\tMask bad data:\n\t\t"
            "bad profiles based on deep values\n\t\t"
            "neighbourhood (dives={}, depth={}m) interquartile range (IQR={})".format(dives_window, depth_window, iqr_multiplier))
    bad_profiles = op.find_bad_profiles(dives, depth, bbp, ref_depth=150)
    baseline, spikes = op.seperate_spikes(bbp, briggs_spike_window)

    bad_points = neighbourhood_iqr(dives, depth, baseline, dives_window, depth_window, iqr_multiplier)

    mask = bad_profiles | bad_points
    baseline[mask] = nan

    if not return_figure:
        return baseline, spikes
    else:
        if verbose:
            print('\tGenerating figure for despiking report')
        fig = op.despiking_report(dives, depth, bbp, baseline, spikes, mask, name='BB ({:.0f}nm)'.format(wavelength))

        return baseline, spikes, fig


def calc_fluorescence(flr_raw, bbp, dives, depth, time, lat, lon, dark_count, scale_factor, par=None, quenching_layer=None, return_figure=False, verbose=True):
    """
    INFORMATION
    -----------
    This function processes Fluorescence and corrects for quenching using
    the Thomalla et al. (2017) approach.

    The standard sequence is applied:
    1. fluorescence_dark_count  (factory correction)
    2. find_bad_profiles  (high Fluorescence in > 300 m water signals bad profile)
    3. seperate_spikes  (using Briggs et al. 2011 - rolling min--max)
    4. photic_depth  (find photic depth based on PAR)
    5. quenching_correction  (corrects for quenching with Thomalla et al. 2017)
    6. neighbourhood_iqr  (find the outliers based on the interquartile range)

    INPUT
    -----
    All inputs must be ungridded np.ndarray or pd.Series data
    flr_raw     raw output from backscatter 470 nm
    bbp         processed backscatter from less noisy channel
    dives       the dive count (round is down dives, 0.5 up dives)
    depth       in metres
    time        as a np.datetime64 array
    lat, lon    latitude and longitude

    dark_count    factory values from the cal sheet
    scale_factor  factory values from the cal sheet

    par           PAR is optional, if not given quenching depth search is limited to 100m
    return_figure return a figure object that shows the before and after
    verbose       will print the progress of the processing

    OUTPUT
    ------
    baseline            uncorrected, but despiked fluorescence
    quench_corrected    quench corrected fluorescence
    quench_layer        the quenching layer as a mask
    figs                figures reporting the despiking and quenching correction
    """

    from .tools import neighbourhood_iqr
    from numpy import array, ma, NaN
    from . import optics as op

    flr_raw = array(flr_raw)
    bbp = array(bbp)
    par = array(par)
    dives = array(dives)
    depth = array(depth)
    time = array(time)
    lat = array(lat)
    lon = array(lon)

    iqr_multiplier = 2
    depth_window = 40
    dives_window = 40

    if verbose:
        print('\n' + '=' * 50 + '\nFluorescence\n\tDark count correction')

    flr_raw[flr_raw < 0] = NaN
    flr_raw -= dark_count
    flr_dark = op.fluorescence_dark_count(flr_raw, depth)
    bad_profiles = op.find_bad_profiles(flr_dark, dives, depth)
    baseline, spikes = op.seperate_spikes(flr_raw, 7)

    if par is not None:
        if verbose:
            print('\tCalculating the photic depth from PAR')
        photic_layer = op.photic_depth(
            par, dives, depth, return_mask=True, return_slopes=False)
    else:
        photic_layer = None

    if verbose:
        print('\tQuenching correction')
    quench_corrected, quench_layer = op.quenching_correction(
        baseline, bbp, dives, depth, time, lat, lon,
        photic_layer=photic_layer,
        quenching_layer=quenching_layer,
        )

    if verbose:
        print(
            "\tMask bad data:\n\t\t"
            "bad profiles based on deep values\n\t\t"
            "neighbourhood (dives={}, depth={}m) interquartile range (IQR={})".format(dives_window, depth_window, iqr_multiplier))
    bad_points = neighbourhood_iqr(dives, depth, quench_corrected, dives_window, depth_window, iqr_multiplier)
    mask = bad_profiles | bad_points

    baseline = ma.masked_where(mask, baseline)
    quench_corrected = ma.masked_where(mask, quench_corrected)

    if return_figure:
        if verbose:
            print('\tGenerating figures for despiking and quenching report')
        figs = op.despiking_report(dives, depth, flr_raw, baseline.data, spikes, mask, name='Fluorescence'),
        figs += op.quenching_report(baseline.data, quench_corrected.data, quench_layer, dives, depth),
        return baseline, quench_corrected, quench_layer, figs
    else:
        return baseline, quench_corrected, quench_layer


def calc_par(par_raw, dives, depth, time, scale_factor_wet_uEm2s, sensor_output_mV, replace_surface_depth=2, curve_max_depth=80, verbose=True):
    """
    INFORMATION
    -----------
    Calculates the theoretical PAR based on an exponential curve fit.

    The processing steps are:
    1. par_scaling  (factory cal sheet scaling)
    2. par_dark_count  (correct deep par values to 0 using 5th %)
    3. par_fill_surface  (return the theoretical curve of par based exponential fit)

    INPUT
    -----
    All inputs must be ungridded np.ndarray or pd.Series data
    par_raw     raw PAR
    dives       the dive count (round is down dives, 0.5 up dives)
    depth       in metres
    time        as a np.datetime64 array
    """

    from numpy import array
    from . import optics as op

    par_raw = array(par_raw)
    dives = array(dives)
    depth = array(depth)
    time = array(time)

    if verbose:
        print('\n' + '=' * 50 + '\nPAR\n\tDark correction')

    # dark correction for par
    par_scaled = op.par_scaling(par_raw, scale_factor_wet_uEm2s, sensor_output_mV)
    par_dark = op.par_dark_count(par_scaled, dives, depth, time)
    if verbose:
        print('\tFitting exponential curve to data')
    par_filled = op.par_fill_surface(par_dark, dives, depth, replace_surface_depth, curve_max_depth)
    par_filled[par_filled < 0] = 0

    photic_depth = op.photic_depth(par_filled, dives, depth, 10, return_mask=False)  # 10 is the dark value

    return par_filled, photic_depth


if __name__ == '__main__':
    pass
