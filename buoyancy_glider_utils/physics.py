from __future__ import print_function
from seawater import bfrq as calc_brunt_vaisala
from seawater import dens as calc_density


def calc_mld(dens_or_temp, depth, dives, thresh=0.01, ref_depth=10, return_as_mask=False):
    """
    Calculates the MLD for a flattened array.
    You can provide density or temperature.
    The default threshold is set for density (0.01).

    INPUT
        dens_or_temp - series/ndarray of temperature/density with length M
        depth - series/ndarray of depth with length M
        dives - series/ndarray of depth with length M
        [thresh] - threshold for difference 0.01
        [ref_depth] - reference depth for difference (10m)

    OUTPUT
        mld as an np.ndarray
    """
    import numpy as np
    from pandas import DataFrame

    def mld_profile(dens_or_temp, depth, thresh, ref_depth, mask=False):

        i = np.nanargmin(np.abs(depth - ref_depth))

        if np.isnan(dens_or_temp[i]):
            mld = np.nan
        else:
            dd = dens_or_temp - dens_or_temp[i]  # density difference
            dd[depth < ref_depth] = np.nan
            abs_dd = abs(dd - thresh)
            depth_idx = np.nanargmin(abs_dd)
            mld = depth[depth_idx]

        if mask:
            return depth <= mld
        else:
            return mld

    arr = np.c_[dens_or_temp, depth, dives]
    col = ['dens', 'depth', 'dives']
    df = DataFrame(data=arr, columns=col)

    grp = df.groupby('dives')
    mld = grp.apply(lambda g: mld_profile(g.dens.values, g.depth.values, thresh, ref_depth, mask=return_as_mask))

    if return_as_mask:
        return np.concatenate([l for l in mld])
    else:
        return mld


def calc_potential_density(salt_PSU, temp_C, pres_db, lat, lon, pres_ref=0):
    """
    The Basestation calculates density from absolute salinity and
    potential temperature. This function is a wrapper for this
    functionality. Note that a reference pressure of 0 is used
    by default.

    INPUT
    salt_PSU    practical salinty
    temp_C      temperature in deg C
    pres_db     pressure in decibar
    lat         latitude in degrees north
    lon         longitude in degrees east

    OUPUT
    potential density

    NOTE
    Using seawater.dens does not yield the same results as
    this function. We get very close results to what the
    SeaGlider Basestation returns with this function.
    The difference of this function with the basestation
    is on average ~ 0.003 kg/m3
    """
    import gsw

    salt_abs = gsw.SA_from_SP(salt_PSU, pres_db, lat, lon)
    temp_pot = gsw.t_from_CT(salt_abs, temp_C, pres_db)

    pot_dens = gsw.pot_rho_t_exact(salt_abs, temp_pot, pres_db, pres_ref)

    return pot_dens
