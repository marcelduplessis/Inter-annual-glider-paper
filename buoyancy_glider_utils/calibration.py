import numpy as _np


def bottle_matchup(gld_time, gld_depth, gld_dives, btl_time, btl_depth, btl_values, min_depth_diff_metres=5, min_time_diff_minutes=120):
    """
    Performs a matchup between glider and bottles based on time and depth.

    INPUT
    gld_time    glider time as datetime64
    gld_depth   glider depth
    gld_dives   dive index of the glider
    btl_time    bottle time as datetime64
    btl_depth   bottle depth
    btl_values  the values of the bottles

    min_depth_diff_metres  [  5]  the minimum allowable depth difference
    min_time_diff_minutes  [120]  the minimum allowable time difference between bottles and glider


    OUPUT
    Returns the bottle values in the format of the glider
    i.e. the length of the output will be the same as gld_*

    """

    # make all input variables np.arrays
    args = gld_time, gld_depth, gld_dives, btl_time, btl_depth, btl_values
    gld_time, gld_depth, gld_dives, btl_time, btl_depth, btl_values = map(_np.array, args)

    # create a blank array that matches glider data (placeholder for calibration bottle values)
    gld_cal = _np.ones_like(gld_depth) * _np.nan

    # loop through each ship based CTD station
    for t in _np.unique(btl_time):
        # index of station from ship CTD
        btl_idx = t == btl_time
        # number of samples per station
        btl_num = btl_idx.sum()

        # string representation of station time
        t_str = str(t.astype('datetime64[m]')).replace('T', ' ')
        t_dif = abs(gld_time - t).astype('timedelta64[m]').astype(float)

        # loop through depths for the station
        if t_dif.min() < min_time_diff_minutes:
            # index of dive where minimum difference occurs
            i = _np.where(gld_dives[_np.nanargmin(t_dif)] == gld_dives)[0]
            n_depths = 0
            for depth in btl_depth[btl_idx]:
                # an index for bottle where depth and station match
                j = btl_idx & (depth == btl_depth)
                # depth difference for glider profile
                d_dif = abs(gld_depth - depth)[i]
                # only match depth if diff is less than given threshold
                if _np.nanmin(d_dif) < min_depth_diff_metres:
                    # index of min diff for this dive
                    k = i[_np.nanargmin(d_dif)]
                    # assign the bottle values to the calibration output
                    gld_cal[k] = btl_values[j]
                    n_depths += 1
            print('SUCCESS: {} ({} of {} samples) match-up within {} minutes'.format(t_str, n_depths, btl_num, t_dif.min()))

    return gld_cal


def model_figs(x, y, model):
    """
    Creates the figure for a linear model fit.

    INPUT
    x, y  - two arrays of the same length of glider and bottle data
    model - must be from the sklearn.linear_model module.

    OUTPUT
    figure axes
    """

    from matplotlib.pyplot import subplots
    from sklearn import metrics

    y_hat = model.predict(x).squeeze()
    m_name = 'Huber Regresion'

    xf = _np.linspace(x.min(), x.max(), 100)[:, None]

    fig, ax = subplots(1, 1, figsize=[6, 5], dpi=120)
    ax.plot(x, y, 'o', c='k', label='Samples')[0]

    if hasattr(model, 'outliers_'):
        ol = model.outliers_
        ax.plot(x[ol], y[ol], 'ow', mew=1, mec='k', label='Outliers ({})'.format(ol.sum()))
    else:
        ol = _np.zeros_like(y).astype(bool)

    formula = '$f(x) = {:.2g}x + {:.2g}$'.format(model.coef_[0], model.intercept_)
    ax.plot(xf, model.predict(xf), c='#AAAAAA', label='{}'.format(formula))

    ax.legend()

    params = model.get_params()
    rcModel = model.__class__().get_params()
    for key in rcModel:
        if rcModel[key] == params[key]:
            params.pop(key)

    max_len = max(len(str(params[p]))for p in params) + 1
    placeholder = u"{{}}:{{: >{}}}\n".format(max_len)
    params_str = "{} Params\n".format(m_name)
    line1_len = len(params_str)
    for key in params:
        params_str += placeholder.format(key, params[key])

    params_str += "{}\nResults (robust)\n".format('-' * (line1_len-1))
    r2_str = "$r^2$ score: {:.2g} ({:.2g})\n"
    rmse_str = "RMSE: {:.2g} ({:.2g})"

    r2_all = metrics.r2_score(y, y_hat)
    r2_robust = metrics.r2_score(y[~ol], y_hat[~ol])
    rmse_all = metrics.mean_squared_error(y, y_hat)**0.5
    rmse_robust = metrics.mean_squared_error(y[~ol], y_hat[~ol])**0.5
    params_str += r2_str.format(r2_all, r2_robust)
    params_str += rmse_str.format(rmse_all, rmse_robust)

    ax.text(x.max(), y.min(), params_str,
            fontdict={'family': 'monospace'}, ha='right',
            bbox=dict(facecolor='none', boxstyle='round,pad=0.5', lw=0.2))

    ax.set_ylabel('Bottle sample')
    ax.set_xlabel('Glider sample')
    ax.set_title('Calibration curve using {}'.format(m_name))

    # ax.set_ylim(0, ax.get_ylim()[1])

    return ax


def robust_linear_fit(gld_var, gld_var_cal, interpolate_limit=3, return_figures=True, **kwargs):
    """
    Perform a robust linear regression using a Huber Loss Function
    to remove outliers.

    INPUT
    gld_var      glider variable
    gld_var_cal  bottle variable on glider indicies
    epsilon [1.35] Larger epsilon is a slacker restriction on
            outliers (fewer points). Outliers are excluded
            completely and not weighted in this regression method.
    fit_intercept [False] forces 0 intercept if False
    return_figures [True] create figure with metrics
    interpolate_limit [3] glider data may have missing points. The glider
            data is thus interpolated to ensure that as many bottle samples
            as possible have a match-up with the glider.

    OUPUT
    model   A fitted model sklearn.linear_model (see input)
            Use model.predict(glider_var) to create the calibrated output.
    """

    from sklearn import linear_model
    from pandas import Series

    # make all input arguments numpy arrays
    args = gld_var, gld_var_cal
    gld_var, gld_var_cal = map(_np.array, args)

    gld_var = Series(gld_var).interpolate(limit=interpolate_limit).values

    # get bottle and glider values for the variables
    i = ~_np.isnan(gld_var_cal)
    y = gld_var_cal[i]
    x = gld_var[i][:, None]

    if "fit_intercept" not in kwargs:
        kwargs["fit_intercept"] = False
    model = linear_model.HuberRegressor(**kwargs)

    model.fit(x, y)

    if return_figures:
        model_figs(x, y, model)

    model._predict = model.predict

    def predict(self, x):
        """
        A wrapper around the normal predict function that takes
        nans into account. An extra dimension is also added if needed.
        """
        x = _np.array(x)
        out = _np.ndarray(x.size) * _np.NaN
        i = ~_np.isnan(x)
        if len(x.shape) != 2:
            x = x[:, None]
        out[i] = self._predict(x[i]).squeeze()

        return out

    model.predict = predict.__get__(model, linear_model.HuberRegressor)

    return model
