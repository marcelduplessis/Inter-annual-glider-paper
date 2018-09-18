from __future__ import print_function


def _process_2D_plot_args(args, gridding_dz=1):

    from numpy import array, ma, nan, ndarray, nanmax, arange
    from pandas import DataFrame, Series
    from xarray import DataArray
    from . import tools

    if len(args) == 3:
        x = array(args[0])
        y = array(args[1]).astype(float)
        z = args[2]
        if isinstance(z, ma.MaskedArray):
            z[z.mask] = nan
        else:
            z = ma.masked_invalid(array(z)).astype(float)

        if (x.size == y.size) & (len(z.shape) == 1):
            bins = arange(0, nanmax(y), gridding_dz)
            df = tools.bin_depths(z, x, y, bins=bins)
            x = df.columns
            y = df.index
            z = ma.masked_invalid(df.values)
        return x, y, z

    elif len(args) == 1:
        z = args[0]
        if isinstance(z, DataArray):
            z = z.to_series().unstack()
        elif isinstance(z, (ndarray, Series)):
            if z.ndim == 2:
                z = DataFrame(z)
            else:
                raise IndexError('The input must be a 2D DataFrame or ndarray')

        x = z.columns.values
        y = z.index.values
        z = ma.masked_invalid(z.values).astype(float)

        return x, y, z


def save_figures_to_pdf(fig_list, pdf_name, **savefig_kwargs):
    import matplotlib.backends.backend_pdf
    from matplotlib import pyplot as plt

    pdf = matplotlib.backends.backend_pdf.PdfPages(pdf_name)
    for fig in fig_list:  # will open an empty extra figure :(
        pdf.savefig(fig.number, dpi=120)
    pdf.close()
    plt.close('all')


class plot_functions(object):
    @staticmethod
    def __new__(*args, **kwargs):

        if len(args) > 1:
            args = args[1:]
        return plot_functions.pcolormesh(*args, **kwargs)

    @staticmethod
    def pcolormesh(*args, **kwargs):
        """
        Plot a section plot of the dives with x-time and y-depth and
        z-variable. The data can be linearly interpolated to fill missing
        depth values. The number of points to interpolate can be set with
        interpolate_dist.

        *args can be:
            - same length x, y, z. Will be gridded with depth of 1 meter.
            - x(m), y(n), z(n, m) arrays
            - z DataFrame where indicies are depth and columns are dives
            - z DataArray where dim0 is dives and dim1 is depth
        **kwargs can be:
            - ax - give an axes to the plotting function
            - robust - use the 0.5 and 99.5 percentile to set color limits
            - gridding_dz - gridding depth [default 1]

        The **kwargs can be anything that gets passed to plt.pcolormesh.
        Note that the colour is scaled to 1 and 99% of z.
        """
        from matplotlib.pyplot import colorbar, subplots
        from numpy import datetime64, nanpercentile
        from datetime import datetime

        ax = kwargs.pop('ax', None)
        robust = kwargs.pop('robust', False)
        gridding_dz = kwargs.pop('gridding_dz', 1)

        x, y, z = _process_2D_plot_args(args, gridding_dz=gridding_dz)

        x_time = isinstance(x[0], (datetime, datetime64))

        if robust & (('vmin' not in kwargs) | ('vmax' not in kwargs)):
            kwargs['vmin'] = nanpercentile(z.data, 0.5)
            kwargs['vmax'] = nanpercentile(z.data, 99.5)

        if ax is None:
            fig, ax = subplots(1, 1, figsize=[11, 4])
        else:
            fig = ax.get_figure()

        im = ax.pcolormesh(x, y, z, rasterized=True, **kwargs)
        ax.cb = colorbar(mappable=im, pad=0.02, ax=ax)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.max(), y.min())
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date' if x_time else 'Dives')

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax

    @staticmethod
    def contourf(*args, **kwargs):
        """
        Plot a section plot of the dives with x-time and y-depth and
        z-variable. The data can be linearly interpolated to fill missing
        depth values. The number of points to interpolate can be set with
        interpolate_dist.

        *args can be:
            - same length x, y, z. Will be gridded with depth of 1 meter.
            - x(m), y(n), z(n, m) arrays
            - z DataFrame where indicies are depth and columns are dives
            - z DataArray where dim0 is dives and dim1 is depth
        **kwargs can be:
            - ax - give an axes to the plotting function
            - robust - use the 0.5 and 99.5 percentile to set color limits
            - gridding_dz - gridding depth [default 1]

        The **kwargs can be anything that gets passed to plt.pcolormesh.
        Note that the colour is scaled to 1 and 99% of z.
        """

        from matplotlib.pyplot import colorbar, subplots
        from numpy import percentile, datetime64
        from datetime import datetime

        ax = kwargs.pop('ax', None)
        robust = kwargs.pop('robust', False)
        gridding_dz = kwargs.pop('gridding_dz', 1)

        x, y, z = _process_2D_plot_args(args, gridding_dz=gridding_dz)

        x_time = isinstance(x[0], (datetime, datetime64))

        if robust & (('vmin' not in kwargs) | ('vmax' not in kwargs)):
            kwargs['vmin'] = percentile(z[~z.mask], 0.5)
            kwargs['vmax'] = percentile(z[~z.mask], 99.5)

        if ax is None:
            fig, ax = subplots(1, 1, figsize=[11, 4])
        else:
            fig = ax.get_figure()

        im = ax.contourf(x, y, z, **kwargs)
        ax.cb = colorbar(mappable=im, pad=0.02, ax=ax)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.max(), y.min())
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date' if x_time else 'Dives')

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax

    @staticmethod
    def scatter(x, y, z, ax=None, robust=False, **kwargs):
        from matplotlib.pyplot import colorbar, subplots
        from numpy import ma, nanpercentile, datetime64, array, nanmin, nanmax
        from datetime import datetime

        x = array(x)
        y = array(y)
        z = ma.masked_invalid(z)

        x_time = isinstance(x[0], (datetime, datetime64))

        if robust:
            kwargs['vmin'] = nanpercentile(z, 0.5)
            kwargs['vmax'] = nanpercentile(z, 99.5)

        if ax is None:
            fig, ax = subplots(1, 1, figsize=[11, 4])
        else:
            fig = ax.get_figure()
        im = ax.scatter(x, y, c=z, rasterized=True, **kwargs)

        ax.cb = colorbar(mappable=im, pad=0.02, ax=ax)
        ax.set_xlim(nanmin(x), nanmax(x))
        ax.set_ylim(nanmax(y), nanmin(y))
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('Date' if x_time else 'Dives')

        [tick.set_rotation(45) for tick in ax.get_xticklabels()]
        fig.tight_layout()

        return ax

    @staticmethod
    def bin_size(depth, **hist_kwargs):
        from matplotlib.pyplot import subplots, colorbar
        from matplotlib.colors import LogNorm
        from numpy import abs, diff, isnan, array

        depth = array(depth)

        x = abs(diff(depth))
        y = depth[1:]
        m = ~(isnan(x) | isnan(y))
        x, y = x[m], y[m]

        fig, ax = subplots(1, 1, figsize=[4, 6])
        im = ax.hist2d(x, y, bins=50, norm=LogNorm(),
                    rasterized=True, **hist_kwargs)[-1]
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_ylabel('Depth (m)')
        ax.set_xlabel('$\Delta$ Depth (m)')

        cb = colorbar(mappable=im, ax=ax, fraction=0.1, pad=0.05)
        cb.set_label('Measurement count')

        fig.tight_layout()
        return ax


if __name__ == '__main__':
    pass
    "fun people"

