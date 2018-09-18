"""
Contains all the classes and functions used to read and import the
data from the basestation netCDF files.

SeaGlider is the only one you'll need.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import pickle
import os
import time

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_basestation_netCDF_files(files_or_globdir, verbose=True):

    return SeaGlider(files_or_globdir, verbose=verbose)


class SeaGlider:
    """
    This contains a class that reads in the base station netCDF file.
    It is designed to be as versatile as possible and dynamically
    reads in data for quick processing and exploration.

    Just pass either a globlike path or a list of files that are to
    be read by the class. You will be able to see basic information
    about the netCDF files by returning the resulting object.
    For a full list of variables return the .vars object.

    The variables can be accessed directly as objects.

    All loaded data is stored in .data[<dimension_name>] as a
    pandas.DataFrame.
    """

    def __init__(self, files_or_globdir, verbose=True):
        from glob import glob
        dt64 = np.datetime64

        # setting up standard locs
        if isinstance(files_or_globdir, (list, np.ndarray)):
            self.files = np.sort(files_or_globdir)
            self.directory = os.path.split(self.files[0])[0]
        elif isinstance(files_or_globdir, str):
            self.directory = files_or_globdir
            self.files = np.sort(glob(self.directory))
        self.vars = VariableDisplay()
        self.dims = {}
        self.verbose = verbose

        if len(self.files) < 1:
            raise OSError("There are no files in the given directory: {}".
                          format(self.directory))

        # loading data for dates and dummy vars
        nc0 = Dataset(self.files[0])
        nc1 = Dataset(self.files[-1])

        # creating dimensions where data is stored
        dims = np.array(list(nc0.dimensions.keys()))
        dims = dims[np.array([not d.startswith('string') for d in dims])]
        dims = np.r_[dims, ['string']]
        self.dims = {k: [] for k in dims}
        self.data = {k: pd.DataFrame() for k in dims}

        for key in nc0.variables:
            dim = nc0.variables[key].dimensions
            # dims can be a valid dimension, () or string_n
            if dim:  # catch empty tuples
                # there are dimensions that are just `string_n` placeholders
                if dim[0].startswith('string'):  # SeaGliderPointVariable
                    var_obj = PointVariable(key, self.files, df=self.data['string'], parent=self)
                    self.dims['string'] += key,
                else:  # SeaGliderDiveVariable
                    var_obj = DiveVariable(key, self.files, df=self.data[dim[0]], parent=self)
                    self.dims[dim[0]] += key,
            else:  # SeaGliderPointVariable
                var_obj = PointVariable(key, self.files, df=self.data['string'], parent=self)
                self.dims['string'] += key,
            # assign as an object of SeaGlider
            setattr(self.vars, key, var_obj)
            setattr(self, key, var_obj)

        t0 = dt64(nc0.getncattr('time_coverage_start').replace('Z', ''))
        t1 = dt64(nc1.getncattr('time_coverage_end').replace('Z', ''))
        self.date_range = np.array([t0, t1], dtype='datetime64[s]')

        nc0.close()
        nc1.close()
        self._update_coords_fix_()

    def __getitem__(self, key):
        from xarray import open_dataset

        if type(key) == int:
            fname = self.files[key]
            nco = open_dataset(fname)
            return nco
        elif type(key) == list:
            if all([type(k) is str for k in key]):
                self.load_multiple_vars(key)
            else:
                return "All arguements must be strings"
        elif type(key) == str:
            return self.vars.__dict__[key]
        else:
            return "Indexing with {} does not yet exist".format(key)

    def __repr__(self):
        df = self.vars._var_dataframe_()
        dfL = df[df.Loaded]
        txt = (
            "\n{data_type}"
            "\n-----------------------------------------"
            "\n    DATA PATH: {fname}"
            "\n    FILES: {num_files}"
            "\n    AVAILABLE VARIABLES: {avail_vars}"
            "\n    LOADED VARIABLES: {loaded_vars}"
            "\n"
            "\nShowing only loaded variables"
            "\n-----------------------------------------"
            "\n{df_str}"
        ).format(
            data_type=self.__class__,
            num_files=self.files.size,
            fname=self.directory,
            avail_vars=str(df.shape[0]),
            loaded_vars=str(dfL.shape[0]),
            df_str=str(dfL),
        )

        return txt

    def _repr_html_(self):

        df = self.vars._var_dataframe_()
        df1 = df[df.Dimension != 'string']
        df2 = df[df.Dimension == 'string']
        df3 = df[df.Loaded]
        basic_usage = """
        <h2>Basic usage</h2>
            <p>
            Files are read into a SeaGlider object that is refered to as <code>SG</code>.
            On initalisation the metadata for dive variables are read in, but the data is not loaded.

            <h5>Variable access</h5>
            Access individual variables wtih <code>SG.vars.var_name</code> or using dictionary
            syntax on <code>SG</code> with <code>SG['var_name']</code>.
            Load the data with <code>SG.vars.var_name.load()</code> and once this
            has been done for a session, you will not have to load the data again.

            <h5>Plotting and gridding</h5>
            Variables can be plotted as sections with <code>...var_name.pcolormesh()</code> or <code>...var_name.scatter()</code>.
            Note that the <code>pcolormesh</code> variant will grid the data with a preset interpolation scheme of one metre depths.
            Missing data in bins is linearly interpolated.
            A custom gridding scheme can be applied with <code>SG.vars.var_name.bindata</code>

            <h5>Saving data</h5>
            Use <code>SG.save(file_name)</code> to save and <code>sgu.load(file_name)</code> to load the data again.
            <br>
            Alternatively data can also be saved in standard formats.
            All loaded data can be accessed from central storage objects in <code>SG.data</code> - a dictionary that
            contains <code>pandas.DataFrames</code> for each dimension. These dataframes can be saved using standard pandas methods.
            Similarly, gridded data is stored centrally as a <code>xarray.Dataset</code> at <code>SG.gridded</code>.
            This dataset can be saved using standard xarray methods.
            <br><br>
            <b>For full documentation see <a href="/#">user documentation</a></b>
            </p>
            <hr>
        """

        html = """

        <hr>
        <div style="float:left; max-width:450px; min-width:450px">
            <h2 style="">Dataset summary</h2>
            <p>
                DATA PATH: <code>{fname}</code><br>
                FILES: <code>{num_files}</code><br>
                DIVE VARIABLES: <code>{plt_vars}</code> (see in .variables)<br>
                STRING VARIABLES: <code>{str_vars}</code><br>
                <br>
                Access all the imported variables with <code>SG.data[dim_name]</code>
            </p>
            <hr>
            {usage}
        </div>
        <div style="width:100%; align: right">
            <div>
            <h2 style="">Table of variables</h2>
            {df_html}
            </div>
        </div>


        """.format(
            usage=basic_usage if self.verbose == 2 else "",
            num_files=self.files.size,
            fname=self.directory if len(self.directory) < 50 else '...' + self.directory[-35:],
            plt_vars=str(df1.shape[0]),
            str_vars=str(df2.shape[0]),
            df_html=df3.to_html(),
        )

        return html

    def _update_coords_fix_(self):
        """
        This function fixes coordinates so that all variables have the most complete
        version of the coordinates. It does this accross dimensions for the dive variables.
        """

        coords = {dim: set() for dim in self.dims}  # create a set for every dimension
        sizes = {}  # create a dictionary that will capture the variable size of first nc file
        for key in self.vars.__dict__:
            var = self.vars.__dict__[key]
            if type(var) == DiveVariable:  # only dive varialbes are considered
                dim = var.dims[0]
                coords[dim].update(var.coords)  # collect all the dimension variables
                sizes[dim] = var._size_same  # store the sizes variables of first nc file

        # Find the dimensions that are the same size
        sizes_rev, same_size = {}, set()
        for k, v in sizes.items():
            # where k is the dimension name and v is the size
            try:
                sizes_rev[v].append(k)  # only for duplicates
                same_size.update([v])  # will store key of duplicates
            except KeyError:
                sizes_rev[v] = [k]

        swap_dim = {}  # create a dictionary that swaps dimensions if the same size
        for key in same_size:
            dims = sizes_rev[key]  # dimensions of same sizes
            length = [len(self.dims[d]) for d in dims]  # get the num of vars in each duplicate dim
            bigger = np.argmax(length)  # get the index of the largest dimension
            keep = dims.pop(bigger)  # the dimension that key will be swapped to
            for d in dims:
                swap_dim[d] = keep  # add to the dictionary for substitution

        for key in self.vars.__dict__:
            var = self.vars.__dict__[key]
            if type(var) == DiveVariable:
                dim = var.dims[0]
                var.coords = list(coords[dim]) + ['dives']
                # if the dimension is in our swap dictionary, then substitute
                # so that more complete coords are assigned
                if dim in swap_dim:
                    d = swap_dim[dim]
                    k = self.dims[dim]
                    var.dims = d,
                    self.dims[d] += [key]
                    self.dims[dim].remove(key)
                    var.coords = list(coords[d]) + ['dives']
                    var.__data__ = self.data[d]
                # assign the most complete coordinates to the variable coordinates

    def save(self, file_name, complevel=1):
        """
        Save the object as a pickle file for later use.
        I don't recommend that you use this. Rather just process the data
        in one go.
        """

        import gzip

        with gzip.open(file_name, 'wb', compresslevel=complevel) as file_object:
            pickle.dump(self, file_object)
        return "SeaGlider object saved to: {}".format(file_name)

    def load_multiple_vars(self, keys):
        """
        Pass a list of keys that will be imported simultaneously rather
        than one by one if accessed using the variable objects themselves.

        These can then be accessed either by the variable objects or
        by .data[<dimension_name>]
        """
        load_dict = {k: [] for k in self.dims}
        for d in load_dict:
            for k in keys:
                if k in self.dims[d]:
                    load_dict[d] += k,
        keys = list(load_dict.keys())

        has_data = []
        for k in keys:
            if len(load_dict[k]) == 0:
                continue
            else:
                has_data += k,
                d0 = load_dict[k][0]
                v0 = self.vars.__dict__[d0]
                if hasattr(v0, 'coords'):
                    load_dict[k] += v0.coords
                print('Dimension: {}\n\t{}'.format(k, str(load_dict[k])).replace("'", ""))
                time.sleep(0.2)

                df = v0._read_nc_files(self.files, load_dict[k])
                if type(v0) is DiveVariable:
                    df = v0._process_coords(df, self.files[0])

                for col in df:
                    self.data[k][col] = df[col]

        if len(has_data) == 1:
            return self.data[has_data[0]]
        else:
            return self.data


class VariableDisplay:

    def _var_dataframe_(self):

        var = {}
        for key in self.__dict__:
            obj = self.__dict__[key]
            var[key] = dct = {}
            dct['Variable'] = obj.name[:24] + '...' if len(obj.name) > 24 else obj.name
            if hasattr(obj, 'dims'):
                dct['Dimension'] = obj.dims[0]
                dct['Loaded'] = True if key in obj.__data__ else False
                # dct['Coordinates'] = obj.coords
            else:
                dct['Dimension'] = "string"
                dct['Loaded'] = True if key in obj.__data__ else False
                # dct['Coordinates'] = ""
        df = pd.DataFrame.from_dict(var, orient='index')
        df = df.set_index('Variable')
        df = df.sort_values(by=['Loaded', 'Dimension'], ascending=False)

        return df

    def __repr__(self):
        df = self._var_dataframe_()

        return str(df)

    def _repr_html_(self):
        df = self._var_dataframe_()
        obj = self.__dict__[df.index[0]]

        html = u"""
        <h3>SeaGlider variables</h3>
        <p style="font-family: monospace">
            Data Path: {fname}<br>
            Number of variables: {num_vars:>16}<br>
        </p>
        <hr style="max-width:35%; margin-left:0px">
        """.format(
            fname='.../' + '/'.join(obj.files[0].split('/')[-3:-1]) + '/*.nc',
            num_vars=str(df.shape[0]),
        )
        html += df.to_html()

        return html


class DiveVariable(object):

    def __init__(self, name, files, df=None, parent=None):
        nco = Dataset(files[0])

        self.__data__ = pd.DataFrame() if df is None else df
        self.__parent__ = parent

        self.files = np.sort(files)
        self.name = name
        self.attrs = dict(nco.variables[name].__dict__)
        self.dims = getattr(nco.variables[name], 'dimensions')
        self.coords = []
        if 'coordinates' in self.attrs:
            self.coords += self.attrs['coordinates'].split()
        self._size_same = nco.variables[name].size

        nco.close()

    def __getitem__(self, key):
        data = self.load(return_data=True)
        return data.loc[key]

    def __repr__(self):
        is_loaded = True if self.__data__.size > 0 else False

        string = ""
        string += "=" * 70
        string += "\nVariable:        {: <30}".format(self.name)
        string += "\nNumber of Files: {: <30}".format(self.files.size)
        string += "\nDimensions:      {}".format(list(self.dims))
        string += "\nCoordinates:     {}".format(list(self.coords))
        string += "\nData:            "
        string += "{} measurements in `.data` in pd.DataFrame format".format(self.__data__.shape[0]) if is_loaded else "Data is not loaded"
        string += "\nAttributes:"
        for key in self.attrs:
            string += "\n\t\t {}: {}".format(key, self.attrs[key])

        return string

    @property
    def values(self):
        return self.series.values

    @property
    def data(self):

        return self.load(return_data=True)

    @property
    def series(self):
        self.load()
        return self.__data__.loc[:, self.name]

    def load(self, return_data=False):
        # neaten up the script by creating labels
        data = self.__data__
        keys = np.unique(self.coords + [self.name])
        files = self.files

        # get keys not in dataframe
        missing = [k for k in filter(lambda k: k not in data, keys)]

        if any(missing):
            df = self._read_nc_files(files, missing)
            # process coordinates - if no coordinates, just loops through
            df = self._process_coords(df, files[0])
            for col in df:
                self.__data__[col] = df[col]
        setattr(self.__parent__, self.name, self)

        if return_data:
            return data.loc[:, keys]

    def _read_nc_files(self, files, keys):
        from tqdm import trange

        if 'dives' in keys:
            dives = True
            keys.remove('dives')
        else:
            dives = False

        data = []
        error = ''
        if self.__parent__.verbose:
            pbar = trange(files.size)
        else:
            pbar = range(files.size)
        for i in pbar:
            fname = files[i]
            nc = Dataset(fname)

            nc_keys = [k for k in filter(lambda k: k in nc.variables, keys)]
            if nc_keys:
                skipped = set(keys) - set(nc_keys)
                if skipped:
                    error += '{} not in {}\n'.format(str(skipped), os.path.split(fname)[1])
                arr = np.r_[[nc.variables[k][:] for k in nc_keys]]

                dives = np.ones([1, nc.variables[nc_keys[0]].size]) * i
                meas_idx = np.arange(nc.variables[nc_keys[0]].size)[None]
                arr = np.r_[arr, dives, meas_idx]
                nc.close()

                cols = nc_keys + ['dives', 'meas_id']
                df = pd.DataFrame(arr.T, columns=cols)

                data += df,
            else:
                error += '{} was skipped\n'.format(fname)

        if len(error) > 0:
            print(error)
        data = pd.concat(data, ignore_index=True)

        return data

    def _process_coords(self, df, reference_file_name):

        # if ('dives' in self.__data__):
        #     df = df.drop(columns='dives')

        # TRY TO GET DEPTH AND TIME COORDS AUTOMATICALLY
        for col in df.columns:
            # DECODING TIMES IF PRESENT
            if ('time' in col.lower()) | ('_secs' in col.lower()):
                time = col
                self.__data__.time_name = time
                nco = Dataset(reference_file_name)
                units = nco.variables[time].getncattr('units')
                df[time + '_raw'] = df.loc[:, time].copy()
                if 'seconds since 1970' in units:
                    df[time] = df.loc[:, time].astype('datetime64[s]')
                else:
                    from xarray.coding.times import decode_cf_datetime
                    df[time] = decode_cf_datetime(df.loc[:, time], units)
                nco.close()

            # CREATE UPCAST COLUMN
            # previously I changed the dive number to be 0.5 if upcast,
            # but this ran into indexing problems when importing columns
            # after the inital import (where depth wasn't present).
            if ('depth' in col.lower()):
                depth = df[col].values
                dives = df.dives.values
                self.__data__.depth_name = col
                # INDEX UP AND DOWN DIVES
                updive = np.ndarray(dives.size, dtype=bool) * False
                for d in np.unique(dives):
                    i = d == dives
                    j = np.argmax(depth[i])
                    # bool slice of the dive
                    k = i[i]
                    # make False until the maximum depth
                    k[:j] = False
                    # assign the bool slice to the updive
                    updive[i] = k

                df['dives'] = dives + (updive / 2)

        return df


class PointVariable:
    def __init__(self, name, files, df=None, parent=None):

        nco = Dataset(files[0])
        self.files = np.sort(files)
        self.__data__ = pd.DataFrame() if df is None else df
        self.__parent__ = parent
        self.name = name
        self.attrs = dict(nco.variables[name].__dict__)
        nco.close()

    @property
    def data(self):

        return self.load(return_data=True)

    @property
    def series(self):

        return self.data.loc[:, self.name]

    @property
    def values(self):

        return self.series.values

    def load(self, return_data=False):

        data = self.__data__
        keys = [self.name]
        files = self.files
        missing = [k for k in filter(lambda k: k not in data, keys)]

        if any(missing):
            df = self._read_nc_files(files, missing)
            try:
                df = df.astype(float)
            except ValueError:
                pass
            for col in df:
                self.__data__[col] = df[col]

        setattr(self.__parent__, self.name, self)
        if return_data:
            return data[[self.name]]

    def __repr__(self):
        is_loaded = True if self.name in self.__data__ else False

        string = ""

        string += "=" * 70
        string += "\nVariable:        {: <30}".format(self.name)
        string += "\nNumber of Files: {: <30}".format(self.files.size)
        string += "\nData:            "
        string += "{} measurements in `.data` in pd.DataFrame format".format(self.data.shape[0]) if is_loaded else "Data is not loaded"
        string += "\nAttributes:"
        if self.attrs == {}:
            string += "      No attributes for variable"
        for key in self.attrs:
            string += "\n\t\t {}: {}".format(key, self.attrs[key])

        return string

    def _read_nc_files(self, files, keys):
        from tqdm import trange

        if type(keys) is str:
            keys = [keys]
        data = []
        idx = []

        for i in trange(files.size):
            fname = files[i]
            nc = Dataset(fname)
            arr = np.r_[[nc.variables[k][:].squeeze() for k in keys]]
            nc.close()
            data += arr,
            idx += i,
        df = pd.DataFrame(np.array(data), index=idx, columns=keys)

        return df


def _load(file_name):
    """
    Load a saved session - just pass the pickled file's name.
    """

    import gzip

    with gzip.open(file_name, 'rb') as file_object:
        sg = pickle.load(file_object)
    return sg
