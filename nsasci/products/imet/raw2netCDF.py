import pandas as pd
import xarray as xr
import nsasci.products.imet.raw as raw

version_log = ["v0.1 - This product unifys the iMet raw files (whenever POPS was involved).",
               "v0.2 - 2019-04-22: using the rawfiles directly from the archive instead of the ones from Dari"]

def generate_name(content):
    base = 'olitbsimet'

    dt = pd.to_datetime(content['start'])
    dt = '{}{:02d}{:02d}.{:02d}{:02d}{:02d}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    popssn = 'popssn{}'.format(content['popssn'])

    name_new = '.'.join([base, dt, popssn, 'nc'])
    return name_new


def create_dataset(content, version = 'v0.1'):
    data = content['data'].copy()

    # problem with column names
    # data.columns = [col.replace('[', '').replace(']','') for col in data.columns]
    data.columns = [col.replace('m/s', 'm*s^-1') for col in data.columns]
    data.columns = [col.replace('cc/s', 'cc*s^-1') for col in data.columns]

    data.index.name = 'datetime'

    ds = xr.Dataset(data)

    ds.attrs['instruments'] = ['iMet']
    ds.attrs['product_name'] = 'raw2netCDF'
    ds.attrs['version'] = '0.1'
    ds.attrs['underlying_products'] = ['iMet_raw']
    ds.attrs['flight_id'] = content['flight_id']


    idx = [version in ver for ver in version_log].index(True)
    log = '\n'.join(version_log[:idx + 1])
    ds.attrs['info'] = log
    return ds


def version_0_1(path_in, path_out_base, skip_if_exists = True):
    """
    * name is based on start time (not launch time)
    :param path_in:
    :param path_out_base:
    :param skip_if_exists:
    :return:
    """
    content = raw.read_file(path_in)

    name_new = generate_name(content)
    path_out = path_out_base.joinpath(name_new)
    if path_out.is_file() and skip_if_exists:
        print('\t File exists')
        return None

    ds = create_dataset(content)
    #     path_out

    ds.to_netcdf(path_out)
    return ds


def version_0_2(path_in, path_out_base, skip_if_exists = True):
    """
    * name is based on start time (not launch time)
    :param path_in:
    :param path_out_base:
    :param skip_if_exists:
    :return:
    """
    version = 'v0.2'
    content = raw.read_file(path_in)

    name_new = generate_name(content)
    path_out = path_out_base.joinpath(name_new)
    if path_out.is_file() and skip_if_exists:
        print('\t File exists')
        return None

    ds = create_dataset(content, version = version)
    #     path_out

    ds.to_netcdf(path_out)
    return ds