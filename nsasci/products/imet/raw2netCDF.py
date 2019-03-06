import pandas as pd
import xarray as xr
import nsasci.products.imet.raw as raw

def generate_name(content):
    base = 'olitbsimet'

    dt = pd.to_datetime(content['start'])
    dt = '{}{:02d}{:02d}.{:02d}{:02d}{:02d}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)

    popssn = 'popssn{}'.format(content['popssn'])

    name_new = '.'.join([base, dt, popssn, 'nc'])
    return name_new


def create_dataset(content):
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
    ds.attrs['info'] = ("v0.1 - This product unifys the iMet raw files (whenever POPS was involved).")
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