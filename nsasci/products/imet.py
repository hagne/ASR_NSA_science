import pandas as pd
import numpy as np

def read_iMet_xlsx(fn, drop_labels = ['date [y-m-d GMT]', 'time [h:m:s GMT]', 'milliseconds', 'seconds since midnight [GMT]', 'elapsed minutes'],
                   verbose = False):
    if verbose:
        print('Reading file {}'.format(fn.name), end = '...')
    out = {}
    xf_head = pd.read_excel(fn ,nrows=30,
                            #               header=17
                            )

    xf_data_idx = int(xf_head[xf_head.iloc[: ,0] == 'date [y-m-d GMT]'].index.values) + 1
    #     print(xf_data_idx)
    xf_data = pd.read_excel(fn, header=xf_data_idx)

    xf_data.columns = [col.strip() for col in xf_data.columns]
    #     return xf_data
    xf_data.index = xf_data['date [y-m-d GMT]'] + pd.to_timedelta(xf_data['time [h:m:s GMT]'])
    xf_data[xf_data==99999] = np.nan

    dt = xf_data['date [y-m-d GMT]'].iloc[0]
    flight_date = '{}-{:02d}-{:02d}'.format(dt.year, dt.month, dt.day)

    dtn = pd.to_datetime(fn.name.split('_')[1], format = '%m%d%y')
    flight_date_n = '{}-{:02d}-{:02d}'.format(dtn.year, dtn.month, dtn.day)

    if flight_date != flight_date_n:
        txt = 'Start date in file ({}) is different than indicated in filename({}).'.format(flight_date, fn.name)
        print(txt)
    #         raise ValueError(txt)

    #     out['datedt'] = dt # test... can be deleted
    flight_no_that_day = fn.name.split('.')[-2].split('_')[-2]
    flight_id = '{:02d}{:02d}{:02d}.{:02d}'.format(dt.year - 2000, dt.month, dt.day, int(flight_no_that_day))
    if verbose:
        print('flight_id: {}'.format(flight_id), end = '...')
    start, end = [str(dt) for dt in xf_data.index[[0 ,-1]]]

    # clean-up
    xf_data.drop(drop_labels, axis=1, inplace=True)

    # the output
    out['data'] = xf_data
    out['date'] = flight_date
    out['start'] = start
    out['end'] = end
    out['flight_id'] = flight_id
    if verbose:
        print('done')
    #####

    return out