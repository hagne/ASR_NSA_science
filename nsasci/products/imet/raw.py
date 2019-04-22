import pandas as pd
import numpy as np


def read_file(path, check4 = 'POPS'):
    assert(check4 in path.name)
    if path.suffix == '.xlsx':
        out = read_xlsx(path)
    elif path.suffix == '.csv':
        out = read_csv(path)
    else:
        raise ValueError("can't be")
    return out

def read_csv(path, drop_labels=['date [y-m-d GMT]', 'time [h:m:s GMT]', 'milliseconds', 'seconds since midnight [GMT]',
                                'elapsed minutes'],
             verbose = False):
    # assert ('POPS' in path.name)

    def read_header(path):
        with path.open() as rein:
            line = rein.readline()
            if 'NOAA' not in line:
                return None
            #         continue
            i = 0
            while 'date [y-m-d GMT]' not in line:
                line = rein.readline()
                i += 1
                if i > 30:
                    print('did not find anything. current line:')
                    print(line)
                    assert (False)
        #         print('{})'.format(i), end = '\t')
        #         print(rein.readline())
        out = {}
        out['lines'] = i
        if verbose:
            print()
        return out

    def read_data(path, header_lines):
        df = pd.read_csv(path, skiprows=header_lines, usecols = range(27))  # i

        # missing data
        df[df == 99999] = np.nan

        # remove spaces from column names
        df.columns = [i.strip() for i in df.columns]

        # create time stamp
        df.index = pd.to_datetime(df['date [y-m-d GMT]']) + pd.to_timedelta(df['time [h:m:s GMT]'])

        # drop some labels
        return df

    header = read_header(path)
    df = read_data(path, header['lines'])
    df.sort_index(inplace=True)
    # drop labels
    df.drop(drop_labels, axis=1, inplace=True)

    out = {}
    out['data'] = df
    dt = df.index[0]
    # out['dt'] = dt
    out['date'] = '{}-{:02d}-{:02d}'.format(dt.year, dt.month, dt.day)
    start, end = [str(dt) for dt in df.index[[0, -1]]]
    out['start'] = start
    out['end'] = end

    splitt = path.name.split('_')
    #     print(splitt)
    if len(splitt) == 6:
        flightno = int(splitt[-2])
        popssn = splitt[-3][-2:]

    elif len(splitt) == 5:
        flightno = int(splitt[-1].split('.')[0])
        popssn = splitt[-2][-2:]

    elif len(splitt) == 4:
        flightno = 1
        if 'SSN' in splitt[-1]:
            popssn = splitt[-1].split('.')[-2][-2:]
        else:
            popssn = '00'
    else:
        raise ValueError('should not happen')
    #     print(popssn)
    #     print(flightno)

    out['flight_id'] = '{:02d}{:02d}{:02d}.{:02d}'.format(dt.year - 2000, dt.month, dt.day, flightno)

    out['popssn'] = popssn
    #      = flight_id
    #     if verbose:
    #         print('done')
    #####
    return out


def read_xlsx(fn, drop_labels = ['date [y-m-d GMT]', 'time [h:m:s GMT]', 'milliseconds', 'seconds since midnight [GMT]', 'elapsed minutes'],
                   verbose = False):
    assert('POPS' in fn.name)

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

    splitt = fn.name.split('_')
    assert (len(splitt) == 6)
    if 'SSN' not in splitt[-3]:
        popssn = '00'
    else:
        popssn= splitt[-3][-2:]
    out['popssn'] = popssn

    if verbose:
        print('done')
    #####

    return out