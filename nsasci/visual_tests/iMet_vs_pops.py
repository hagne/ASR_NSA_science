from atmPy.aerosols.instruments import POPS
# import icarus
import pathlib

import numpy as np
import xarray as xr
import pandas as pd

from ipywidgets import widgets
from IPython.display import display

import matplotlib.pylab as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

from nsasci import database
import sqlite3



def read_POPS(path):
    # print(path.glob('*'))
    hk = POPS.read_housekeeping(path, pattern = 'hk', skip_histogram= True)
    hk.get_altitude()
    return hk

def read_iMet(path):
    ds = xr.open_dataset(path)
    # return ds
    alt_gps = ds['GPS altitude [km]'].to_pandas()
    alt_bar = ds['altitude (from iMet PTU) [km]'].to_pandas()

    df = pd.DataFrame({'alt_gps': alt_gps,
              'alt_bar': alt_bar})
    return df

class Data_container(object):
    def __init__(self, controller, path2data1, path2data2):
        self.controller = controller
        self.delta_t = 0
        path2data1 = pathlib.Path(path2data1)
        read_data = read_iMet #icarus.icarus_lab.read_imet
        self.dataset1 = Data(self, path2data1, read_data, glob_pattern = 'oli*')

        path2data2 = pathlib.Path(path2data2)
        read_data = read_POPS
        self.dataset2 = Data(self, path2data2, read_data, glob_pattern='olitbspops*')
#         path2data2 = Path(path2data2)

class Data(object):
    def __init__(self, datacontainer, path2data, read_data, glob_pattern = '*'):
        self.controller = datacontainer.controller
        self._datacontainer = datacontainer
        self.read_data = read_data
        self.path2data = path2data
        self._path2active = None


        self.path2data_list = sorted(list(path2data.glob(glob_pattern)))
        self.path2active = self.path2data_list[0]

    @property
    def path2active(self):
        return self._path2active

    @path2active.setter
    def path2active(self, value):
        self._datacontainer.delta_t = 0
        self._path2active = value
        self.controller.send_message('opening {}'.format(self._path2active.name))
        # print(self._path2active.name)
        self.active = self.read_data(self._path2active)

    def previous(self):
        idx = self.path2data_list.index(self.path2active)
        if idx == 0:
            print('first')
            pass
        elif idx == -1:
            raise ValueError('not possibel')
        else:
            self.path2active = self.path2data_list[idx - 1]

    def next(self):
        idx = self.path2data_list.index(self.path2active)
        if idx == len(self.path2data_list) - 1:
            print('last')
            pass
        elif idx == -1:
            raise ValueError('not possibel')
        else:
            self.path2active = self.path2data_list[idx + 1]


class View(object):
    def __init__(self, controller):
        self.controller = controller

        self.plot = Plot(self)
        self.controlls = Controlls(self)

# View
class Plot(object):
    def __init__(self, view):
        self.view = view
        self.controller = view.controller
        self.f = None
        self.a = None
        self.at = None

    def initiate(self):
        self.f, self.a = plt.subplots()
        self.f.autofmt_xdate()

        self.at = self.a.twinx()

        self.plot_active_d1()
        self.plot_active_d2()
        self.update_xlim()
        return self.a, self.at

    def plot_active_d1(self):
        # self.controller.data.dataset1.active.data['altitude (from iMet PTU) [km]'].plot(ax = self.a, label = 'altitude (from iMet PTU) [km]')
        # self.controller.data.dataset1.active.data['GPS altitude [km]'].plot(ax = self.a, label = 'GPS altitude [km]')
        self.controller.data.dataset1.active.plot(ax = self.a)
        self.a.legend(loc = 2)

    def update_1(self):
        self.a.clear()
        self.plot_active_d1()
        self.update_xlim()

    def plot_active_d2(self):
        self.controller.data.dataset2.active.data.Altitude.plot(ax = self.at, color = colors[2])
        self.at.legend(loc = 1)

    def update_2(self, keep_limits = False):
        if isinstance(self.at, type(None)):
            return

        xlim = self.at.get_xlim()
        ylim = self.at.get_ylim()

        self.at.clear()
        self.plot_active_d2()
        if not keep_limits:
            self.update_xlim()
        else:
            self.at.set_xlim(xlim)
            self.at.set_ylim(ylim)

    def update_xlim(self):
        xmin = np.min([self.controller.data.dataset1.active.index.min(), self.controller.data.dataset2.active.data.index.min()])
        xmax = np.max([self.controller.data.dataset1.active.index.max(), self.controller.data.dataset2.active.data.index.max()])
        self.a.set_xlim(xmin, xmax)


class Controlls(object):
    def __init__(self, view):
        self.view = view
        self.controller = view.controller

    def initiate(self):

        tab_children = []
        ###########################
        # data 1 box
        d1_vbox_childs = []
        ##
        ###
        d1_button_next = widgets.Button(description='next measurement')
        d1_button_prev = widgets.Button(description='prev measurement')

        d1_button_next.on_click(self.on_d1_botton_next)
        d1_button_prev.on_click(self.on_d1_botton_prev)

        d1_box_h_1 = widgets.HBox([d1_button_prev, d1_button_next])
        ###
        d1_vbox_childs.append(d1_box_h_1)

        ##
        ###
        d1_text_path = widgets.Text(placeholder='path name', disabled = False)
        self.d1_text_path = d1_text_path
        d1_vbox_childs.append(d1_text_path)

        ##
        d1_vbox = widgets.VBox(d1_vbox_childs)
        tab_children.append({'element': d1_vbox, 'title': 'iMet'})

        ############################
        # data 2 box
        d2_vbox_childs = []
        ##
        ###
        d2_button_next = widgets.Button(description='next measurement')
        d2_button_prev = widgets.Button(description='prev measurement')
        self.d2_dropdown_fnames = widgets.Dropdown(options=[i.name for i in self.controller.data.dataset2.path2data_list],
                                              value=self.controller.data.dataset2.path2active.name,
                                            #     description='N',
                                                disabled=False,
                                            )

        d2_button_next.on_click(self.on_d2_botton_next)
        d2_button_prev.on_click(self.on_d2_botton_prev)
        self.d2_dropdown_fnames.observe(self.on_change_d2_dropdown_fnames)

        d2_box_h_1 = widgets.HBox([d2_button_prev, d2_button_next, self.d2_dropdown_fnames])
        ###
        d2_vbox_childs.append(d2_box_h_1)

        ##
        ###
        # text field showing the path
        d2_text_path = widgets.Text(placeholder='path name', disabled = False)
        self.d2_text_path = d2_text_path

        d2_vbox_childs.append(d2_text_path)

        ##
        d2_vbox = widgets.VBox(d2_vbox_childs)
        tab_children.append({'element': d2_vbox, 'title': 'POPS'})

        # others box



        # Tab
        tab = widgets.Tab([child['element'] for child in tab_children])
        for e ,child in enumerate(tab_children):
            tab.set_title(e ,child['title'])

        # accordeon

        self.accordeon_assigned = widgets.Valid(value=False,
                                                description='bound?',
                                                )

        self.dropdown_popssn= widgets.Dropdown(options=['00', '14', '18'],
                                                    # value='2',
                                                    description='popssn',
                                                    disabled=False,
                                                    )

        self.inttext_deltat = widgets.IntText(value=0,
                                              description='deltat',
                                              disabled=False
                                              )
        self.inttext_deltat.observe(self.on_inttext_deltat)

        self.dropdown_gps_bar_bad = widgets.Dropdown(
            options=['gps', 'baro', 'bad', 'bad_but_usable_gps', 'bad_but_usable_baro'],
            value='gps',
            description='which alt to use:',
            disabled=False,
            )


        self.button_bind_measurements = widgets.ToggleButton(description = 'bind/unbind measurements')
        # button_bind_measurements.on_click(self.deprecated_on_button_bind_measurements)
        self.button_bind_measurements.observe(self.on_button_bind_measurements)



        accordon_box = widgets.VBox([self.accordeon_assigned, self.dropdown_popssn, self.inttext_deltat, self.dropdown_gps_bar_bad, self.button_bind_measurements])
        accordion_children = [accordon_box]
        accordion = widgets.Accordion(children=accordion_children)
        accordion.set_title(0,'do_stuff')

        # messages
        self.messages = widgets.Textarea('\n'.join(self.controller._message), layout={'width': '100%'})
        # message_box = widgets.HBox([self.messages])
        # OverVbox

        overVbox = widgets.VBox([tab, accordion, self.messages])
        display(overVbox)
        ####################
        self.update_d1()
        self.update_d2()
        self.update_accordeon()

    def on_inttext_deltat(self, evt):
        if evt['name'] == "value":
            self.controller.data.delta_t = int(evt['new'])
            dt = int(evt['new']) - int(evt['old'])
            self.controller.data.dataset2.active.data = self.controller.data.dataset2.active.data.shift(periods=-dt,
                                                                                  freq=pd.to_timedelta(1, 's'))
            self.controller.view.plot.update_2(keep_limits = True)


    def update_d1(self):
        self.d1_text_path.value = self.controller.data.dataset1.path2active.name
        self.dropdown_popssn.value = self.controller.data.dataset1.path2active.name.split('.')[-2][-2:]
        self.inttext_deltat.value = self.controller.data.delta_t

    def on_d1_botton_next(self, evt):
        self.controller.data.dataset1.next()
        self.update_d1()
        self.update_accordeon()
        self.controller.view.plot.update_1()

    def on_d1_botton_prev(self, evt):
        self.controller.data.dataset1.previous()
        self.update_d1()
        self.update_accordeon()
        self.controller.view.plot.update_1()

    def update_d2(self):
        self.d2_text_path.value = self.controller.data.dataset2.path2active.name
        self.d2_dropdown_fnames.value = self.controller.data.dataset2.path2active.name
        self.inttext_deltat.value = self.controller.data.delta_t

    def update_accordeon(self):
        active = self.controller.database.active_set()
        self.button_bind_measurements.unobserve(self.on_button_bind_measurements)
        if active.shape[0] == 1:
            # print('blabla')
            # print(active)
            # print(active.popssn[0])
            self.dropdown_popssn.value = active.popssn[0]
            self.inttext_deltat.value = active.delta_t_s[0]
            self.accordeon_assigned.value = True
            self.button_bind_measurements.value = True
        elif active.shape[0] == 0:
            self.accordeon_assigned.value = False
            self.button_bind_measurements.value = False
            pass
        self.button_bind_measurements.observe(self.on_button_bind_measurements)


    def on_change_d2_dropdown_fnames(self, change):
        # self.controller.test = change
        # print(change)
        if change['type'] == 'change' and change['name'] == 'value':
            # print("changed to %s" % change['new'])
            base = self.controller.data.dataset2.path2data
            # self.controller.data.dataset2.active = base.joinpath(change['new'])
            self.controller.data.dataset2.path2active = base.joinpath(change['new'])
            # self.update_d2()
            self.update_accordeon()
            self.d2_text_path.value = self.controller.data.dataset2.path2active.name
            self.controller.view.plot.update_2()


    def on_d2_botton_next(self, evt):
        self.controller.data.dataset2.next()
        self.update_d2()
        self.update_accordeon()
        self.controller.view.plot.update_2()

    def on_d2_botton_prev(self, evt):
        self.controller.data.dataset2.previous()
        self.update_d2()
        self.update_accordeon()
        self.controller.view.plot.update_2()

    def deprecated_on_button_bind_measurements(self, evt):
        self.controller.database.bind_measurements()

    def on_button_bind_measurements(self, evt):
        if evt['name'] == 'value':
            if evt['new'] == True:
                self.controller.database.bind_measurements()
            if evt['new'] == False:
                self.controller.database.unbind_measurements()
            self.update_accordeon()
        # print('baustelle')

class Database(database.NsaSciDatabase):
    def __init__(self, controller, path2db):
        # super().__init__(path2db)
        self.path2db = path2db
        self.controller = controller
        self.tbl_name = 'match_datasets_imet_pops'

    def add_active_set(self):
        imet_active_name = self.controller.data.dataset1.path2active.name
        pops_active_name = self.controller.data.dataset2.path2active.name

        # # test if exists
        # qu = 'select * from match_datasets_imet_pops WHERE fn_imet="{}"'.format(imet_active_name)
        # # print('imet: {}'.format(qu))
        # imet_exists = pd.read_sql(qu, self.db).shape[0]
        # qu = 'select * from match_datasets_imet_pops WHERE fn_pops="{}"'.format(pops_active_name)
        # # print('pops: {}'.format(qu))
        # pops_exists = pd.read_sql(qu, self.db).shape[0]
        # if imet_exists:
        #     self.controller.send_message('ERROR: iMet file was assigned before')
        #     return
        # elif pops_exists:
        #     self.controller.send_message('ERROR: POPS file was assigned before')
        #     return
        #
        # # get next index
        # qu = 'select Max(idx) from match_datasets_imet_pops'
        # next_idx = pd.read_sql(qu, self.db).iloc[0, 0]
        # if not next_idx:
        #     next_idx = 1
        # else:
        #     next_idx = int(next_idx) + 1
        #
        # #bind
        dic = dict(fn_imet=imet_active_name,
                   fn_pops=pops_active_name,
                   popssn=self.controller.view.controlls.dropdown_popssn.value,
                   delta_t_s= self.controller.data.delta_t,
                   which_alt = self.controller.view.controlls.dropdown_gps_bar_bad.value
                   )
        with sqlite3.connect(self.path2db) as db:
            # get next index
            # qu = 'select Max(idx) from match_datasets_imet_pops'
            # next_idx = pd.read_sql(qu, db).iloc[0, 0]
            # if not next_idx:
            #     next_idx = 1
            # else:
            #     next_idx = int(next_idx) + 1
            qu = 'select id from match_datasets_imet_pops'
            next_idx = pd.read_sql(qu, db)  # .iloc[0, 0]
            next_idx = next_idx.astype(int).max().iloc[0]
            # print(next_idx)
            # print(type(next_idx))
            # print(type(next_idx).__name__)
            # print(next_idx == np.nan)
            # print(next_idx == np.float64(np.nan))
            # print(float(next_idx) == float(np.nan))
            # print(np.float64(next_idx) == np.float64(np.nan))
            if not next_idx:
                next_idx = 1
            elif np.isnan(next_idx):
                next_idx = 1
            else:
                next_idx = int(next_idx) + 1


            df = pd.DataFrame(dic, index=[next_idx])
            df.index.name = 'id'

            # self.add_line2db(df, 'match_datasets_imet_pops')
            table_name = 'match_datasets_imet_pops'

            df.to_sql(table_name, db,
                    #                  if_exists='replace'
                    if_exists='append'
                    )

    def update_values(self):
        qu = """UPDATE match_datasets_imet_pops
                SET
                popssn={popssn},
                delta_t_s={delta_t_s}
                WHERE
                fn_imet="{fn_imet}"
                AND
                fn_pops="{fn_pops}"
                """.format(popssn=self.controller.view.controlls.dropdown_popssn.value,
                           delta_t_s = self.controller.view.controlls.inttext_deltat.value,
                           fn_imet = self.controller.data.dataset1.path2active.name,
                           fn_pops = self.controller.data.dataset2.path2active.name)
        with sqlite3.connect(self.path2db) as db:
            db.execute(qu)

    def active_set(self):
        with sqlite3.connect(self.path2db) as db:
            qu = '''select * from match_datasets_imet_pops 
            where 
            fn_imet="{}"
            and
            fn_pops="{}"
            '''.format(self.controller.data.dataset1.path2active.name, self.controller.data.dataset2.path2active.name)
            out = pd.read_sql(qu, db)
            # out = db.execute(qu).fetchall()
        return out

    def bind_measurements(self):
        if self.active_set().shape[0] == 0:
            self.add_active_set()
        elif self.active_set().shape[0] == 1:
            self.update_values()

    def unbind_measurements(self):
        with sqlite3.connect(self.path2db) as db:
            qu = 'DELETE from {} WHERE id = {}'.format(self.tbl_name, self.active_set().id[0])
            db.execute(qu)


class Controller(object):
    def __init__(self,
                 path2data1,
                 path2data2,
                 path2database):
        self._message = []
        self.data = Data_container(self, path2data1, path2data2)
        self.view = View(self)
        self.database = Database(self, path2database)


    def send_message(self, txt):
        # print(txt)
        # self._message +=self._message + '\n' + txt
        self._message.append(txt)
        if len(self._message) > 10:
            self._message = self._message[-10:]
        try:
            mt = list(reversed(self._message))
            self.view.controlls.messages.value = '\n'.join(mt)
        except AttributeError:
            pass