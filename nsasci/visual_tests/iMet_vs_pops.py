from atmPy.aerosols.instruments import POPS
import icarus
import pathlib

import numpy as np
import xarray as xr
import pandas as pd

from ipywidgets import widgets
from IPython.display import display

import matplotlib.pylab as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']




def read_POPS(path):
    # print(path.glob('*'))
    hk = POPS.read_housekeeping(path, pattern = 'hk')
    hk.get_altitude()
    return hk

def read_iMet(path):
    ds = xr.open_dataset(path)

    alt_gps = ds['GPS altitude [km]'].to_pandas()
    alt_bar = ds['GPS altitude [km]'].to_pandas()

    df = pd.DataFrame({'alt_gps': alt_gps,
              'alt_bar': alt_bar})
    return df

class Data_container(object):
    def __init__(self, controller, path2data1, path2data2):
        self.controller = controller

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

    def update_2(self):
        self.at.clear()
        self.plot_active_d2()
        self.update_xlim()

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

        # display(tab)

        # OverVbox
        self.messages = widgets.Textarea('\n'.join(self.controller._message))
        overVbox = widgets.VBox([tab, self.messages])
        display(overVbox)
        ####################
        self.update_d1()
        self.update_d2()

    def update_d1(self):
        self.d1_text_path.value = self.controller.data.dataset1.path2active.name

    def on_d1_botton_next(self, evt):
        self.controller.data.dataset1.next()
        self.update_d1()
        self.controller.view.plot.update_1()

    def on_d1_botton_prev(self, evt):
        self.controller.data.dataset1.previous()
        self.update_d1()
        self.controller.view.plot.update_1()

    def update_d2(self):
        self.d2_text_path.value = self.controller.data.dataset2.path2active.name
        self.d2_dropdown_fnames.value = self.controller.data.dataset2.path2active.name

    def on_change_d2_dropdown_fnames(self, change):
        # self.controller.test = change
        # print(change)
        if change['type'] == 'change' and change['name'] == 'value':
            # print("changed to %s" % change['new'])
            base = self.controller.data.dataset2.path2data
            # self.controller.data.dataset2.active = base.joinpath(change['new'])
            self.controller.data.dataset2.path2active = base.joinpath(change['new'])
            # self.update_d2()
            self.d2_text_path.value = self.controller.data.dataset2.path2active.name
            self.controller.view.plot.update_2()


    def on_d2_botton_next(self, evt):
        self.controller.data.dataset2.next()
        self.update_d2()
        self.controller.view.plot.update_2()

    def on_d2_botton_prev(self, evt):
        self.controller.data.dataset2.previous()
        self.update_d2()
        self.controller.view.plot.update_2()


class Controller(object):
    def __init__(self,
                 path2data1,
                 path2data2):
        self._message = []
        self.data = Data_container(self, path2data1, path2data2)
        self.view = View(self)


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