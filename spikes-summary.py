import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as sio
import glob
import os
import h5py
from sklearn.cluster import KMeans
from warnings import warn

from bokeh.io import curdoc, gridplot
from bokeh.layouts import row, widgetbox, layout
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, Select
from bokeh.plotting import figure, show
from bokeh.models.glyphs import MultiLine
from bokeh.models import Range1d
# for 2d histogram
from bokeh.models import LogColorMapper, LogTicker, ColorBar, LinearColorMapper
from matplotlib.mlab import bivariate_normal
from bokeh.models import Image


from bokeh.charts import HeatMap, output_file, show
import pandas as pd

#####
max_counts = list()
global max_counts

# how to run this: bokeh serve --show spikes-summary.py

# G. Telian
# Adesnik Lab
# UC Berkeley
# 20161012

def load_spike_file(path):
    """
    Loads spikes file from specified path
    """
    print('\n\n##### LOADING #####')
    mat  = h5py.File(path)
    spks = mat['spikes']
    assigns     = np.asarray(spks['assigns']).T        # ndarray shape(n) n: num of spike times for all units
    trials      = np.asarray(spks['trials']).T         # ndarray shape(n) n: num of spike times for all units
    spike_times = np.asarray(spks['spiketimes']).T     # ndarray shape(n) n: num of spike times for all units
    waves       = np.asarray(spks['waveforms']).T      # ndarray shape(n x m x p) n: num of spike times for all units m: num of range(m)les in waveform
    trial_times = np.asarray(spks['trial_times']).T    # p: num of recording channels
    labels      = np.asarray(spks['labels']).T
    unwrapped   = np.asarray(spks['unwrapped_times']).T
    nsamp       = waves.shape[1]
    nchan       = waves.shape[2]  # get number of channels used

    uniqassigns = np.unique(assigns)  # get unique unit numbers
    unit_type   = labels[:, 1]        # get unit type 0=noise, 1=multi-unit, 2=single-unit, 3=unsorted
    ids         = labels[:, 0]
    nunit       = len(ids)

    return labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type, trial_times, unwrapped

## get directory list
os.chdir('/Users/Greg/Desktop/spikes/')
spike_files = list()
for (dirpath, dirnames, filenames) in os.walk('.'):
    glob_list = glob.glob(dirpath + os.path.sep + '*spikes.mat')

    if glob_list:
        spike_files.extend(glob_list)

spike_files = sorted(spike_files)


# Set up plot

## FIGURE 1 experiment firing rate
x = np.arange(10)
y = x**2
s1 = ColumnDataSource(data=dict(left=x, right=x+0.5, top=y, bottom=np.zeros(len(x))))
p1 = figure(plot_height=400, plot_width=600, title="firing rate across entire experiment",
              tools="crosshair, box_zoom, pan,reset,save,wheel_zoom")
p1.quad('left', 'right', 'top', 'bottom', source=s1, color="#4f8fff")
p1.xaxis.axis_label = 'time (min)'
p1.yaxis.axis_label = 'firing rate (Hz)'

## FIGURE 2 autocorrelogram
s2 = ColumnDataSource(data=dict(left=x, right=x+0.5, top=y, bottom=np.zeros(len(x))))
p2 = figure(plot_height=400, plot_width=600, title="autocorrelogram",
              tools="crosshair, box_zoom, pan,reset,save,wheel_zoom")
p2.quad('left', 'right', 'top', 'bottom', source=s2, color="#4f8fff")
# p2.line('x', 'y', source=s2, line_width=3, line_alpha=0.6)
p2.xaxis.axis_label = 'time (sec)'
p2.yaxis.axis_label = 'counts/bin'

## FIGURE 3 ISI distribution
s3 = ColumnDataSource(data=dict(left=x, right=x+0.5, top=y, bottom=np.zeros(len(x))))
p3 = figure(plot_height=400, plot_width=600, title="ISI distribution",
              tools="crosshair, box_zoom, pan,reset,save,wheel_zoom")
p3.quad('left', 'right', 'top', 'bottom', source=s3, color="#4f8fff")
# p3.line('x', 'y', source=s3, line_width=3, line_alpha=0.6)
s33 = ColumnDataSource(data=dict(left=x/2, right=x/2+0.5, top=y/2, bottom=np.zeros(len(x))))
p3.quad('left', 'right', 'top', 'bottom', source=s33, color="#ff0000", alpha=0.5)
p3.xaxis.axis_label = 'time (sec)'
p3.yaxis.axis_label = 'counts/bin'

# FIGURE 4 spike waveforms
s4 = ColumnDataSource(data=dict(x=x, y=y))
p4 = figure(plot_height=400, plot_width=600, title="spike waveforms",
              tools="crosshair, box_zoom, pan,reset,save,wheel_zoom")

xpts = np.array([-.09, -.12, .0, .12,  .09])
ypts = np.array([-.1,   .02, .1, .02, -.1])
s44 = ColumnDataSource(dict(
        xs=[xpts*(1+i/10.0)+xx for i, xx in enumerate(x)],
        ys=[ypts*(1+i/10.0)+yy for i, yy in enumerate(y)],
    )
)
glyph = MultiLine(xs="xs", ys="ys", line_color="darkgrey", line_width=0.5, line_alpha=0.5)
p4.add_glyph(s44, glyph)

p4.line('x', 'y', source=s4, line_width=3, line_alpha=1.0, line_color='#4f8fff')
p4.xaxis.axis_label = 'time (sec)'
p4.yaxis.axis_label = 'voltage'
s444 = ColumnDataSource(data=dict(band_x=x,band_y=y))
p4.patch('band_x', 'band_y', source=s444, color='#4f8fff', fill_alpha=0.1)

# ##### FIGURE 4 2d histogram of spike waveforms
# N = 100
# X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
# Z1 = bivariate_normal(X, Y, 0.1, 0.9, -1.0, 1.0) +  \
#     0.1 * bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
# image = Z1 * 1e6

# color_mapper = LogColorMapper(palette="Viridis256", low=1, high=1e7)

# p4 = figure(x_range=(0,1), y_range=(0,1),plot_height=400, toolbar_location=None)
# p4.image(image=[image], color_mapper=color_mapper,
#            dh=[1.0], dw=[1.0], x=[0], y=[0])

# # p4 = figure(x_range=(0,0.002), y_range=(-20,20), toolbar_location=None)

## FIGURE 5 top label
p5 = figure(plot_height=50, plot_width=600, toolbar_location=None,
    title='Select a spikes file', title_text_font_size='14pt')
p5.title.text_color = "black"
p5.outline_line_color = "black"
p5.background_fill_color = 'black'

## FIGURE 6 post-isi vs pre-isi
s6 = ColumnDataSource(data=dict(x=x, y=y))
p6 = figure(plot_height=400, plot_width=400, title="post-isi vs pre-isi",
              tools="crosshair, box_zoom, pan,reset,save,wheel_zoom")
p6.scatter('x', 'y', source=s6, marker='o', size=0.25,
              line_color="black", fill_color="#4f8fff", alpha=0.25)
p6.xaxis.axis_label = 'pre-isi (msec)'
p6.yaxis.axis_label = 'post-isi (msec)'
p6.x_range = Range1d(start=-10, end=1000)
p6.y_range = Range1d(start=-10, end=1000) 


##################################################################

# Set up widgets
## select box
sel_exp = Select(title='Spikes files:', value=spike_files[0], options=spike_files)
sel_unit = Select(title='Unit:', value='0', options=['a', 'b'])

##################################################################

# Set up callbacks
def update_title(attrname, old, new):
    os.chdir('/Users/Greg/Desktop/spikes/')
    p5.title.text = 'Loading: ' + sel_exp.value[2:]
    p5.border_fill_color = 'red'
    p5.border_fill_alpha = 0.4

    fpath = os.path.join(os.path.realpath('.'), sel_exp.value[2:])


    labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit,\
            unit_type, trial_times, unwrapped = load_spike_file(fpath)
    p5.title.text = 'Loaded and ready to go: ' + os.path.basename(fpath)
    p5.border_fill_color = 'green'
    p5.border_fill_alpha = 0.4

    # p5.title_text_color = 'green'
    gu_inds   = np.where(np.logical_and(labels[:, 1] > 0, labels[:, 1] < 3) == True)[0]
    gu_ids    = labels[gu_inds, 0]
    global labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type, trial_times, unwrapped
    global gu_ids

    update_unit_select(attrname, old, new)
sel_exp.on_change('value', update_title)

def update_figure01(attrname, old, new):
    # default values
    p5.border_fill_color = 'red'
    p5.border_fill_alpha = 0.4
    binsize = 1
    spk_times = unwrapped[np.where(assigns == gu_ids[uind])[0]]/60.0
    bins      = np.arange(0, unwrapped[-1]/(60.0), binsize)
    counts, pltbins = np.histogram(spk_times, bins=bins)
    counts = counts/(60.0)

    s1.data = dict(left=pltbins[1:], right=pltbins[:-1], top=counts, bottom=np.zeros(len(counts)))

def update_figure02(attrname, old, new):
    binsize = 0.001 #seconds
    num_samples = 5000

    spk_times = unwrapped[np.where(assigns == gu_ids[uind])[0]]
    if len(spk_times) < num_samples:
        num_samples = len(spk_times)
    xt = spk_times
    bins = np.arange(-0.050, 0.050, binsize)
    bin0_ind = np.logical_and(bins[:-1] > -binsize, bins[:-1] < binsize)

    counts = np.zeros(bins.shape[0]-1)
    xt_sample = np.random.choice(np.ravel(xt), size=num_samples, replace=False, p=None)
    for x in xt_sample:
         counts += np.histogram(xt, bins + x)[0]
    counts[bin0_ind,] = 0
    # s2.data = dict(x=bins[:-1], y=counts)
    s2.data = dict(left=bins[1:], right=bins[:-1], top=counts, bottom=np.zeros(len(counts)))

def update_figure03(attrname, old, new):
    spk_times = unwrapped[np.where(assigns == gu_ids[uind])[0]]
    isi = np.diff(np.ravel(spk_times))
    rpv = np.sum(isi <= 0.0015)
    binsize = 0.001 #seconds
    bins = np.arange(0, 0.100, binsize)
    counts = np.histogram(isi, bins)[0]
    max_counts = np.max(counts)
    global max_counts

    # s3.data = dict(x=bins[:-1], y=counts)
    s33.data = dict(left=[0], right=[0.0015], top=[max_counts], bottom=[0])
    s3.data = dict(left=bins[1:], right=bins[:-1], top=counts, bottom=np.zeros(len(counts)))
    p3.title.text = 'rbvs: ' + str(rpv) + '/' + str(len(isi)) + '   ({:2.2f})%'.format(rpv/float(len(isi))*100)
    

def update_figure04(attrname, old, new):
    print('fig 4')
    wave_inds   = np.random.choice(np.ravel(np.where(assigns == gu_ids[uind])[0]), size=1000, replace=False, p=None)
    wave_select = waves[wave_inds, :, :]
    mean_wave   = np.mean(wave_select, axis=0)
    std_wave    = np.std(wave_select, axis=0)

    x = np.arange(0, mean_wave.shape[0])/30000.0
    y = mean_wave
    max_wave = mean_wave[:, np.argmin(np.min(mean_wave, axis=0))]
    yerr = std_wave[:, np.argmin(np.min(mean_wave, axis=0))]
    offset = np.arange(0, mean_wave.shape[0])
    print('after offset')
    best_wave_chan_index = np.argmin(np.min(mean_wave, axis=0))
    all_waves = waves[wave_inds, :, best_wave_chan_index] # shoule be a matrix (num waves x time samples)
    xs = [x.tolist() for i in range(all_waves.shape[0])]
    ys = [all_waves[i, :].tolist() for i in range(all_waves.shape[0])]
    
    s44.data = dict(xs=xs, ys=ys)
    s4.data = dict(x=x, y=max_wave)

    # Define Bollinger Error Bands.
    upperband = max_wave + yerr
    lowerband = max_wave - yerr

    # Bollinger shading glyph:
    band_x = np.append(x, x[::-1])
    band_y = np.append(lowerband, upperband[::-1])
    s444.data = dict(band_x=band_x, band_y=band_y)


    # 2d histogram stuff THAT DOESNT FUCKING WORK
    # # DOESNT UPDATE IMAGE!!!
    # print('figure 4 called')
    # N = 100
    # X, Y = np.mgrid[-3:3:complex(0, N), -2:2:complex(0, N)]
    # Z1 = bivariate_normal(X, Y, 0.2, 0.3, 1.0, 1.0) +  \
    #     0.1 * bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    # image = Z1 * 1e6
    # color_mapper = LogColorMapper(palette="Viridis256", low=1, high=1e7)
    # # p4 = figure(x_range=(0,1), y_range=(0,1),plot_height=400, toolbar_location=None)
    # p4.image(image=[image], color_mapper=color_mapper,
    #        dh=[1.0], dw=[1.0], x=[0], y=[0])
    

    # 2d histogram stuff
    # find best waveform
    # print('best_wave_chan_index')
    # best_wave_chan_index = np.argmin(np.min(mean_wave, axis=0))
    # print(best_wave_chan_index)
    # all_waves = waves[wave_inds, :, int(best_wave_chan_index)] # shoule be a matrix (num waves x time samples)
    # x_coords = list()
    # y_coords = list()

    # for i in range(all_waves.shape[0]):
    #     for j in range(all_waves.shape[1]):
    #         x_coords.append(x[j])
    #         y_coords.append(all_waves[i, j])
    # y_min = np.min(all_waves)
    # y_max = np.max(all_waves)
    # print(y_max)
    # y = np.arange(y_min, y_max, 10)
    # print(y)
    # # x is defined above
    # print('2d histogram2d')
    # counts = np.histogram2d(x_coords, y_coords, bins=[x, y], normed=True)[0]
    # mapper = LinearColorMapper(palette="Viridis256", low=0, high=np.max(counts))
    # print('done getting counts')
    # xx = []
    # yy = []
    # color = []
    # zz = []
    # for i in range(counts.shape[0]):
    #     for j in range(counts.shape[1]):
    #         print(counts.shape)
    #         xx.append(x[i])
    #         yy.append(y[j])
    #         count = counts[i][j]
    #         zz.append(count)

    # plt.figure()
    # plt.imshow(counts, interpolation='nearest', aspect='auto')
    # plt.show()

    # source = ColumnDataSource(
    #     data=dict(x=xx, y=yy, z=zz)
    # )
    # # the axis limits are set correctly but nothing appears!!!
    # print('rectangle')#width=10, height=10
    # p4.rect('x', 'y',
    #        source=source,
    #        fill_color={'field': 'z', 'transform': mapper})

    # p4.grid.grid_line_color = 'black'
    # p4.axis.axis_line_color = None
    # p4.axis.major_tick_line_color = None


def update_figure06(attrname, old, new):
    spk_times     = np.ravel(unwrapped[np.where(assigns == gu_ids[uind])[0]])
    num_spk_times = len(spk_times)

    bisi          = np.zeros((num_spk_times-2, 2))
    for k in range(1, num_spk_times - 1):
        t_before     = spk_times[k] - spk_times[k-1]
        t_after      = spk_times[k+1] - spk_times[k]

        bisi[k-1, 0] = t_before
        bisi[k-1, 1] = t_after

    bisi = bisi*1000

    s6.data = dict(x=bisi[:, 0], y=bisi[:, 1])
    p5.border_fill_color = 'white'

def update_unit_select(attrname, old, new):
    gu_ids_list = [str(int(x)) for x in gu_ids]
    sel_unit.options = gu_ids_list
    uind = gu_ids_list.index(str(sel_unit.value))
    global uind
    update_figure01(attrname, old, new)
    update_figure02(attrname, old, new)
    update_figure03(attrname, old, new)
    update_figure04(attrname, old, new)
    update_figure06(attrname, old, new)

sel_unit.on_change('value', update_unit_select)

# Set up layouts and add to document
inputs = widgetbox(sel_exp, sel_unit)

#curdoc().add_root(row(inputs, p1, p2, p3, p4, width=800))
l = layout([
  [p1, p3, p6],
  [p2, p4],
  [p5],
  [inputs],
], sizing_mode='fixed') #fixed
curdoc().add_root(l)
curdoc().title = "spikes-summary"
















