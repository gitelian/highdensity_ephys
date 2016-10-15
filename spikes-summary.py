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
from bokeh.plotting import figure

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
p1 = figure(plot_height=400, plot_width=600, title="initialized",
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

## FIGURE 4 spike waveforms
s4 = ColumnDataSource(data=dict(x=x, y=y))
p4 = figure(plot_height=400, plot_width=600, title="spike waveforms",
              tools="crosshair, box_zoom, pan,reset,save,wheel_zoom")
p4.line('x', 'y', source=s4, line_width=3, line_alpha=0.6)
p4.xaxis.axis_label = 'time (sec)'
p4.yaxis.axis_label = 'voltage'

##################################################################

# Set up widgets
## select box
sel_exp = Select(title='Spikes files:', value=spike_files[0], options=spike_files)
sel_unit = Select(title='Unit:', value='0', options=['a', 'b'])

##################################################################

# Set up callbacks
def update_title(attrname, old, new):
    os.chdir('/Users/Greg/Desktop/spikes/')
    p1.title.text = 'Loading ' + sel_exp.value
    fpath = os.path.join(os.path.realpath('.'), sel_exp.value[2:])
    labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit,\
            unit_type, trial_times, unwrapped = load_spike_file(fpath)
    p1.title.text = os.path.basename(fpath) + '  Loaded and ready to go'
    gu_inds   = np.where(np.logical_and(labels[:, 1] > 0, labels[:, 1] < 3) == True)[0]
    gu_ids    = labels[gu_inds, 0]
    global labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type, trial_times, unwrapped
    global gu_ids
    update_unit_select(attrname, old, new)
sel_exp.on_change('value', update_title)

def update_figure01(attrname, old, new):
    # default values
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
    wave_inds   = np.random.choice(np.ravel(np.where(assigns == gu_ids[uind])[0]), size=1000, replace=False, p=None)
    wave_select = waves[wave_inds, :, :]
    mean_wave   = np.mean(wave_select, axis=0)
    semm_wave   = sp.stats.sem(wave_select, axis=0)

    x = np.arange(0, mean_wave.shape[0])/30000.0
    y = mean_wave
    max_wave = mean_wave[:, np.argmin(np.min(mean_wave, axis=0))]
    yerr = semm_wave[:, np.argmin(np.min(mean_wave, axis=0))]
    offset = np.arange(0, mean_wave.shape[0])

    s4.data = dict(x=x, y=max_wave)

 # Define Bollinger Bands.
    # upperband = max_wave + yerr
    # lowerband = max_wave - yerr

    # # Bollinger shading glyph:
    # band_x = np.append(x, x[::-1])
    # band_y = np.append(lowerband, upperband[::-1])
    # p4.patch(band_x, band_y, color='#7570B3', fill_alpha=0.2)
    # doesn't clear previous error shading

def update_unit_select(attrname, old, new):
    gu_ids_list = [str(int(x)) for x in gu_ids]
    sel_unit.options = gu_ids_list
    uind = gu_ids_list.index(str(sel_unit.value))
    global uind
    update_figure01(attrname, old, new)
    update_figure02(attrname, old, new)
    update_figure03(attrname, old, new)
    update_figure04(attrname, old, new)

sel_unit.on_change('value', update_unit_select)

# Set up layouts and add to document
inputs = widgetbox(sel_exp, sel_unit)

#curdoc().add_root(row(inputs, p1, p2, p3, p4, width=800))
l = layout([
  [p1, p3],
  [p2, p4],
  [inputs],
], sizing_mode='fixed') #fixed
curdoc().add_root(l)
curdoc().title = "spikes-summary"
















