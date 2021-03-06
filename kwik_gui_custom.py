#!/bin/bash
import numpy as np
from IPython import get_ipython
from os import path
# what previously worked prio to 2/2017
#from phycontrib.kwik_gui import KwikController

# what currently works
from phycontrib.kwik import KwikController
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

# Custom plotting code for phy and klustakwik spike sorter.
# G. Telian
# Adesnik Lab
# UC Berkeley
# 20170207

class kwik_gui_custom(object):

    def __init__(self, kwik_path):

        print('\n-----Initializing kwik-gui-----')

        if path.exists(kwik_path):
            # launch kwik-gui
            try:
                ipython = get_ipython()
                ipython.magic('gui qt')
                c = KwikController(kwik_path)
                gui = c.create_gui()
                gui.show()
                #self.c = c # old way
                self.c = c.model
                self.gui = gui
                self.enable_plot = True
#                f, ax = plt.subplots(1, 2, figsize=(16, 6))
#                self.f = f
#                self.ax = ax
            except:
                raise ValueError('Cannot open gui!--go find Greg!')
        else:
            raise ValueError('Kwik file not found!--make sure the path is correct')

        @self.gui.connect_
        def on_select(cluster_ids):
            self.plot_metrics(cluster_id=cluster_ids[0])

    def plot_metrics(self, cluster_id=None, bin_size=0.0005, max_time=0.100):

        if self.enable_plot is False:
            # do nothing
            None
        else:
            # close any open plots
            #plt.gca()
            #plt.close('all')
            if cluster_id:
                # get indices of specified cluster
                uind = np.where(self.c.spike_clusters == cluster_id)[0]
            else:
                print('no cluster ID was provided')

            if len(uind) ==  0:
                print('cluster ID not found')
            else:
                # get spike times and bin them
                spk_times = self.c.spike_times[uind]
                isi       = np.diff(spk_times)
                bins      = np.arange(0, max_time, bin_size);
                counts    = np.histogram(isi, bins=bins)[0]
                rpvs      = np.where(isi < 0.0015)[0]
                rpv_perc  = len(rpvs)/float(len(isi))*100

                # get waveform snippets and measure amplitude
                samp_perc  = 0.01 # size of sample to draw from unit indices (so things actually plot quickly)
                rand_inds  = np.sort(np.random.choice(uind, int(uind.shape[0]*samp_perc)))
                waves      = self.c.all_waveforms[rand_inds] # 4 seconds
                wave_times = self.c.spike_times[rand_inds] ### USE ORIGINAL SPIKE TIMES
                mean_waves = waves.mean(axis=0) # average waveform for each channel
                best_chan  = np.argmin(np.min(mean_waves, axis=0)) # samples x channels (find channel with largest waveform)
                best_waves = waves[:, :, best_chan] # snippets x samples x channels
                amp        = np.max(best_waves, axis=1) - np.min(best_waves, axis=1)
                wave_drift = np.argmin(np.min(waves, axis=1), axis=1)
                # moving average method from stackoverflow (http://stackoverflow.com/questions/13728392/moving-average-or-running-mean)
                N          = 10
                avg_amp    = np.convolve(amp, np.ones((N,))/N, mode='valid')
                samp_diff  = wave_times.shape[0] - avg_amp.shape[0]
                t_avg_amp  = wave_times[0:-samp_diff]

                rp_height = np.max(counts)*1.1
                try:
                    cluster_group = self.c.cluster_groups[cluster_id]
                except:
                    cluster_group = 'in progress'

                print('cluster ID and group: ' + str(cluster_id) + ', ' + cluster_group + \
                        '\nrbvs: ' + str(len(rpvs)) + '/' + str(len(isi)) + '   ({:2.2f})%'.format(rpv_perc))
#                # prepare figure #                f, ax = plt.subplots(1, 2, figsize=(16, 6)) # isi distribution
#                ax0 = plt.subplot(1, 2, 1)
#                ax0.clear()
#                ax0.bar(bins[:-1], counts, width=bin_size, color='#4f8fff',\
#                        align='center', edgecolor='none')
#                ax0.add_patch(
#                        patches.Rectangle((0.0, 0.0), 0.0015, rp_height,
#                            facecolor='#ff0000', edgecolor='#ff0000', alpha=0.5))
#                plt.suptitle('cluster ID and group: ' + str(cluster_id) + ', ' + cluster_group + \
#                        '\nrbvs: ' + str(len(rpvs)) + '/' + str(len(isi)) + '   ({:2.2f})%'.format(rpv_perc))
#                plt.title('ISI distribution')
#                plt.xlabel('time (s)')
#                plt.ylabel('counts/bin')
#
#                # amplitude and drift vs time
#                ax1 = plt.subplot(1, 2, 2)
#                ax1.clear()
#                ax1.plot(t_avg_amp, avg_amp, color='#4f8fff', linewidth=2)
#                plt.xlabel('time (s)')
#                plt.ylabel('peak-to-peak amplitude')
#                plt.title('amplitude and electrode drift')
#                ax1.set_ylim(0, 150)
#
#                ax2 = self.ax[1].twinx()
#                ax2.clear()
#                ax2.plot(wave_times, wave_drift, color='#ff0000', alpha=0.5)
#                ax2.set_ylabel('electrode contact')
#                ax2.set_ylim(0, 32)
#
#                plt.show()
#                self.f.canvas.draw()


print(sys.argv)
kwik = kwik_gui_custom(sys.argv[1])
