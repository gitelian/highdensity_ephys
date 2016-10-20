#!/bin/bash
import numpy as np
from IPython import get_ipython
from os import path
from phycontrib.kwik_gui import KwikController
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
                self.c = c
            except:
                raise ValueError('Cannot open gui!--go find Greg!')
        else:
            raise ValueError('Kwik file not found!--make sure the path is correct')

    def plot_metrics(self, cluster_id=None, bin_size=0.0005, max_time=0.100):

        # close any open plots
        plt.close()
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

            # get waveform snippets and measure amplitude TODO TEST ALL THIS!!!
            samp_perc  = 0.1 # size of sample to draw from unit indices (so things actually plot quickly)
            rand_inds  = np.random.choice(uind, int(uind*samp_perc)
            waves      = self.c.all_waveforms[uind[rand_inds]]
            wave_times = spk_times[rand_inds]
            mean_waves = waves.mean(axis=0)
            best_chan  = np.argmin(np.min(mean_waves, axis=0))
            best_waves = waves[:, :, best_chan] # snippets x samples x channels
            amp        = np.max(best_waves, axis=1) - np.min(best_waves, axis=1)
            wave_drift = np.argmin(np.min(waves, axis=1), axis=1) # check this does is measure the drift of the largest min across channels?
            # moving average method from stackoverflow (TODO TEST THIS)
            cumsum     = np.cumsum(numpy.insert(amp, 0, 0))
            N          = 10
            avg_amp    = (cumsum[N:] - cumsum[:-N]) / N

            # plot isi distribution
            rp_height = np.max(counts)*1.1
            cluster_group = self.c.cluster_groups[58]
            f, ax_hist = plt.subplots(1, 1)
            plt.bar(bins[:-1], counts, width=bin_size, color='#4f8fff',\
                    align='center', edgecolor='none')
            ax_hist.add_patch(
                    patches.Rectangle((0.0, 0.0), 0.0015, rp_height,
                        facecolor='#ff0000', edgecolor='#ff0000', alpha=0.5))
            plt.title('cluster ID: ' + str(cluster_id) + \
                    'cluster group: ' + cluster_group + \
                    'rbvs: ' + str(rpvs) + '/' + str(len(isi)) + '   ({:2.2f})%'.format(rpv_perc))

            ## TODO add suplots with grid n stuff

            plt.show()


