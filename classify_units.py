#!/bin/bash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as sio
import h5py
import glob
import os
import re

from scipy import stats
from scipy.interpolate import interp1d
from scipy import signal as sig
import itertools as it
from sklearn.cluster import KMeans
from sklearn import mixture
import multiprocessing as mp
import time
import random as rd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold

def load_spike_file(path):
    """
    Loads spikes file from specified path
    """
    mat  = h5py.File(path)
    spks = mat['spikes']

    spks.labels      = np.asarray(spks['labels']).T     # ndarray shape(n x m) n: num elements, m0 = unit label, m1 = type of unit (i.e. single, multi, garbage)
    del spks['labels']
    spks.assigns     = np.asarray(spks['assigns']).T    # ndarray shape(n) n: num of spike times for all units
    del spks['assigns']
    spks.trials      = np.asarray(spks['trials']).T     # ndarray shape(n) n: num of spike times for all units
    del spks['trials']
    spks.spike_times = np.asarray(spks['spiketimes']).T # ndarray shape(n) n: num of spike times for all units
    del spks['spiketimes']
    spks.waves       = np.asarray(spks['waveforms']).T  # ndarray shape(n x m x p) n: num of spike times for all units m: num of range(m)les in waveform
    del spks['waveforms']
                                    # p: num of recording channels
    spks.nsamp       = spks.waves.shape[1]
    spks.nchan       = spks.waves.shape[2]  # get number of channels used

    uniqassigns = np.unique(spks.assigns)                   # get unique unit numbers
    spks.unit_type = spks.labels[:,1]                          # get unit type 0=noise, 1=singlunit, 3=multi-unit, 4=unsorted
    spike_ids      = spks.labels[spks.unit_type > 0,0];             # get good spike ids
    ids_set        = [set(uniqassigns), set(spike_ids)]        # convert a list containing two sets
    spks.ids       = np.sort(list(set.intersection(*ids_set))) # convert to set, find intersection, return list of good unit ids
    spks.nunit     = len(spks.ids)

    return spks

def load_mat_file(file_path, variable_name='spike_measures'):
    print('\n----- load_mat_file -----')
    print('Loading data from: ' + file_path + '\nvariable: ' + variable_name)
    mat = h5py.File(file_path)
    mat = mat[variable_name][:].T
    print('\nvariable shape ' + str(mat.shape))
    return mat

def calculate_runspeed(run_mat,Fs=30000.0):#,highthresh=250.0,lowthresh=25.0,stimstart=1.5,stimstop=2.5):
    # original time for 350 trials: 19.8s
    # time with parallel pool using 'apply': 13.2s
    # time with parallel pool using 'apply_async' with 1-4 processes: 13.0s, 8.42, 6.14, 5.79s
    # no improvement when processes exceeds number of virtual cores

    print('\n----- calculate_runspeed -----')
    processes = 4
    pool = mp.Pool(processes)
    t = time.time()

    ntrials = run_mat.shape[0]
    nsamples = run_mat.shape[1]
    down_samp = 5.0
    down_samp_fs = round(Fs/down_samp)

    vel_mat = np.zeros((ntrials,nsamples/down_samp))
    gauss_window = make_gauss_window(round(down_samp_fs),255.0,down_samp_fs)
    window_len = len(gauss_window)
    trial_time = np.linspace(0,vel_mat.shape[1]/down_samp_fs - 1/down_samp_fs,vel_mat.shape[1])

    x_t = run_mat[:,0::down_samp]

    print('computing run speed with {0} processes'.format(processes))
    results = [pool.apply_async(calculate_runspeed_derivative, args=(x_t[i,:], gauss_window,window_len,down_samp_fs,i)) for i in range(x_t.shape[0])]
    results = [p.get() for p in results]
    # ensure output is in correct order. 'apply_async' does not ensure order preservation
    order,data = zip(*[(entry[0],entry[1]) for entry in results])
    sort_ind = np.argsort(order)
    for ind in sort_ind:
            vel_mat[ind,:] = data[ind]

    elapsed = time.time() - t
    pool.close()
    print('total time: ' + str(elapsed))

    return vel_mat, trial_time

def calculate_runspeed_derivative(x_t,gauss_window,window_len,down_samp_fs,order):
	# calculate slope of last 500 samples (last 83ms), use it to extrapolate x(t) signal. assumes mouse runs at
	# that constant velocity. Prevents convolution from producing a huge drop in velocity due to repeating last values
	x_t = np.append(x_t,np.linspace(x_t[-1],(x_t[-1]-x_t[-500])/500.0 *window_len+ x_t[-1],window_len))
	dx_dt_gauss_window = np.append(0,np.diff(gauss_window))/(1.0/down_samp_fs)
	dx_dt = np.convolve(x_t,dx_dt_gauss_window,mode='same')
	dx_dt = dx_dt[:-window_len]
	return order,dx_dt

def get_depth(best_chan, exp_info):

    tip_depth = exp_info['e1_depth']
    probe     = exp_info['e1_probe']

    if probe == 'a1x16':
        depth_vec = np.abs(np.arange(0,400,25)-tip_depth)
    elif probe == 'a1x32':
        depth_vec = np.abs(np.arange(0,800,25)-tip_depth)

    depth = depth_vec[best_chan]
    return depth

def update_spikes_measures_mat(fid_list=[], data_dir_path='/media/greg/Data/Neuro/')
    """
    Run with no arguements to overwrite spike_measures.mat file with new measurements
    Run with a list of experiment IDs (i.e. FID####) to only add measurements from
    units found in the spikes files associated with those experiments.
    """

    ##### LOOK FOR EXISTING SPIKE_MEASURES.MAT FILE #####
    # determine whether to overwrite existing file, create a new one, or update
    # a current one.
    print('\n----- update_spikes_measures_mat -----')
    # load in spike measure mat file
    spike_measures_path = data_dir_path + 'spike_measures.mat'

    if os.path.exists(spike_measures_path):
        # load spike measures mat file
        print('Loading spike measures mat file: ' + spike_measures_path)
        spike_msr_mat = load_mat_file(spike_measures_path, variable_name='spike_measures')

        if len(fid_list) == 0:
            print('No FID list provided. Current spike_measures.mat will\n' + \
                    'be overwritten and updated with all spikes files.')
            overwrite=True
            spike_msr_mat = np.zeros((1, 7))
        else:
            overwrite=False
    else:
        # create matrix if it does not exist
        # fid, electrode, unit_id, depth, cell_type, duration, ratio
        print('spike_measures.mat does not exist...creating a new one')
        spike_msr_mat = np.zeros((1, 7))
        overwrite=True

    ##### LOAD IN EXPERIMENT DETAILS CSV FILE #####
    experiment_details_path = data_dir_path + 'experiment_details.csv'
    print('Loading experiment details file: ' + experiment_details_path)
    df_exp_det_regular_index = pd.read_csv(experiment_details_path,sep=',')
    df_exp_det_fid_index = df_exp_det_regular_index.set_index('fid')
    print(df_exp_det_fid_index)

    ##### GET PATHS TO ALL OF THE APPROPRIATE SPIKES MAT FILES #####
    # looks through experiment data directory which should have the structure
    # of /experiment_dir/electrode_sub_directories/*spikes.mat
    if len(fid_list) == 0:
        path2spikes_files = np.sort(glob.glob(data_dir_path + 'FID*/FID*_e*/*spikes.mat'))
    else:
        path2spikes_files = list()
        for fid_name in fid_list:
            path2spikes_files.extend(np.sort(glob.glob(data_dir_path + fid_name + '*/' + 'FID*_e*/' + '*spikes.mat')))

    ##### ITERATE THROUGH ALL SPIKES.MAT FILES AND UPDATE SPIKE_MEASURES AS NECESSARY #####

    for count, spikes_path in enumerate(path2spikes_files):
        spikes_fname = os.path.basename(spikes_path)

        # find the fid number
        fid_match = re.search(r'fid\d{1,4}', spikes_fname, re.IGNORECASE)
        fid = int(fid_match.group()[3::])

        # find what electrode the unit is from
        fid_match = re.search(r'e\d{1,2}', spikes_fname, re.IGNORECASE)
        e_num = int(fid_match.group()[1::])

        # check if experiment is in the experiment details csv file
        # need this info to extract electrode depth

        if fid not in df_exp_det_fid_index.index:
            raise Exception('fid' + str(fid) + ' not found in experiment details csv file.\n\
                            Please update file and try again.')
        else:
            print('Loading experiment info for fid: ' + str(fid))
            exp_info = df_exp_det_fid_index.loc[fid]

        print('Loading sorted spike mat file: ' + spikes_fname)

        # loads in matlab structure as python objects
        # makes syntax for getting data more similar to matlab's structure syntax
        spks = load_spike_file(spikes_path)
        # labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type = load_spike_file(sort_file)

        # Loop through units from sorted spike file
        print(spks.ids)
        print('\n')
        for k, unit_id in enumerate(spks.ids):

            if spks.unit_type[k] > 0 and spks.unit_type[k] < 3: # don't include noise and unsorted clusters

                # If the unit measure is not in the measure mat file OR overwrite is True
                # The sorted spike file will be loaded and measures will be calculated

                unit_id_index  = np.where(spike_msr_mat[:, 2] == unit_id)[0]
                e_num_index    = np.where(spke_msr_mat[:, 1]  == e_num)[0]
                fid_index      = np.where(spike_msr_mat[:, 0] == fid)[0]
                unit_mat_index = np.intersect1d(np.intersect1d(e_num_index, unit_id_index), fid_index)
                # unit_in_mat will be zero if the unit is NOT in the
                # spike_easures.mat file
                unit_in_mat    = unit_id_index in fid_index and e_num_index in fid_index

                # if unit is in spikes measure file and overwrite is False don't
                # do anything
                if  len(unit_mat_index) == 0 or overwrite is True:

                    print('Working on: ' + 'FID' + str(fid) + ' unit: ' + str(unit_id))
                    spike_bool     = spks.assigns == unit_id
                    wave_temp      = spks.waves[spike_bool]
                    mean_wave_temp = np.mean(wave_temp,axis=0)
                    std_wave_temp  = np.std(wave_temp,axis=0)
                    wave_min       = np.min(mean_wave_temp,axis=0)
                    best_chan      = np.argmin(wave_min) # biggest negative deflection is peak of waveform

                    # Upsample via cubic spline interpolation and calculate waveform measures from
                    # upsampled waveforms. This gives a more precise, less obviously discretized measurements
                    newsamp = spks.nsamp*4
                    xnew = np.linspace(0, spks.nsamp-1, newsamp)
                    f = interp1d(range(spks.nsamp), mean_wave_temp[:,best_chan], kind='cubic')
                    ynew = f(xnew)
                    min_index  = np.argmin(ynew)
                    max_index1 = np.argmax(ynew[(23*4+1):-1])+23*4+1
                    max_index0 = np.argmax(ynew[0:(23*4+1)])
                    min_value  = ynew[min_index]
                    max_value1 = ynew[max_index1]
                    max_value0 = ynew[max_index0]
                    duration   = (max_index1-min_index)/(30.0*4+1)
                    wave_ratio = (max_value1 - max_value0)/(max_value0 + max_value1)

                    # Append depth, wave duration, and wave ratio to respective lists
                    depth = get_depth(best_chan,exp_info,region)
                    dur   = duration
                    ratio = wave_ratio

                    # spike_measures columns order: fid, electrode, unit_id,
                    # depth, cell_type, duration, ratio, MU/RS/FS/UC
                    if over_write is True:
                        spike_msr_mat = np.append(spike_msr_mat, np.array(\
                                [fid, e_num, unit_id, depth, spks.unit_type[k], dur, ration]))
                    elif overwrite is False:
                        spike_msr_mat[unit_mat_index, :] = \
                                [fid, e_num, unit_id, depth, spks.unit_type[k], dur, ration]))

    print('saving spikes measure file: ' + spike_measure_path)
    # lexsort sorts by first entry last...last entry firs (e.g. by FID,
    # electrode, unit_id, and then by depth).
    spkmsr = spike_msr_mat
    spike_msr_sort_inds = np.lexsort((spkmsr[4, :], spkmsr[3, :], spkmsr[2, :],\
            spkmsr[1, :], spkmsr[0, :]))
    spike_msr_mat = spkmsr[spike_msr_sort_inds, :]
    # SAVE MAT FILE
    a = dict()
    a['spike_msr_mat'] = spike_msr_mat
    io.savemat('spike_measures_path', a)

def classify_units(data_dir_path,region):

    print('\n----- classify_units function -----')
    ##### Load in spike measures .mat file #####
    spike_measures_path = data_dir_path + 'spike_measures.mat'
    if os.path.exists(spike_measures_path):
        print('Loading spike measures .mat file: ' + spike_measures_path)
        spike_msr_mat = load_mat_file(spike_measures_path, variable_name='spike_measures')
        spkmsr = spike_msr_mat
        # spike_measures columns order:
        # fid 0, electrode 1, unit_id 2, depth 3, sort_type 4, duration 5,
        # ratio 6, cell_type 7
        spike_msr_sort_inds = np.lexsort((spkmsr[:, 4], spkmsr[:, 3], spkmsr[:, 2],\
                spkmsr[:, 1], spkmsr[:, 0]))
        spike_msr_mat = spkmsr[spike_msr_sort_inds, :]

    good_unit_inds  = np.where(spike_msr_mat[:, spkmsr[:, 4])
    dur_array       = spike_msr_mat[:, 5]
    ratio_array     = spike_msr_mat[:, 6]
    num_rows        = dur_array.shape[0]
    dur_array       = dur_array.reshape(num_rows, 1)
    ratio_array     = ratio_array.reshape(num_rows, 1)
    dur_ratio_array = np.concatenate((dur_array, ratio_array),axis=1)

    ## GMM Clustering
    clf = mixture.GMM(n_components=2, covariance_type='full')
    clf.fit(dur_ratio_array)
    pred_prob = clf.predict_proba(dur_ratio_array)
    gmm_means = clf.means_

    if gmm_means[0,0] < gmm_means[1,0] and gmm_means[0,1] > gmm_means[1,1]:
        pv_index = 0
        rs_index = 1
    else:
        pv_index = 1
        rs_index = 0

    ## Assign PV or RS label to a unit if it has a 0.90 probability of belonging
    ## to a group otherwise label it as UC for unclassified
    cell_type_list = []
    for val in pred_prob:
        if val[pv_index] >= 0.90:
            cell_type_list.append('PV')
        elif val[rs_index] >= 0.90:
            cell_type_list.append('RS')
        else:
            cell_type_list.append('UC')

    df_spk_msr['cell_type'] = cell_type_list

    print('saving csv measure file: ' + spike_measures_path)
    df_spk_msr.to_csv(spike_measures_path, sep=',')

    pv_bool = np.asarray([True if x is 'PV' else False for x in cell_type_list])
    rs_bool = np.asarray([True if x is 'RS' else False for x in cell_type_list])
    uc_bool = np.asarray([True if x is 'UC' else False for x in cell_type_list])

    fig = plt.subplots()
    plt.scatter(dur_ratio_array[pv_bool,0],dur_ratio_array[pv_bool,1],color='r',label='PV')
    plt.scatter(dur_ratio_array[rs_bool,0],dur_ratio_array[rs_bool,1],color='g',label='RS')
    plt.scatter(dur_ratio_array[uc_bool,0],dur_ratio_array[uc_bool,1],color='k',label='UC')
    plt.xlabel('duration (ms)')
    plt.ylabel('ratio')
    plt.legend(loc='upper right')
    plt.show()

def fwhm(x,y):
    '''
    function width = fwhm(x,y)
    Full-Width at Half-Maximum (FWHM) of the waveform y(x)
    and its polarity.
    The FWHM result in 'width' will be in units of 'x'
    Rev 1.2, April 2006 (Patrick Egan)
    Translated from MATLAB to Python December 2014 by Greg I. Telian
    '''

    y = y/max(y)
    N = len(y)
    lev50 = 0.5

    if y[0] < lev50:
        center_index = np.argmax(y)
        Pol = +1
    else:
        center_index = np.argmin(y)
        Pol = -1

    i = 1
    while np.sign(y[i] - lev50) == np.sign(y[i+1] - lev50):
        i += 1

    interp = (lev50 - y[i-1]) / (y[i] - y[i-1])
    tlead = x[i-1] + interp*(x[i] - x[i-1])

    i = center_index + 1
    while ((np.sign(y[i] - lev50) == np.sign(y[i-1] - lev50)) and (i <= N-1)):
        i = i+1

    if i != N:
        interp = (lev50 - y[i-1]) / (y[i] - y[i-1])
        ttrial = x[i-1] + interp*(x[i] - x[i-1])
        width  = ttrial - tlead

    return width

def make_gauss_window(length,std,Fs,make_plot=False):
    '''
    Takes input of window length and alpha and retruns a vector containing a
    smoothing kernel. Output the full width at half max to the display

    Input
    length: length of window, in number of samples
    alpha: alpha parameter for gaussian shape
    Fs: sampling rate of actual data
    Outputs
    smooth_win: vector of smoothing kernel points
    FWHM: full width at half maximum value, in seconds
    also outputs FWHM to display

    J. Schroeder
    Boston University, Ritt lab
    5/31/2012

    Translated from MATLAB to Python December 2014 by Greg I. Telian
    '''
    length = int(length)
    std = float(std)
    Fs = float(Fs)

    window = sig.gaussian(length,std)
    window = window/sum(window)

    if make_plot is True:
        fig = plt.subplots()
        plt.plot(np.arange(length)/Fs,window)

	FWHM = fwhm(np.arange(length)/Fs,window)
	print('Full width at half max for Gauss Kernel is ' + str(FWHM*1000) + ' ms')

    return window

def classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.25,mean_thresh=250,sigma_thresh=150,low_thresh=200,display=False):
    unique_conditions = np.unique(stim)
    trials_per_condition = len(stim)/len(unique_conditions)
    stim_period = (trial_time >= stim_start) & (stim_stop <= 2.50)
    trials_ind_dict = {}
    trials_ran_dict = {}
    mean_vel = []
    sigm_vel = []

    for cond in unique_conditions:
        temp_trial_list = np.where(stim == cond)[0]
        #temp_bool_list = [False]*trials_per_condition
        temp_bool_list = [False]*len(temp_trial_list)
        count = 0
        for trial_ind in temp_trial_list:
            vel = vel_mat[trial_ind][stim_period]
            mean_vel.append(np.mean(vel))
            sigm_vel.append(np.std(vel))
            if np.mean(vel) >= mean_thresh and np.std(vel) <= sigma_thresh and (sum(vel <= low_thresh)/len(vel)) <= 0.1:
                temp_bool_list[count] = True
            count += 1

            if cond < 10:
                trials_ind_dict['cond0' + str(int(cond))] = temp_trial_list
                trials_ran_dict['cond0' + str(int(cond))] = temp_bool_list
            else:
                trials_ind_dict['cond' + str(int(cond))] = temp_trial_list
                trials_ran_dict['cond' + str(int(cond))] = temp_bool_list

    if display:
        bins = range(0,1000,5)
        fig, ax = plt.subplots()
        plt.subplot(2,1,1)
        mean_counts, _ = np.histogram(mean_vel,bins)
        plt.bar(bins[:-1],mean_counts,width=5.0)
        plt.subplot(2,1,2)
        sigm_counts, _ = np.histogram(sigm_vel,bins)
        plt.bar(bins[:-1],sigm_counts,width=5.0)
        plt.show()

    return trials_ind_dict,trials_ran_dict

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    # Select which experiments to analyze
    #fids = ['0871','0872','0873']
    #fids = ['1118', '1123']
    fids = ['0872']
    region = 'vM1'
    unit_count_list = list()
    for fid in fids:
        usr_dir = os.path.expanduser('~')
        sorted_spikes_dir_path = usr_dir + '/Documents/AdesnikLab/SortedSpikes/'
        fid_region = 'fid' + fid + '_' + region
        sort_file_paths = glob.glob(sorted_spikes_dir_path + fid_region + '*.mat')

        data_dir_path = usr_dir + '/Documents/AdesnikLab/Data/'
        data_dir_paths  = glob.glob(data_dir_path + fid + '*.dat')

        # #Calculate runspeed
        run_mat = load_run_file(data_dir_paths[0]).value
        vel_mat, trial_time = calculate_runspeed(run_mat)

        # #Plot runspeed
        # plot_running_subset(trial_time,vel_mat,conversion=True)

        # # Get stimulus id list
        stim = load_stimsequence(data_dir_paths[0])

        # # Create running trial dictionary
        cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
                mean_thresh=175,sigma_thresh=150,low_thresh=200,display=True)
        # easy running thresholds
        #cond_ind_dict,trials_ran_dict = classify_run_trials(stim,vel_mat,trial_time,stim_start=1.25,stim_stop=2.50,
        #        mean_thresh=175,sigma_thresh=150,low_thresh=100,display=True)

        # Find the condition with the least number of trials
        min_run_trials  = min([sum(trials_ran_dict[x]) for x in trials_ran_dict.keys()])

        # Put data into a Pandas dataframe
        df = make_df(sort_file_paths,data_dir_path,region=region)
        df = remove_non_modulated_units(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50, stim_stop=2.50)

        # plot tuning curves
        depth = df['depth']
        cell_type_list = df['cell_type']

        em, es = make_evoke_rate_array(df, trials_ran_dict, base_start=0, base_stop=1.0, stim_start=1.50,
                stim_stop=2.50)
        make_tuning_curves(em, es,depth=depth,cell_type_list=cell_type_list, control_pos=7,
                fig_title='Evoked Firing Rate--fid' + fid + region + ' full pad',
                share_yax=False)
