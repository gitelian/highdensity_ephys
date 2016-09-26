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
from scipy.interpolate import interp1d
from sklearn.cluster import KMeans
from sklearn import mixture
import itertools as it

def load_spike_file(path):
    """
    Loads spikes file from specified path
    """
    mat  = h5py.File(path)
    spks = mat['spikes']

    labels      = np.asarray(spks['labels']).T     # ndarray shape(n x m) n: num elements, m0 = unit label, m1 = type of unit (i.e. single, multi, garbage)
    assigns     = np.asarray(spks['assigns']).T    # ndarray shape(n) n: num of spike times for all units
    trials      = np.asarray(spks['trials']).T     # ndarray shape(n) n: num of spike times for all units
    spike_times = np.asarray(spks['spiketimes']).T # ndarray shape(n) n: num of spike times for all units
    waves       = np.asarray(spks['waveforms']).T  # ndarray shape(n x m x p) n: num of spike times for all units m: num of range(m)les in waveform
                                    # p: num of recording channels
    nsamp       = waves.shape[1]
    nchan       = waves.shape[2]  # get number of channels used

    uniqassigns = np.unique(assigns)                   # get unique unit numbers
    unit_type   = labels[:, 1]                          # get unit type 0=noise, 1=singlunit, 3=multi-unit, 4=unsorted
    ids         = labels[:, 0]
    nunit       = len(ids)

    return labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type

def load_v73_mat_file(file_path, variable_name='spike_measures'):
    print('\n----- load_v73_mat_file -----')
    print('Loading data from: ' + file_path + '\nvariable: ' + variable_name)
    mat = h5py.File(file_path)
    mat = mat[variable_name][:].T
    print('\nvariable shape ' + str(mat.shape))
    return mat

def load_mat_file(file_path, variable_name='spike_msr_mat'):
    print('\n----- load_mat_file -----')
    print('Loading data from: ' + file_path)
    mat = sio.loadmat(file_path, struct_as_record=False, squeeze_me=True)
    return mat[variable_name]

def get_depth(best_chan, exp_info):

    tip_depth = exp_info['e1_depth']
    probe     = exp_info['e1_probe']

    if probe == 'a1x16':
        depth_vec = np.abs(np.arange(0,400,25)-tip_depth)
    elif probe == 'a1x32':
        depth_vec = np.abs(np.arange(0,800,25)-tip_depth)

    depth = depth_vec[best_chan]
    return depth

def update_spikes_measures_mat(fid_list=[], data_dir_path='/media/greg/data/neuro/'):
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


    if len(fid_list) == 0:
        print('No FID list provided. Current spike_measures.mat will\n' + \
                'be overwritten and updated with all spikes files.')
        overwrite=True
        spike_msr_mat = np.zeros((1, 8))

    elif os.path.exists(spike_measures_path):
        # load spike measures mat file
        print('Loading spike measures mat file: ' + spike_measures_path)
        spike_msr_mat = load_mat_file(spike_measures_path, variable_name='spike_msr_mat')
        overwrite=False

        if spike_msr_mat.shape[1] != 8:
            raise Exception('Spike_msr_mat is NOT the correct size')
    else:
        # create matrix if it does not exist
        # fid, electrode, unit_id, depth, unit_id, duration, ratio, cell_type
        # cell_type (0=MU, 1=RS, 2=FS, 3=UC)
        print('spike_measures.mat does not exist...creating a new one')
        spike_msr_mat = np.zeros((1, 8))
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

    #TODO: check to make sure the path exists!!! Error out if it doesn't

    ##### ITERATE THROUGH ALL SPIKES.MAT FILES AND UPDATE SPIKE_MEASURES AS NECESSARY #####

    for count, spikes_path in enumerate(path2spikes_files):
        spikes_fname = os.path.basename(spikes_path)

        # find the fid number
        try:
            fid_match = re.search(r'fid\d{1,4}', spikes_fname, re.IGNORECASE)
            fid = int(fid_match.group()[3::])
        except:
            raise Exception('Experiment FID#### not found!\nRename: ' + spikes_path)

        # find what electrode the unit is from
        try:
            e_match = re.search(r'e\d{1,2}', spikes_fname, re.IGNORECASE)
            e_num = int(e_match.group()[1::])
        except:
            raise Exception('Spikes file does not have electrode information!\nRename: ' + spikes_path)

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
        labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type = load_spike_file(spikes_path)

        # Loop through units from sorted spike file
        print('unit IDs: ' + str(ids))
        print('\n')
        for k, unit_id in enumerate(ids):

            if unit_type[k] > 0 and unit_type[k] < 3: # don't include noise and unsorted clusters

                # If the unit measure is not in the measure mat file OR overwrite is True
                # The sorted spike file will be loaded and measures will be calculated

                unit_id_index  = np.where(spike_msr_mat[:, 2] == unit_id)[0]
                e_num_index    = np.where(spike_msr_mat[:, 1]  == e_num)[0]
                fid_index      = np.where(spike_msr_mat[:, 0] == fid)[0]
                unit_mat_index = np.intersect1d(np.intersect1d(e_num_index, unit_id_index), fid_index)
                # unit_in_mat will be zero if the unit is NOT in the
                # spike_easures.mat file
                unit_in_mat    = unit_id_index in fid_index and e_num_index in fid_index
                # if unit is in spikes measure file and overwrite is False don't
                # do anything

                print('Working on: ' + 'FID' + str(fid) + ' unit: ' + str(unit_id))
                spike_inds     = np.where(assigns == unit_id)[0]
                wave_temp      = waves[spike_inds, :, :]
                mean_wave_temp = np.mean(wave_temp,axis=0)
                std_wave_temp  = np.std(wave_temp,axis=0)
                wave_min       = np.min(mean_wave_temp,axis=0)
                best_chan      = np.argmin(wave_min) # biggest negative deflection is peak of waveform

                # Upsample via cubic spline interpolation and calculate waveform measures from
                # upsampled waveforms. This gives a more precise, less obviously discretized measurements
                nmid  = nsamp/2 + 1
                newsamp = nsamp*4
                xnew = np.linspace(0, nsamp-1, newsamp)
                f = interp1d(range(nsamp), mean_wave_temp[:,best_chan], kind='cubic')
                ynew = f(xnew)
                min_index  = np.argmin(ynew)
                max_index1 = np.argmax(ynew[min_index:-1])+min_index+1
                max_index0 = np.argmax(ynew[0:(min_index)])
                min_value  = ynew[min_index]
                max_value1 = ynew[max_index1]
                max_value0 = ynew[max_index0]
                duration   = (max_index1-min_index)/(30.0*4+1)
                wave_ratio = (max_value1 - max_value0)/(max_value0 + max_value1)
#                ## REMOVE THIS ##
#                print(nmid)
#                print('wave_ratio: ' + str(wave_ratio))
#                print('wave_duration: ' + str(duration))
#                plt.figure()
#                plt.plot(ynew,'k')
#                plt.plot(max_index1, ynew[max_index1], '*r')
#                plt.plot(max_index0, ynew[max_index0], '*b')
#                plt.plot(min_index, ynew[min_index], '*g')
#                plt.show()
#                fail()
#                plt.pause(10)
#                plt.close()

                # Append depth, wave duration, and wave ratio to respective lists
                depth = get_depth(best_chan,exp_info)
                dur   = duration
                ratio = wave_ratio

                # spike_measures columns order: fid, electrode, unit_id,
                # depth, unit_id (MU=1, SU=2), duration, ratio, MU/RS/FS/UC
                if overwrite is True:
                    # add everything to new matrix
                    spike_msr_mat = np.append(spike_msr_mat, np.array(\
                            [fid, e_num, unit_id, depth, unit_type[k], dur, ratio, 0]).reshape(1,8), axis=0)
                elif overwrite is False and unit_in_mat:
                    # use unit index to update matrix row
                    spike_msr_mat[unit_mat_index, :] = \
                            np.array([fid, e_num, unit_id, depth, unit_type[k], dur, ratio, 0]).reshape(1,8)
                elif overwrite is False and unit_in_mat is False:
                    # append to existing matrix
                    spike_msr_mat = np.append(spike_msr_mat, np.array(\
                            [fid, e_num, unit_id, depth, unit_type[k], dur, ratio, 0]).reshape(1,8), axis=0)

        # Free up memory so I don't get a MemoryError
        print('clearing up memory: removing spikes file data')
        del labels, assigns, trials, spike_times, waves, nsamp, nchan, ids, nunit, unit_type

    # if the first row is 0 remove it by skipping it with slicing
    if np.count_nonzero(spike_msr_mat[0, :]) is 0:
        spike_msr_mat = spike_msr_mat[1::, :]

    print('saving spikes measure mat file: ' + spike_measures_path)
    # lexsort sorts by last entry to first (e.g. by FID, electrode, unit_id, and depth).
    # spike_measures columns order: fid [0], electrode [1], unit_id [2],
    # depth [3], unit_id [4] (MU=1, SU=2), duration [5], ratio [6],
    # MU/RS/FS/UC [7]
    spike_msr_sort_inds = np.lexsort((spike_msr_mat[:, 3], spike_msr_mat[:, 1], spike_msr_mat[:, 0]))
    spike_msr_mat = spike_msr_mat[spike_msr_sort_inds, :]
    # SAVE MAT FILE
    a = dict()
    a['spike_msr_mat'] = spike_msr_mat
    sio.savemat(spike_measures_path, a)

def classify_units(data_dir_path='/media/greg/data/neuro/'):

    print('\n----- classify_units function -----')
    ##### Load in spike measures .mat file #####
    spike_measures_path = data_dir_path + 'spike_measures.mat'
    if os.path.exists(spike_measures_path):
        # load spike measures mat file
        print('Loading spike measures mat file: ' + spike_measures_path)
        spike_msr_mat = load_mat_file(spike_measures_path, variable_name='spike_msr_mat')

    if spike_msr_mat.shape[1] != 8:
            raise Exception('Spike_msr_mat is NOT the correct size')

        # spike_measures columns order: fid [0], electrode [1], unit_id [2],
        # depth [3], unit_id [4] (MU=1, SU=2), duration [5], ratio [6],
        # MU/RS/FS/UC [7]

    good_unit_inds  = np.where(spike_msr_mat[:, 4] == 2)[0]
    dur_array       = spike_msr_mat[good_unit_inds, 5]
    ratio_array     = spike_msr_mat[good_unit_inds, 6]
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
    for ind, val in enumerate(pred_prob):
        if val[rs_index] >= 0.90:
            spike_msr_mat[good_unit_inds[ind], 7] = 1
            #cell_type_list.append('RS')
        elif val[pv_index] >= 0.90:
            spike_msr_mat[good_unit_inds[ind], 7] = 2
            #cell_type_list.append('PV')
        else:
            spike_msr_mat[good_unit_inds[ind], 7] = 3
            #cell_type_list.append('UC')

    print('saving spikes measure mat file: ' + spike_measures_path)
    # SAVE MAT FILE
    a = dict()
    a['spike_msr_mat'] = spike_msr_mat
    sio.savemat(spike_measures_path, a)

    rs_bool = np.where(spike_msr_mat[good_unit_inds, 7] == 1)[0]
    pv_bool = np.where(spike_msr_mat[good_unit_inds, 7] == 2)[0]
    uc_bool = np.where(spike_msr_mat[good_unit_inds, 7] == 3)[0]

    fig = plt.subplots()
    plt.scatter(dur_ratio_array[rs_bool,0],dur_ratio_array[rs_bool,1],color='g',label='RS')
    plt.scatter(dur_ratio_array[pv_bool,0],dur_ratio_array[pv_bool,1],color='r',label='PV')
    plt.scatter(dur_ratio_array[uc_bool,0],dur_ratio_array[uc_bool,1],color='k',label='UC')
    plt.xlabel('duration (ms)')
    plt.ylabel('ratio')
    plt.legend(loc='upper right')
    plt.show()

########## MAIN CODE ##########
########## MAIN CODE ##########

if __name__ == "__main__":
    #TODO replace file path seps with filesep equivalent
    update_spikes_measures_mat(fid_list=[], data_dir_path='/media/greg/data/neuro/')
    classify_units(data_dir_path='/media/greg/data/neuro/')


