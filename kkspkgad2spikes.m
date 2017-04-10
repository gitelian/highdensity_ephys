%KK2SPIKES.m Write spikes structure from KlustaKwik output.
%   KK2SPIKES() takes no arguements; it pulls up a dialogue box in which the user
%   is asked to select an experiment directory. KK2SPIKES will look for a
%   SpikeGadgets .rec file as well as directories containing KlustaKwik .kwik
%   files. Using the information from these two files it will create "spikes"
%   a matlab array similar to that produced by UltraMegaSorter2000 spike sorter.
%
%   The experiment directory is expected to have a .rec file and n directories
%   where n is the number of electrodes used in the experiment. This directory
%   structure is produced by running spkgad2flatarray.m
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20160815
%
%   20160815 update: can process experiments that have no trials. periods of
%   time that do NOT correspond to a trial are given a stimulus ID of zero.
%%
file_path = uigetdir('/media/greg/data/neuro/', 'Select folder to extract spike data');

if file_path == 0
    error('no directory was selected')
end

[~, fname, ~] = fileparts(file_path);
fid = fname(1:7);

rec_file_struct = dir([file_path filesep fid '*.rec']);
if isempty(rec_file_struct)
    error('no .rec file found')
else
    [~, rec_fname, ~] = fileparts(rec_file_struct.name);
end

exp_dir_struct = dir([file_path filesep fid '_e*']);

if isempty(exp_dir_struct)
    error('no electrode folders found')
end

dio_file_struct = dir([file_path filesep fid '*_dio.mat']); % change to *_dio.mat; change rec to dio
if isempty(dio_file_struct)
    error('no dio file found')
else
    [~, dio_fname, ~] = fileparts(dio_file_struct.name);
end

num_exp = length(exp_dir_struct);

% all variables below this are created using the above variables
% save above varialbes to a temp file, clear all, and load them in
save([tempdir filesep 'kk2spikes_temp.mat'], '-v7.3')

for exp_i = 1:num_exp
    exp_dir_name = exp_dir_struct(exp_i).name;
%     path2rec     = [file_path filesep rec_fname '.rec'];
    path2dio     = [file_path filesep dio_fname '.mat']; % change to .mat
    path2kwik    = [file_path filesep exp_dir_name filesep];
    kwik_struct  = dir([path2kwik filesep rec_fname '*.kwik']);
    phy_struct   = dir([path2kwik filesep rec_fname '*.phy.dat']);
    prb_struct   = dir([path2kwik filesep '*.prb']);
    disp('USING PROBE FILE TO GET NUMBER OF CONTACTS OF ELECTRODE!')
    disp(prb_struct.name)
    num_chan     = str2double(prb_struct.name(4:5));

    if isempty(kwik_struct)
        warning(['no .kwik file found in ' path2kwik])
    elseif isempty(phy_struct)
        warning(['no .phy.dat file found in ' path2kwik])
    else
        disp(['creating spikes file for ' exp_dir_name])
        phy_name  = [path2kwik phy_struct.name];
        kwik_name = [path2kwik kwik_struct.name];


%% add hdf5 file info and directions on how to access parameter info
diary([path2kwik 'h5parameters.txt']);
h5disp(kwik_name);
diary off;
spikes.info = h5info(kwik_name);
spikes.params = 'open test file h5parameters.txt';

%% create the unit ID assignment
%  in UltraMegaSorter this is spikes.assigns
fprintf('\n#####\nloading in spike assignment vector\n#####\n')
spikes.assigns = hdf5read(kwik_name, '/channel_groups/0/spikes/clusters/main');

%% create the unwrapped times array
%  in UltraMegaSorter this is spikes.unwrapped_times
fprintf('\n#####\nloading spikes times\n#####\n')
spk_inds = hdf5read(kwik_name, '/channel_groups/0/spikes/time_samples');
spikes.unwrapped_times = single(spk_inds)/30000; % divide by sampline rate to get spike times in seconds

%% how to create the nx2 array where each unique cluster id is paired with its cluster group
%  that is is the unit in the noise, multiunit, good, and unclustered
%  category. this is the same format as in the UltraMegaSorter spike
%  structure output.
fprintf('\n#####\nmatching unit IDs with cluster IDs\n#####\n')
uid = hdf5read(kwik_name, '/channel_groups/0/spikes/clusters/main');
cluster_ids = unique(uid);
labels = nan(length(cluster_ids), 2);
for k = 1:length(cluster_ids)
    test = h5info(kwik_name, ['/channel_groups/0/clusters/main/' num2str(cluster_ids(k))]);
    for ind = 1:length(test.Attributes)
        if strcmp(test.Attributes(ind).Name, 'cluster_group')
            clst_group_ind = ind;
        end
    end

    labels(k, :) = [cluster_ids(k), test.Attributes(clst_group_ind).Value];
end

spikes.labels = labels;
clear labels

%% create the following arrays
%  trials, stimuli, spiketimes <-- length equals length of entire experiment
%  stimsequence, stimulus_sample_num, stimulus_times <-- length equals
%  number of trials
%  in UltraMegaSorter this is spikes.trials
fprintf('\n#####\nloading digital input lines to ID trial times and condition types\n#####\n')

% dio                 = readTrodesFileDigitalChannels(path2rec);
load(path2dio)

num_samples         = length(dio.timestamps);
% could the channel index be or OUT OF ORDER???
if strcmp(dio.channelData(1).id, 'Din1') == 0
    warning('***** Channel 1 does not equal Din1 *****')
else
    % if it is ever out of order add a loop here to find the index of the
    % correct channel and use it below. Then add this type of check anywhere
    % readTrodesFileDigitalChannels is used to extract digital inputs/outputs
    fprintf('\n#####\nDIO Channel 1 equals Din1\n#####\n')
end
state_change_inds   = find(diff(dio.channelData(1).data) ~= 0) + 1; % indices of all state changes
num_state_changes   = length(state_change_inds);
disp(['number of state changes: ' num2str(num_state_changes)])
trials              = zeros(num_samples, 1, 'single');
stimuli             = zeros(num_samples, 1, 'single');
stimsequence        = zeros(num_state_changes/2, 1, 'single');
stimulus_sample_num = zeros(num_state_changes/2, 2, 'single');
stimulus_times      = zeros(num_state_changes/2, 2, 'single');
trial_times         = zeros(num_state_changes/2, 2, 'single'); % will have time the trial starts and stops (puts bounds on the spike times)
spiketimes          = double(spk_inds)/30000;

% only split data into trials if they exist
if num_state_changes > 0

    % use the state change indices to label each sample with the ID of the trial.
    % digital input 1 goes high during stimulus presentation. I need to label
    % samples before and after the stimulus period to indicate that it is a part
    % of the trial. this way we can have a baseline period and after stimulus
    % period for each trial.
    trial_count     = 1;
    dt_state_change = diff(state_change_inds); % computes time before each start, stop, start, ..., stop event
    iti_duration    = dt_state_change(2:2:end); % -stop1 + start2 + ...
    dt_before_after = uint64(iti_duration/2);
    dt_before_after = [mean(dt_before_after); dt_before_after; mean(dt_before_after)];
    progressbar('labeling trials')

    for k = 1:2:num_state_changes - 1 % jumping by 2 will always select the start time with k and the stop time with k+1
        
        if dynamic_time
            ind0 = state_change_inds(k) - time_before*30000;   % start time index
            ind1 = state_change_inds(k+1) + time_after*30000; % stop time index
        else
            ind0 = state_change_inds(k) - dt_before_after(trial_count);   % start time index
            ind1 = state_change_inds(k+1) + dt_before_after(trial_count + 1); % stop time index
        end
        stimulus_sample_num(trial_count, :) = [state_change_inds(k), state_change_inds(k+1)]; % get index of stim start/start (i.e. object starts moving into place and when it starts leaving)
        stimulus_times(trial_count, :)      = (double([state_change_inds(k), state_change_inds(k+1)]) - double(ind0))/30000; % gets time of stimulus start
        trials(ind0:ind1) = trial_count;

        % determine what stimulus was presented bu counting the number of high
        % pusles on the second digital input line.
        num_pulses = length(find(diff(dio.channelData(2).data(ind0:ind1)) < 0));
        stimuli(ind0:ind1) = num_pulses;
        stimsequence(trial_count) = num_pulses;
        trial_times(trial_count, :)  = (double([ind0, ind1]) - double(ind0))/30000;

        % create spiketimes array here
        % get indices of actual spike times that occured during this trial
        % And convert the indices to time in seconds relative to the beginning of trials.
        temp_spk_ind0 = find(spk_inds >= ind0, 1,'first');
        temp_spk_ind1 = find(spk_inds <= ind1, 1, 'last');
        spiketimes(temp_spk_ind0:temp_spk_ind1) = (double(spk_inds(temp_spk_ind0:temp_spk_ind1)) - double(ind0))/30000;

        if trial_count == num_state_changes/2 % when trial_count equals the number of trials take the rest of the spikes
            spiketimes(temp_spk_ind1:end) = (double(spk_inds(temp_spk_ind1:end)) - double(ind1))/30000;
            spiketimes(spiketimes < 0) = 0;
        end
        progressbar(trial_count/(num_state_changes/2))
        trial_count = trial_count + 1;
    end
    progressbar(1)
else
    %% do this if there are NO trials (e.g. just a continuous test recording).
    warning(['NO TRIAL DATA WAS FOUND for ' rec_fname])
    spiketimes = spiketimes(end) - double(spiketimes(1));
end

spikes.trials              = trials(spk_inds);
spikes.stimuli             = stimuli(spk_inds);
spikes.spiketimes          = spiketimes;
spikes.stimsequence        = stimsequence;
spikes.stimulus_sample_num = stimulus_sample_num;
spikes.stimulus_times      = stimulus_times;
spikes.trial_times         = trial_times;

clear trials stimuli spiketimes stimsequence stimulus_sample_num ...
    stimulus_times dio

%% retrive waveforms for all units
fprintf('\n#####\nloading raw data for waveform extraction\n#####\n')

aio = fopen(phy_name);
raw_data = fread(aio,'int16=>int16');
fclose(aio);
clear aio
raw_data = reshape(raw_data, num_chan, length(raw_data)/num_chan);
% raw_data = reshape(single(raw_data), num_chan, length(raw_data)/num_chan);
% TEMP COMMENTS
progressbar('filtering data')
for r = 1:num_chan
    raw_data(r, :) = single(neuro_filt(raw_data(r, :)));
    progressbar(r/num_chan)
end
progressbar(1)

% get 15 samples before and 45 samples after (2ms worth of data)
num_units       = size(spikes.labels, 1);
num_spikes      = length(spikes.assigns);
waveforms       = zeros(num_spikes, 60, num_chan, 'single');
num_raw_samples = size(raw_data, 2);

progressbar('spike waveform extraction')
for spike_ind = 1:num_spikes
    % get index that this spike occurred during the experiment
    % here index 1 corresponds to the first recorded sample NOT the first
    % time a spike occurred!
    i = spk_inds(spike_ind);

    % get raw data and filter it.
    % 15 samples before i and 45 after (including i)
    % filter works on the columns of a matrix
    % transpose so filter works on time (electrodes x time)'
%     output = neuro_filt(double(raw_data(:, (i-15):(i+45-1))'));
%     output = output';
    if i <= 15
        output = raw_data(:, (1:60));
    elseif i >= num_raw_samples - 45
        output = raw_data(:, end-60+1:end);
    else
        output = raw_data(:, (i-15):(i+45-1));
    end

    % reshape output so it fits in the waveforms matrix.
    % spike-index x samples x electrode
    waveforms(spike_ind, :, :) = reshape(output', 60, num_chan);

    progressbar(spike_ind/num_spikes)
end

progressbar(1)
spikes.waveforms = waveforms;

clear raw_data waveforms

% delete file and replace it with new one
if exist([path2kwik filesep fid '-e' num2str(exp_i) '-spikes.mat'], 'file') == 1
    delete([path2kwik filesep fid '-e' num2str(exp_i) '-spikes.mat'])
end

fprintf(['\n#####\nSaving Data for ' fid '-e' num2str(exp_i) '\n#####\n'])
save([path2kwik filesep fid '-e' num2str(exp_i) '-spikes.mat'], 'spikes', '-v7.3')

% clear all variables and load in variables in the temp file for the net iteration

if exp_i ~= num_exp
    fprintf('\n#####\nCLEARING DATA\n#####\n')
    clear all
    load([tempdir filesep 'kk2spikes_temp.mat'])
end
    end % end else
end % end for loop

clear all
