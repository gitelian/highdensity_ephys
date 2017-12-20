%MSSPKGAD2SPIKES.m Write spikes structure from MountainSort output.
%   MSSPKGAD2SPIKES() takes no arguements; it pulls up a dialogue box in which the user
%   is asked to select an experiment directory. MSSPKGAD2SPIKES will look for a
%   SpikeGadgets .rec file as well as directories containing Mountainsort
%   .mda files. Using the information from these two files it will create "spikes"
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
%
%   20171219 update: will work with MountainSort output
%   NOTE: probe file name must have the number of channels as the 4th and
%   5th characters!!!
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

dio_file_struct = dir([file_path filesep fid '*_dio.mat']); % change to *_dio.mat;
if isempty(dio_file_struct)
    error('no dio file found')
else
    [~, dio_fname, ~] = fileparts(dio_file_struct.name);
end

run_file_struct = dir([file_path filesep fid '*_dio.run']);
if isempty(run_file_struct)
    error('no run file found')
else
    [~, run_fname, ~] = fileparts(run_file_struct.name);
end

num_exp = length(exp_dir_struct);

% all variables below this are created using the above variables
% save above varialbes to a temp file, clear all, and load them in
save([tempdir filesep 'ms2spikes_temp.mat'], '-v7.3')

for exp_i = 1:num_exp
    exp_dir_name = exp_dir_struct(exp_i).name;
    path2dio     = [file_path filesep dio_fname '.mat'];
    path2run     = [file_path filesep run_fname '.run'];
    path2ms    = [file_path filesep exp_dir_name filesep];
    mda_struct  = dir([path2ms filesep rec_fname '*firings-curated.mda']);
    phy_struct   = dir([path2ms filesep rec_fname '*.phy.dat']);
    prb_struct   = dir([path2ms filesep '*.geom.csv']);
    
    %   NOTE: probe file name must have the number of channels as the 4th and
    %   5th characters!!!
    disp('USING PROBE FILE TO GET NUMBER OF CONTACTS OF ELECTRODE!')
    disp(prb_struct.name)
    num_chan     = str2double(prb_struct.name(4:5));

    if isempty(mda_struct)
        warning(['no curated.mda file found in ' path2ms])
    elseif isempty(phy_struct)
        warning(['no raw data .mda file found in ' path2ms])
    else
        disp(['creating spikes file for ' exp_dir_name])
        phy_name  = [path2ms phy_struct.name];
        mda_name = [path2ms mda_struct.name];
        
        
        %% add basic info
        spikes.info = mda_name;
        spikes.params = 'used default MountainSort pipeline';
        
%% load in MountainSort curated spike time info
mda_spikes = readmda(mda_name);

%% create the unit ID assignment
%  in UltraMegaSorter this is spikes.assigns
fprintf('\n#####\nloading in spike assignment vector\n#####\n')
spikes.assigns = mda_spikes(3, :);

%% create the unwrapped times array
%  in UltraMegaSorter this is spikes.unwrapped_times
fprintf('\n#####\nloading spikes times\n#####\n')
spk_inds = mda_spikes(2, :); % each entry is the sample index when a spike occured
spikes.unwrapped_times = single(spk_inds)/30000; % divide by sampling rate to get spike times in seconds

%% how to create the nx2 array where each unique cluster id is paired with its cluster group
%  that is the unit in the noise, multiunit, good, and unclustered
%  category. this is the same format as in the UltraMegaSorter spike
%  structure output.
%  MountainSort NOTE: only accepted units are included in the curated
%  output. Don't know about MUA...but we shouldn't use MUA with such a good
%  spike sorter.
fprintf('\n#####\nmatching unit IDs with cluster IDs\n#####\n')
cluster_ids  = unique(mda_spikes(3, :));
labels       = nan(length(cluster_ids), 2);
best_channel = nan(length(cluster_ids), 2);

for k = 1:length(cluster_ids)
    labels(k, :) = [cluster_ids(k), 1]; % 1 for all single units
    % find first index for cluster ID k
    cluster_inds = find(mda_spikes(3, :) == cluster_ids(k), 2); % 1:MUA, 2:single-unit
    best_channel(k, :) = [cluster_ids(k), mda_spikes(1, cluster_inds)];
end

spikes.labels       = labels;
spikes.best_channel = best_channel;
spikes.peak_channel = mda_spikes(1, :);
clear labels best_channel

%% create the following arrays
%  trials, stimuli, spiketimes <-- length equals length of entire experiment
%  stimsequence, stimulus_sample_num, stimulus_times <-- length equals
%  number of trials
%  in UltraMegaSorter this is spikes.trials
fprintf('\n#####\nloading digital input lines to ID trial times and condition types\n#####\n')

load(path2dio)
run = load(path2run, '-mat');

num_samples = length(dio.timestamps);
% could the channel index be or OUT OF ORDER???
if strcmp(dio.channelData(1).id, 'Din1') == 0
    warning('***** Channel 1 does not equal Din1 *****')
else
    % if it is ever out of order add a loop here to find the index of the
    % correct channel and use it below. Then add this type of check anywhere
    % readTrodesFileDigitalChannels is used to extract digital inputs/outputs
    fprintf('\n#####\nDIO Channel 1 equals Din1\n#####\n')
end
state_change_inds   = find(diff(dio.channelData(stim_ind.trial_boolean).data) ~= 0) + 1; % indices of all state changes
num_state_changes   = length(state_change_inds);
disp(['number of state changes: ' num2str(num_state_changes)])
trials              = zeros(num_samples, 1, 'single');
stimuli             = zeros(num_samples, 1, 'single');
stimsequence        = run.stimsequence;
stimulus_times      = run.stimulus_times;
stimulus_sample_num = zeros(num_state_changes/2, 2, 'single');
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
        
        if dynamic_time == 0
            ind0 = state_change_inds(k) - time_before*30000;   % start time index
            ind1 = state_change_inds(k+1) + time_after*30000; % stop time index
        else
            ind0 = state_change_inds(k) - dt_before_after(trial_count);   % start time index
            ind1 = state_change_inds(k+1) + dt_before_after(trial_count + 1); % stop time index
        end
        stimulus_sample_num(trial_count, :) = [state_change_inds(k), state_change_inds(k+1)]; % get index of stim start/start (i.e. object starts moving into place and when it starts leaving)
        trials(ind0:ind1) = trial_count;

        % determine what stimulus was presented by counting the number of high
        % pusles on the second digital input line.
        num_pulses = length(find(diff(dio.channelData(stim_ind.stim_id).data(ind0:ind1)) < 0));
        stimuli(ind0:ind1) = num_pulses;
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
%     raw_data(r, :) = single(neuro_filt(raw_data(r, :)));
    raw_data(r, :) = int16(neuro_filt(raw_data(r, :)));
    progressbar(r/num_chan)
end
progressbar(1)

% get 15 samples before and 45 samples after (2ms worth of data)
num_units       = size(spikes.labels, 1);
num_spikes      = length(spikes.assigns);
% waveforms       = zeros(num_spikes, 60, num_chan, 'single');
waveforms       = zeros(num_spikes, 60, num_chan, 'int16'); % may not need SINGLE...make sure this works!!!
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
if exist([path2ms filesep fid '-e' num2str(exp_i) '-spikes.mat'], 'file') == 1
    delete([path2ms filesep fid '-e' num2str(exp_i) '-spikes.mat'])
end

fprintf(['\n#####\nSaving Data for ' fid '-e' num2str(exp_i) '\n#####\n'])
save([path2ms filesep fid '-e' num2str(exp_i) '-spikes.mat'], 'spikes', '-v7.3')

% clear all variables and load in variables in the temp file for the net iteration

if exp_i ~= num_exp
    fprintf('\n#####\nCLEARING DATA\n#####\n')
    clear all
    load([tempdir filesep 'ms2spikes_temp.mat'])
end
    end % end else
end % end for loop

clear all
