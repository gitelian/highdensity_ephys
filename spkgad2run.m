%SPKGAD2RUN.m Write run distance structure from SpikeGadgets recording.
%   SPKGAD2RUN() takes no arguements; it pulls up a dialogue box in which the user
%   is asked to select an experiment directory. SPKGAD2RUN will look for a
%   SpikeGadgets .rec file. Using the information from the digital trial
%   line and the digital encoder line it calculates total run distance and
%   stimulus ID. It saves the cummulative runspeed in a total_samplesx1
%   vector and also breaks up running epochs corresponding to trials and
%   saves them in a cell array. The stimulus sequence is also saved in a
%   total_samples x 1 vector.
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20160815
%%

file_path = uigetdir('/media/greg/data/neuro/', 'Select experiment folder to extract run data');

if file_path == 0
    error('no directory was selected')
end

[~, fname, ~] = fileparts(file_path);
fid = fname(1:7);

% rec_file_struct = dir([file_path filesep fid '*.rec']); % change to *_dio.mat; change rec to dio
dio_file_struct = dir([file_path filesep fid '*_dio.mat']); % change to *_dio.mat; change rec to dio
if isempty(dio_file_struct)
    error('no dio file found')
else
    [~, dio_fname, ~] = fileparts(dio_file_struct.name);
end

path2dio     = [file_path filesep dio_fname '.mat']; % change to .mat

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
% dio                 = readTrodesFileDigitalChannels(path2rec); % replace with load(path2dio)
load(path2dio)

% dio channel map
if jb_behavior == 0
    stim_ind.trial_boolean = 1;
    stim_ind.stim_id       = 2;
    stim_ind.running       = 4;
elseif jb_behavior == 1
    stim_ind.trial_boolean = 9;
    stim_ind.stim_id       = 10;
    stim_ind.running       = 11;
    stim_ind.LED_opto      = 12;
    stim_ind.licking       = 13;
    
    % this will be the index for the end of the last trial.
    % used to find stimulus IDs that happend between end of last trial and end of the current trial.
    last_trial_index = 0;
end

num_samples         = length(dio.timestamps);
state_change_inds   = find(diff(dio.channelData(stim_ind.trial_boolean).data) ~= 0) + 1; % indices of all state changes
num_state_changes   = length(state_change_inds);
stimsequence        = zeros(num_state_changes/2, 1);
stimulus_times      = zeros(num_state_changes/2, 2, 'single');

% calculate beginning and end of trials
fprintf('\n#####\ncalculating beginning and end of trials\n#####\n');
dt_state_change = diff(state_change_inds); % computes time before each start, stop, start, ..., stop event
iti_duration    = dt_state_change(2:2:end); % -stop1 + start2 + ...
dt_before_after = uint64(iti_duration/2);
dt_before_after = [mean(dt_before_after); dt_before_after; mean(dt_before_after)];

% load running encoder data and calculate distance traveled
fprintf('\n#####\nloading encoder data and calculating distance traveled\n#####\n');
encoder         = zeros(num_samples, 1);
encoder(find(diff(dio.channelData(stim_ind.running).data) > 0)+1) = 1;
run_dist        = cumsum(encoder);
run_cell        = cell(num_state_changes/2, 1);

% make a lick and led_opto cell if licking data is available
if jb_behavior == 1
    lick_cell = cell(num_state_changes/2, 1);
end

trial_count     = 1;
progressbar('extracting running distance')

for k = 1:2:num_state_changes - 1 % jumping by 2 will always select the start time with k and the stop time with k+1
    if dynamic_time == 0
        ind0 = state_change_inds(k) - time_before*30000;  % start time index
        ind1 = state_change_inds(k+1) + time_after*30000; % stop time index
    else
        ind0 = state_change_inds(k) - dt_before_after(trial_count);   % start time index
        ind1 = state_change_inds(k+1) + dt_before_after(trial_count + 1); % stop time index
    end
    
    % determine what stimulus was presented by counting the number of high
    % pusles on the second digital input line.
    if jb_behavior == 0
        num_pulses                     = length(find(diff(dio.channelData(stim_ind.stim_id).data(ind0:ind1)) < 0));
        stimsequence(trial_count)      = num_pulses;
        stimulus_times(trial_count, :) = (double([state_change_inds(k), state_change_inds(k+1)]) - double(ind0))/30000; % gets time of stimulus/trial start
        
    elseif jb_behavior == 1
        
        % led_opto should be 0 if LED was off and 1 if LED was on
        led_opto = length(find(diff(dio.channelData(stim_ind.LED_opto).data(last_trial_index:ind1) > 0)));
        
        % offset stim IDs by 9 to indicate LED/optogenetic trials vs no stim trials
        if led_opto == 1
            stim_offset = 9;
        elseif led_opto == 0
            stim_offset = 0;
        end
        
        % determine which stimulus was presented or if it was a catch trial
        num_pulses  = length(find(diff(dio.channelData(stim_ind.stim_id).data(last_trial_index:ind1)) < 0));
        high_times  = find(diff(dio.channelData(stim_ind.stim_id).data(last_trial_index:ind1)) > 0);
        low_times   = find(diff(dio.channelData(stim_ind.stim_id).data(last_trial_index:ind1)) < 0);
        diff_times  = (low_times - high_times)/30000;
        catch_trial = find(diff_times > 6);
        
        if catch_trial
            stimsequence(trial_count) = 9 + stim_offset;
        else
            stimsequence(trial_count) = num_pulses + stim_offset;
        end
        
        % get the lick indices for this trial subtract off the trial start
        % time and convert to seconds
        lick_times = (find(diff(dio.channelData(stim_ind.licking).data(ind0:ind1) < 0)) - state_change_inds(k))/30000;
        % add lick times to lick cell
        lick_cell{trial_count, 1} = lick_times;
        
        % save the trial start and stop times (dio channel 1/9)
        stimulus_times(trial_count, :) = (double([state_change_inds(k), state_change_inds(k+1)]) - double(ind0))/30000; % gets time of stimulus/trial start
        
        % update last trial index
        last_trial_index = ind1;
    end

    % add run distance to run cell
    run_cell{trial_count, 1} = run_dist(ind0:ind1);
    progressbar(trial_count/(num_state_changes/2));
    trial_count = trial_count + 1;
    
end

progressbar(1)
fprintf('\n#####\nsaving data\n#####\n');
save([file_path filesep dio_fname '.run'], 'run_cell', 'run_dist', 'stimsequence', 'stimulus_times', '-v7.3')

clear all
