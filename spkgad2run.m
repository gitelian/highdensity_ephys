%SPKGAD2RUN.m Write run distance structure from SpikeGadgets recording.
%   SPKGAD2RUN() takes no arguements; it pulls up a dialogue box in which the user
%   is asked to select an experiment directory. SPKGAD2RUN will look for a
%   SpikeGadgets .rec file. Using the information from the digital trial
%   line and the digital encoder line it calculates total run distance and
%   stimulus ID. It saves the cummulative runspeed in a total_samplesx1
%   vector and also breaks up running epochs corresponding to trials and
%   saves them in a cell array. The stimulus sequence is also saved in a
%   total_samplesx1 vector.
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

rec_file_struct = dir([file_path filesep fid '*.rec']);
if isempty(rec_file_struct)
    error('no .rec file found')
else
    [~, rec_fname, ~] = fileparts(rec_file_struct.name);
end


path2rec     = [file_path filesep rec_fname '.rec'];

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
dio                 = readTrodesFileDigitalChannels(path2rec);
num_samples         = length(dio.timestamps);
state_change_inds   = find(diff(dio.channelData(1).data) ~= 0) + 1; % indices of all state changes
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
encoder(find(diff(dio.channelData(4).data) > 0)+1) = 1;
run_dist        = cumsum(encoder);
run_cell        = cell(num_state_changes/2, 1);

trial_count     = 1;
progressbar('extracting running distance')

for k = 1:2:num_state_changes - 1 % jumping by 2 will always select the start time with k and the stop time with k+1
    ind0 = state_change_inds(k) - dt_before_after(trial_count);   % start time index
    ind1 = state_change_inds(k+1) + dt_before_after(trial_count + 1); % stop time index

    % determine what stimulus was presented by counting the number of high
    % pusles on the second digital input line.
    num_pulses                     = length(find(diff(dio.channelData(2).data(ind0:ind1)) < 0));
    stimsequence(trial_count)      = num_pulses;
    stimulus_times(trial_count, :) = (double([state_change_inds(k), state_change_inds(k+1)]) - double(ind0))/30000; % gets time of stimulus start

    % add run distance to run cell
    run_cell{trial_count, 1} = run_dist(ind0:ind1);
    progressbar(trial_count/(num_state_changes/2));
    trial_count = trial_count + 1;
end

progressbar(1)
fprintf('\n#####\nsaving data\n#####\n');
save([file_path filesep rec_fname '.run'], 'run_cell', 'run_dist', 'stimsequence', 'stimulus_times', '-v7.3')

clear all
