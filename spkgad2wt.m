

function spkgad2wt()

%% load digital line data
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
    path2rec          = [file_path filesep rec_fname '.rec'];
end

wt_file_struct = dir([file_path filesep fid '*wt.mat']);
if isempty(wt_file_struct)
    error('no whisker tracking file found')
else
    [~, wt_fname, ~] = fileparts(wt_file_struct.name);
    path2wt          = [file_path filesep wt_fname '.mat'];
end

% load tracked mat file and rearrance data into wt matrix.
% spikes2neo expects a matrix with all the whisking data
fprintf('\n#####\nloading whisker tracking data and organizing into one matrix\n#####\n');
load(path2wt)
wt = [ang, sp, amp, phs, vel, wsk];

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

%% find HSV ttl pulses and assign timestamps to whisker tracking data

fprintf('\n#####\nfinding HSV ttl pulses\n#####\n');
% there are some errant edges at the beginning due to noise caused by the HSV
% smoothing first fixes this
hsv_ttl = dio.channelData(5).data;
hsv_ttl_filt = sgolayfilt(double(hsv_ttl), 4, 31);
frame_inds   = find(diff(hsv_ttl_filt > 0.5) == 1)+1;

frame_diff = length(frame_inds) - length(wt);
if abs(frame_diff) > 1
    error('number of ttl pulses and wt frames NOT EQUAL')
elseif frame_diff = 1
   frame_inds(1) = [];
elseif frame_diff == -1
     frame_inds(1) = [];
end

%%
trial_count     = 1;
progressbar('splitting whisker tracking data by trials')

for k = 1:2:num_state_changes - 1 % jumping by 2 will always select the start time with k and the stop time with k+1
    ind0 = state_change_inds(k) - dt_before_after(trial_count);   % start time index
    ind1 = state_change_inds(k+1) + dt_before_after(trial_count + 1); % stop time index

    % determine what stimulus was presented by counting the number of high
    % pusles on the second digital input line.
    num_pulses                     = length(find(diff(dio.channelData(2).data(ind0:ind1)) < 0));
    stimsequence(trial_count)      = num_pulses;
    stimulus_times(trial_count, :) = (double([state_change_inds(k), state_change_inds(k+1)]) - double(ind0))/30000; % gets time of stimulus start

    % the HSV signal is sampled at 500Hz Need to calculate the corresponding
    % index for the slowly sampled data.
    temp_hsv_ind0 = find(frame_inds >= ind0, 1,'first');
    temp_hsv_ind1 = find(frame_inds <= ind1, 1, 'last');
    wt_cell{trial_count, 1} = wt(temp_hsv_ind0:temp_hsv_ind1, :);
    progressbar(trial_count/(num_state_changes/2));
    trial_count = trial_count + 1;
end

progressbar(1)
fprintf('\n#####\nsaving data\n#####\n');
save([file_path filesep rec_fname '.wtr'], 'wt_cell', 'wt', 'stimsequence', 'stimulus_times', '-v7.3')
send_text_message(...
    '3237127849',...
    'sprint',...
    'spkgad2wt COMPLETE',...
    ['spkgad2wt for ' fname ' has finished'])
clear all





