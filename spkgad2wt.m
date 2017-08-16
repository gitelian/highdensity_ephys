

function spkgad2wt()

%% load digital line data
file_path = uigetdir('/media/greg/data/neuro/', 'Select experiment folder to extract whisker tracking data');

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

dio_file_struct = dir([file_path filesep fid '*_dio.mat']); % change to *_dio.mat; change rec to dio
if isempty(dio_file_struct)
    error('no dio file found')
else
    [~, dio_fname, ~] = fileparts(dio_file_struct.name);
end

wt_file_struct = dir([file_path filesep fid '*wt.mat']);
if isempty(wt_file_struct)
    error('no whisker tracking file found')
else
    [~, wt_fname, ~] = fileparts(wt_file_struct.name);
    path2wt          = [file_path filesep wt_fname '.mat'];
end

run_file_struct = dir([file_path filesep fid '*_dio.run']);
if isempty(run_file_struct)
    error('no run file found')
else
    [~, run_fname, ~] = fileparts(run_file_struct.name);
    path2run          = [file_path filesep run_fname '.run'];
end

% load tracked mat file and rearrange data into wt matrix.
% spikes2neo expects a matrix with all the whisking data
fprintf('\n#####\nloading whisker tracking data\n#####\n');
load(path2wt)
run = load(path2run, '-mat');

% wt = [ang, sp, amp, phs, vel, wsk];

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
% dio                 = readTrodesFileDigitalChannels(path2rec);
path2dio     = [file_path filesep dio_fname '.mat']; % change to .mat
load(path2dio)
num_samples         = length(dio.timestamps);
state_change_inds   = find(diff(dio.channelData(1).data) ~= 0) + 1; % indices of all state changes
num_state_changes   = length(state_change_inds);
stimsequence        = run.stimsequence;
stimulus_times      = run.stimulus_times;

% calculate beginning and end of trials
fprintf('\n#####\ncalculating beginning and end of trials\n#####\n');
dt_state_change = diff(state_change_inds); % computes time before each start, stop, start, ..., stop event
iti_duration    = dt_state_change(2:2:end); % -stop1 + start2 + ...
dt_before_after = uint64(iti_duration/2);
dt_before_after = [mean(dt_before_after); dt_before_after; mean(dt_before_after)];

%%
trial_count = 1;
progressbar('splitting whisker tracking data by trials')
wt_cell = cell(num_state_changes/2, 1);

for k = 1:2:num_state_changes - 1 % jumping by 2 will always select the start time with k and the stop time with k+1
    
    if dynamic_time
        ind0 = state_change_inds(k) - time_before*30000;   % start time index
        ind1 = state_change_inds(k+1) + time_after*30000; % stop time index
    else
        ind0 = state_change_inds(k) - dt_before_after(trial_count);   % start time index
        ind1 = state_change_inds(k+1) + dt_before_after(trial_count + 1); % stop time index
    end
    
    % determine what stimulus was presented by counting the number of high
    % pusles on the second digital input line.
    num_pulses                     = length(find(diff(dio.channelData(2).data(ind0:ind1)) < 0));
    
%     % the HSV signal is sampled at 500Hz Need to calculate the corresponding
%     % index for the slowly sampled data.
%     temp_hsv_ind0 = find(frame_inds >= ind0, 1,'first');
%     temp_hsv_ind1 = find(frame_inds <= ind1, 1, 'last');
%     frame_cell{trial_count, 1} = temp_hsv_ind0:temp_hsv_ind1;
%     wt_cell{trial_count, 1} = wt(temp_hsv_ind0:temp_hsv_ind1, :);
    
    wt_cell{trial_count, 1} = [ang{trial_count, 1}, sp{trial_count, 1},...
        amp{trial_count, 1}, phs{trial_count, 1}, vel{trial_count, 1}, wsk{trial_count, 1}];
    progressbar(trial_count/(num_state_changes/2));
    trial_count = trial_count + 1;
end

progressbar(1)
fprintf('\n#####\nsaving data\n#####\n');
save([file_path filesep rec_fname '.wtr'], 'wt_cell', 'stimsequence',...
    'stimulus_times', '-v7.3')

clear all





