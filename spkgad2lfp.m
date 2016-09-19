%SPKGAD2LFP.m Write LFPs to cell array
%   SPKGAD2LFP() takes no arguements; it pulls up a dialogue box in which the user
%   is asked to select a .LFP directory inside of an experiment directory.
%   SPKGAD2LFP will look for a .rec file. Using the information from the
%   digital trial line and the digital encoder line it extracts LFPs and
%   The stimulus sequence is also saved in an total_samplesx1 vector.
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20160815
%%
main_data_path = '/media/greg/data/neuro/';
[file_path, file_name, file_ext] = fileparts(uigetdir(main_data_path, 'Select LFP folder to extract neural data'));
file_path = [file_path filesep file_name file_ext];

if file_path == 0
    error('no directory was selected')
elseif ~strcmp(file_ext, '.LFP')
    error('not a .SPK directory!')
end

[fpath, fname, ~] = fileparts(file_path);
fid = fname(1:7);
echan_num = [1,8; 9,16]; % specify the channels numbers corresponding to each electrode
num_electrodes = size(echan_num, 1);

rec_file_struct = dir([fpath filesep fid '*.rec']);
if isempty(rec_file_struct)
    error('no .rec file found')
else
    [~, rec_fname, ~] = fileparts(rec_file_struct.name);
end
path2rec = [fpath filesep rec_fname '.rec'];

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
dio                 = readTrodesFileDigitalChannels(path2rec);
num_samples         = length(dio.timestamps);
state_change_inds   = find(diff(dio.channelData(1).data) ~= 0) + 1; % indices of all state changes
num_state_changes   = length(state_change_inds);
stimsequence        = zeros(num_state_changes/2, 1);

% calculate beginning and end of trials
fprintf('\n#####\ncalculating beginning and end of trials\n#####\n');
dt_state_change = diff(state_change_inds); % computes time before each start, stop, start, ..., stop event
iti_duration    = dt_state_change(2:2:end); % -stop1 + start2 + ...
dt_before_after = uint64(iti_duration/2);
dt_before_after = [mean(dt_before_after); dt_before_after; mean(dt_before_after)];

%%
progressbar('electrodes', 'channels', 'trials')
for electrode = 1:num_electrodes
    num_channels = 4*(echan_num(electrode, 2) - echan_num(electrode, 1) + 1);
    chan_count = 1;
    for ntrode = echan_num(electrode, 1):echan_num(electrode, 2)
        for chan = 1:4
            % Load LFP data from one channel
            fprintf(['\n#####\nloading LPFs from channel: ' num2str(chan_count) '\n#####\n']);
            fext = ['.LFP_nt' num2str(ntrode) 'ch' num2str(chan) '.dat'];
            ffullname = [file_path filesep fname fext];
            data = readTrodesExtractedDataFile(ffullname);
            decimation = data.decimation;

            %% iterate through trials and get stimulus ID
            trial_count     = 1;
            for k = 1:2:num_state_changes - 1 % jumping by 2 will always select the start time with k and the stop time with k+1
                ind0 = state_change_inds(k) - dt_before_after(trial_count);   % start time index
                ind1 = state_change_inds(k+1) + dt_before_after(trial_count + 1); % stop time index

                % determine what stimulus was presented by counting the number of high
                % pusles on the second digital input line.
                num_pulses = length(find(diff(dio.channelData(2).data(ind0:ind1)) < 0));
                stimsequence(trial_count) = num_pulses;

                % the LFP signal was downsampled! Need to calculate the
                % corresponding index for the downsampled data.
                dind0 = round(ind0/decimation);
                dind1 = round(ind1/decimation);

                % on the first pass of each electrode make an lfp matrix and
                % assign it to the lfp cell. 'chan_count' counts the number of
                % electrodes iterated through on a given electrode NOT total
                % number of channels processed.
                if chan_count == 1
                    num_down_samples = dind1 - dind0 +1;
                    lfp_mat = zeros(num_down_samples, num_channels, 'int16');
                    lfp{trial_count, 1} = lfp_mat;
                end

                % add lfp to lfp matrix in lfp cell
                lfp{trial_count, 1}(:, chan_count) = data.fields.data(dind0:dind1);
                progressbar([], [],  trial_count/(num_state_changes/2))
                trial_count = trial_count + 1;
            end % trial loop
            progressbar([], chan_count/(num_channels), [])
            chan_count = chan_count + 1;
        end % chan 1-4 loop
    end % ntrode loop

    %% save data
    fprintf('\n#####\nsaving data\n#####\n');
    progressbar(electrode/num_electrodes, [], [])
    new_folder_path = [fpath filesep fid '_e' num2str(electrode)];
    if exist(new_folder_path, 'dir') == 0
        disp('Making new electrode directory')
        mkdir(new_folder_path)
    end
    lfp_fname = [fname '-e' num2str(electrode) '-LFP'];
    save([new_folder_path filesep lfp_fname  '.mat'], 'lfp', 'stimsequence', '-v7.3')
end % n electrode loop

progressbar(1)
send_text_message(...
    '3237127849',...
    'sprint',...
    'spkgad2lfp COMPLETE',...
    ['spkgad2lfp for ' fname ' has finished'])

clear all
