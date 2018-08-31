%%SPKGAD2DIO.m Opens .rec file and saves digital input and output values to
%%a mat file for quick loading in other functions.
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20161205
%
%   EDIT: 20170518
%   spkgad2flatarray now calls this simplified version of spkgad2dio. all
%   parameters in the user input section are now saved to the dio.mat file.
%%

rec_file_struct = dir([fpath filesep fid '*.rec']);
if isempty(rec_file_struct)
    error('no .rec file found')
else
    [~, rec_fname, ~] = fileparts(rec_file_struct.name);
end

% dio channel map
if jb_behavior == 0
    stim_ind.trial_boolean = 1;
    stim_ind.stim_id       = 2;
    stim_ind.running       = 4;
    stim_ind.led1          = 5;
    stim_ind.led2          = 6;
    stim_ind.camera        = 14;
elseif jb_behavior == 1
    stim_ind.jitter        = 7;
    stim_ind.trial_boolean = 9;
    stim_ind.stim_id       = 10;
    stim_ind.running       = 11;
    stim_ind.licking       = 12;
    stim_ind.camera        = 14;
    stim_ind.puffer        = 15;
    stim_ind.led1          = 5;
end

path2rec     = [fpath filesep rec_fname '.rec'];

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
dio = readTrodesFileDigitalChannels(path2rec);

if neuro
    save([fpath filesep fid '_dio.mat'], 'dio',...
        'echan_num', 'probe_type', 'dynamic_time', 'time_before', 'time_after',...
        'jb_behavior', 'stim_ind', 't_after_stim', 'stim_duration', ...
        '-v7.3')
else
    save([fpath filesep fid '_dio.mat'], 'dio',...
        'dynamic_time', 'time_before', 'time_after',...
        'jb_behavior', 'stim_ind', 't_after_stim', 'stim_duration', ...
        '-v7.3')
end




