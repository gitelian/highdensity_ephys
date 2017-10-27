%SPKGAD2DIO_BEHAVIOR.m Extract digital line data and save to mat file
%   SPKGAD2DIO_BEHAVIOR() takes no arguements; it pulls up a dialogue box in which the user
%   is asked to select a .rec file for digital data extraction.
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20170713
%% User Input
% specify whether proceding code dynamically determines time before and
% after stimulus onset OR use the parameters below.
dynamic_time = 0;
time_before = 1.0;
time_after  = 2.0;
% time_after  = 1.2;
warning('make sure the TIME BEFORE and TIME AFTER stimulus onset is properly set!')

%% main
main_data_path = '/media/greg/data/behavior/';
[file_name, fpath, filter_index] = uigetfile([main_data_path '.rec'], 'Select .rec folder to extract digital data');
f_dir = dir([fpath 'FID*' num2str(filter_index) '*.rec']);
[~, fid, ~] = fileparts(f_dir.name); % fid is the file name

% fid = regexp(f_dir.name, 'FID\d{4}', 'match'); % any number of characters expression: .* (any character, any number of times)
% fid = fid{1};
rec_file_struct = dir([fpath filesep fid '*.rec']);
if isempty(rec_file_struct)
    error('no .rec file found')
else
    [~, rec_fname, ~] = fileparts(rec_file_struct.name);
end

path2rec = [fpath filesep rec_fname '.rec'];

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
dio = readTrodesFileDigitalChannels(path2rec);
save([fpath filesep fid '_dio.mat'], 'dio',...
    'dynamic_time', 'time_before', 'time_after',...
    '-v7.3')

disp('successfully complete')
clear all

