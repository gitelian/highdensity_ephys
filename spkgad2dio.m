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

% %% User Input
% 
% % specify whether proceding code dynamically determines time before and
% % after stimulus onset OR use the parameters below.
% dynamic_time = 1;
% time_before = 1.0;
% time_after  = 2.0;

%% Main Code
% main_dir_path = uigetdir('/media/greg/data/neuro/', 'Select experiment folder to extract run data');
% 
% if main_dir_path == 0
%     error('no directory was selected')
% end

% [~, fname, ~] = fileparts(main_dir_path);
% fid = fname(1:7);

rec_file_struct = dir([main_dir_path filesep fid '*.rec']);
if isempty(rec_file_struct)
    error('no .rec file found')
else
    [~, rec_fname, ~] = fileparts(rec_file_struct.name);
end


path2rec     = [main_dir_path filesep rec_fname '.rec'];

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
dio = readTrodesFileDigitalChannels(path2rec);
save([main_dir_path filesep fid '_dio.mat'], 'dio',...
    'dynamic_time', 'time_before', 'time_after',...
    'echan_num', 'probe_type',...
    '-v7.3')




