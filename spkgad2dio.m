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


path2rec     = [fpath filesep rec_fname '.rec'];

% load trial digital line and find trial start and end indices
fprintf('\n#####\nloading trial digital line and finding trial start and end indices\n#####\n');
dio = readTrodesFileDigitalChannels(path2rec);
save([fpath filesep fid '_dio.mat'], 'dio',...
    'dynamic_time', 'time_before', 'time_after',...
    'echan_num', 'probe_type',...
    '-v7.3')




