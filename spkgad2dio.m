%%SPKGAD2DIO.m Opens .rec file and saves digital input and output values to
%%a mat file for quick loading in other functions.
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20161205
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
dio = readTrodesFileDigitalChannels(path2rec);
save([file_path filesep fid '_dio.mat'], 'dio', '-v7.3')