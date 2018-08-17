%SPKGAD2FLATARRAY.m Combine raw data files into a flattened array for spike
%   sorting. SPKGAD2FLATARRAY will prompt the user for a spikes directory.
%   This directory should contain the raw data saved on each channel of the
%   electrode. It will combine this data into a flattened array (e.g.
%   chan01time01, chan02time01,..., chan32time01, chan01time02,
%   chan02time02,..., chan32timeN). Different electrodes should be
%   specified in the echan_num matrix where each row contains the beginning
%   and end 'trode' numbers containing the data for that electrode. For
%   example, if the first electrode has 32 channels it will be broken up
%   into 8 trodes, this is specified by having a trode number 1 and 8.
%   A directory will be created in the experiment file for each electrode.
%
%   G. Telian
%   Adesnik Lab
%   UC Berkeley
%   20160815
%
%   EDIT: 20170518
%   spkgad2flatarray now calls a simplified version of spkgad2dio. all
%   parameters in the user input section are now saved to the dio.mat file.
%%

%% User Input
% specify the trode-channels numbers corresponding to each electrode
% echan_num = [1,8; 9,16]; % a1x32 (two)
% echan_num = [1,2; 3,4]; % buzaki-16 2 shank
% echan_num = [1,8]; % a1x32 (one)
% echan_num = [1,16];      % lbl64 (one)
% echan_num = [1,4];
% echan_num = [1,8; 9,12]; % a1x32, a1x16
echan_num = [1,4; 5,12]; % a1x16, a1x32
%echan_num = [1,4; 5,8];  % a1x16, a1x16

% number of probes used
% probe_type = {'a1x32-poly2', 'a1x32-poly2'};
% probe_type = {'a1x32-poly2'};
% probe_type = {'a1x16-buzk2'};
% probe_type = {'a1x32-poly2', 'a1x32-linear'};
% probe_type = {'lbl64_batch02'};
% probe_type = {'a1x16-linear'};
probe_type = {'a1x16-linear', 'a1x32-poly2'};
% probe_type = {'a1x16-linear', 'a1x32-poly2'};
% probe options: a1x16-linear, a1x32-linear, a1x32-poly2, Not ready: cnt64

% specify whether proceding code dynamically determines time before and
% after stimulus onset OR use the parameters below.
dynamic_time = 0;
time_before = 2.0; % 1 (8 pos)
time_after  = 2.0; % 2 (8 pos)
jb_behavior = 1;
warning('make sure the TIME BEFORE and TIME AFTER stimulus onset is properly set!')

%% Main Code
main_data_path = '/media/greg/data/neuro/';
[main_dir_path, file_name, file_ext] = fileparts(uigetdir(main_data_path, 'Select SPIKES folder to extract neural data'));
dir_path = dir([main_dir_path filesep file_name filesep '*SPK']);
file_path = [main_dir_path filesep file_name filesep dir_path.name];

if isempty(dir_path)
    error('not a .SPK directory!')
end

fpath = dir_path.folder;
fname = dir_path.name;
fid = fname(1:7);
num_chan =  (echan_num(:,2)-echan_num(:,1)+1)*4;
num_electrodes = size(echan_num, 1);
progressbar('electrodes', '.dat channels', '.mda channels')

for electrode = 1:num_electrodes

    %% create flat array .dat file for waveform extraction
    %  the mda file created by mountainsort is too big to load in to memory
    %  for this reason we have to make two files containing the exact same
    %  data...its a waste but we need the waveforms.

    chan_count = 1;

    for ntrode = echan_num(electrode, 1):echan_num(electrode, 2)
        for chan = 1:4

            fext = ['.LFP_nt' num2str(ntrode) 'ch' num2str(chan) '.dat'];
            ffullname = [file_path filesep fid fext];
            data = readTrodesExtractedDataFile(ffullname);

            %% setup data array
            if chan_count == 1
                nsamples = length(data.fields.data);
                dtype = data.fields.type;

                if strcmpi(dtype, 'int16')
                    dmat = zeros(1, nsamples*num_chan(electrode), 'int16');
                    % TODO add more checks for different datatypes
                else
                    error('could not identify the datatype')
                end
            end

            %% add extracted data to data array
            disp(['adding channel ' num2str(chan_count)])
            progressbar([], chan_count/num_chan(electrode));

            dmat(1, chan_count:num_chan(electrode):nsamples*num_chan(electrode)) = data.fields.data'; %/10;
            chan_count = chan_count + 1;
        end
    end

% save data array as a binary file
%     progressbar(electrode/num_electrodes, [])
    new_folder_path = [fpath filesep fid '_e' num2str(electrode)];
%    new_folder_path = [fpath filesep fid '_e' num2str(electrode) filesep 'mountainsort_test'];

    if exist(new_folder_path, 'dir') == 0
        disp('Making new electrode directory')
        mkdir(new_folder_path)
    end
    phy_dat_fname = [fid '_e' num2str(electrode)];
    fid2write = fopen([new_folder_path filesep phy_dat_fname  '.phy.dat'], 'w');
    fwrite(fid2write, dmat, 'int16');
    fclose(fid2write);
    
    %% create mda file for MountainSort
    chan_count = 1;

    for ntrode = echan_num(electrode, 1):echan_num(electrode, 2)
        for chan = 1:4

            fext = ['.LFP_nt' num2str(ntrode) 'ch' num2str(chan) '.dat'];
            ffullname = [file_path filesep fid fext];
            data = readTrodesExtractedDataFile(ffullname);

            %% setup data array
            if chan_count == 1
                nsamples = length(data.fields.data);
                dtype = data.fields.type;
                if strcmpi(dtype, 'int16')
                    dmat = zeros(num_chan(electrode), nsamples, 'int16');
                else
                    error('could not identify the datatype')
                end
            end

            %% add extracted data to data array
            disp(['adding channel ' num2str(chan_count)])
            progressbar([], [], chan_count/num_chan(electrode));

            dmat(chan_count, :) = data.fields.data(:)'; %/10; % REMOVE NSAMPLES
            chan_count = chan_count + 1;
        end
    end

    % save data array as a mda file
    progressbar(electrode/num_electrodes, [], [])

    % write to mda format with mountainlab matlab code
    phy_dat_fname = [fid '_e' num2str(electrode)];
    fid2write = [new_folder_path filesep phy_dat_fname  '.mda'];
    writemda16i(dmat, fid2write);
    
    clear dmat

    %% add files needed by Mountainsort

    % TODO: select probe file based on electrode configuration not just
    % number of channels used

    % template files should be located in the general data directory
    % add params.prm file with experiment name to directory
    prb_file = dir([main_data_path probe_type{electrode} '*.geom.csv']);

    if isempty(prb_file)
        error('probe file NOT FOUND')
    % if prb_file has more than 1 entry then the probe name was ambiguous
    % and it found more than one file with similar names
    elseif length(prb_file) > 1
        error('prb_file name is AMBIGUOUS')
    else
        prb_file = prb_file.name;
    end

%     num_channels = (echan_num(electrode, 2) - echan_num(electrode, 1) + 1)*4;
%     % prb_file has to be the FULL file name!
%     updateKKandSlurmFiles(new_folder_path, phy_dat_fname, prb_file, num_channels);

    % add probe file to directory
    copyfile([main_data_path prb_file], [new_folder_path filesep prb_file])
    
    % copy params.json to the main experiment directory??? or just leave it
    % somewhere common?

end

progressbar(1)

%% Open and save dio to .mat file as well as user set parameters above

fprintf('\n#### Running spkgad2dio #####\n')
try
    spkgad2dio
    fprintf('\n#### spkgad2dio worked! ####\n')
catch
    error('#### spkgad2dio did not work! ####\n')
end

disp(fid)
clear all





