
%%

main_data_path = '/media/greg/Data/Neuro/';
file_path = uigetdir(main_data_path, 'Select SPIKES folder to extract neural data');

if file_path == 0
    error('no directory was selected')
end

[fpath, fname, ~] = fileparts(file_path);
fid = fname(1:7);
echan_num = [1,8; 9,16]; % specify the channels numbers corresponding to each electrode
num_electrodes = size(echan_num, 1);
progressbar('electrodes', 'channels')

for electrode = 1:num_electrodes

    chan_count = 1;

    for ntrode = echan_num(electrode, 1):echan_num(electrode, 2)
        for chan = 1:4

            fext = ['.LFP_nt' num2str(ntrode) 'ch' num2str(chan) '.dat'];
            ffullname = [file_path filesep fname fext];
            data = readTrodesExtractedDataFile(ffullname);

            %% setup data array
            if chan_count == 1
                nsamples = length(data.fields.data);
                dtype = data.fields.type;

                if strcmpi(dtype, 'int16')
                    dmat = zeros(1, nsamples*32, 'int16');
                    % TODO add more checks for different datatypes
                else
                    error('could not identify the datatype')
                end
            end

            %% add extracted data to data array
            disp(['adding channel ' num2str(chan_count)])
            progressbar([], chan_count/32);

            dmat(1, chan_count:32:nsamples*32) = data.fields.data'/10;
            chan_count = chan_count + 1;
        end
    end

    %% save data array as a binary file
    progressbar(electrode/num_electrodes, [])
    new_folder_path = [fpath filesep fid '_e' num2str(electrode)];
    if exist(new_folder_path, 'dir') == 0
        disp('Making new electrode directory')
        mkdir(new_folder_path)
    end
    phy_dat_fname = [fname '-e' num2str(electrode)];
    fid2write = fopen([new_folder_path filesep phy_dat_fname  '.phy.dat'], 'w');
    fwrite(fid2write, dmat, 'int16');
    fclose(fid2write);

    %% add files needed by KlustaKwik
    % template files should be located in the general data directory
    % add params.prm file with experiment name to directory
    updateKKandSlurmFiles(new_folder_path, phy_dat_fname);
    % add probe file to directory
    copyfile([main_data_path '32chan.prb'], [new_folder_path filesep '32chan.prb'])

end

progressbar(1)

clear all
