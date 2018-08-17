



% G. Telian
% Adesnik Lab
% UC Berkeley
% 201809610

function run_mountainsort()

data_dir  = '/media/greg/data/neuro';
json_path = [data_dir filesep 'params.json'];
mlp_path  = [data_dir filesep 'mountainsort3.mlp'];

% need to switch paths to use system library rather than MATLABs own
% https://stackoverflow.com/questions/28460848/matlab-system-command
% Save library paths
MatlabPath = getenv('LD_LIBRARY_PATH');

% Make Matlab use system libraries
% setenv('LD_LIBRARY_PATH', getenv('PATH'))
path1 = getenv('PATH');
path1 = [path1 ':/home/greg/code/mountain_suite/mountainview/bin'];
path1 = [path1 ':/home/greg/code/mountain_suite/mountainview/bin'];
path1 = [path1 ':/home/greg/code/mountain_suite/mountainlab/bin'];
setenv('PATH', path1)

% select experiments to process
dir_cell = uigetdir2(data_dir, 'Select experiment directories to spike sort');

% iterate through all experiment directories
% process in serial

for k_dir = 1:length(dir_cell)
    
    % change to current experiment directory
    cd(dir_cell{k_dir})
    
    % look for electrode directories
    electrode_dirs = dir([dir_cell{k_dir} filesep '*_e*']);
    
    if ~isempty(electrode_dirs)
        
        % iterate through electrode directories
        for kk_dir = 1:length(electrode_dirs)
            
            % store electrode directory name
            e_dir = electrode_dirs(kk_dir).name;
            
            % look for probe geometry file
            prb_geom = dir([e_dir filesep '*geom.csv']);
            
            % look for mda file
            mda_cell = dir([e_dir filesep e_dir '.mda']);
            
            % verify they are not empty
            
            if length(prb_geom) == 1 && length(mda_cell) == 1
                SORT = 1;
            else
                SORT = 0;
                disp(['Missing either mda or geom.csv files for ' e_dir])
            end
            
            if SORT
                % call mountainsort
                system( ['mlp-run ' mlp_path ' sort '...
                    '--raw=' e_dir filesep mda_cell.name ' '...
                    '--geom=' e_dir filesep prb_geom.name ' '...
                    '--firings_out=' e_dir filesep e_dir '-firings.mda' ' '...
                    '--samplerate=30000' ' '...
                    '--_params=' json_path ' '...
                    '--curate=true'])
                %             system( ['mlp-run mountainsort3.mlp sort '...
                %                 '--raw=' e_dir filesep mda_cell.name ' '...
                %                 '--geom=' e_dir filesep prb_geom.name ' '...
                %                 '--firings_out=' e_dir filesep e_dir '-firings.mda' ' '...
                %                 '--samplerate=30000' ' '...
                %                 '--_params=' json_path ' '...
                %                 '--curate=true'])
            end

        end
    else
        disp(['EMPTY: ' dir_cell{k_dir}])
    end
    
end

% Reassign old library paths
setenv('LD_LIBRARY_PATH', MatlabPath)

end


