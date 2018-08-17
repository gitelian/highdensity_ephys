



% G. Telian
% Adesnik Lab
% UC Berkeley
% 201809611

function run_mountainview()

% Save library paths
MatlabPath = getenv('LD_LIBRARY_PATH');

% Make Matlab use system libraries
% setenv('LD_LIBRARY_PATH', getenv('PATH'))
path1 = getenv('PATH');
path1 = [path1 ':/home/greg/code/mountain_suite/mountainview/bin'];
path1 = [path1 ':/home/greg/code/mountain_suite/mountainview/bin'];
path1 = [path1 ':/home/greg/code/mountain_suite/mountainlab/bin'];
setenv('PATH', path1)

data_dir  = '/media/greg/data/neuro/';

% select experiments to process
dir_name = uigetdir(data_dir, 'Select electrode directory to view sorted spikes');

% get directory name and path
[exp_path, e_name] = fileparts(dir_name);

% change current directory to experiment directory
cd(exp_path)

% look for raw mda file
mda_raw = dir([dir_name filesep e_name '.mda']);

% look for probe geometry file
prb_geom = dir([dir_name filesep '*geom.csv']);

% look for firings mda file
mda_firings = dir([dir_name filesep '*-firings.mda']);

if ~isempty(mda_raw) || ~isempty(prb_geom) || ~isempty(mda_firings)
    
    % run mountainview
    system(['mountainview' ' '...
        '--raw=' [e_name filesep mda_raw.name] ' '...
        '--geom=' [e_name filesep prb_geom.name] ' '...
        '--firings=' [e_name filesep mda_firings.name] ' '...
        '--samplerate=30000'])
else
    warning('spike sorting either was not run or it failed. make sure it completes properly')
end

% Reassign old library paths
setenv('LD_LIBRARY_PATH', MatlabPath)

% go back to MATLAB startup directory
startup

end


