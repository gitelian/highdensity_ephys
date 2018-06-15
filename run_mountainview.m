



% G. Telian
% Adesnik Lab
% UC Berkeley
% 201809611

function run_mountainview()

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
    unix(['mountainview' ' '...
        '--raw=' [e_name filesep mda_raw.name] ' '...
        '--geom=' [e_name filesep prb_geom.name] ' '...
        '--firings=' [e_name filesep mda_firings.name] ' '...
        '--samplerate=30000'])
else
    warning('spike sorting either was not run or it failed. make sure it completes properly')
end

% go back to MATLAB startup directory
startup

end


