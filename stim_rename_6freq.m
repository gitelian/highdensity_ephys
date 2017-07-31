%STIM_RENAME_6FREQ.m Rename stimulus IDs for 6 frequency stimulation experiment.
%   Loads .run file and changes the stimsequence IDs. Each run is an 8pos
%   dpdl experiment with no object. Therefore each experiment only has 3
%   types of trials (no light, light 1, and light 2). The number of trials
%   per run is requirred.



[fname, fpath, ~] = uigetfile('/media/greg/data/behavior/*.run');
load([fpath filesep fname], '-mat')
stimsequence_original = stimsequence;

trial_per_run = 135;
run = 1;
count = 0;

for k = 1:length(stimsequence)
    stim_id = stimsequence(k);
    % rename stim ID
    if stim_id <= 9
        stim_id_temp = 1;
    elseif stim_id >= 10 && stim_id <= 18
        stim_id_temp = 2;
    elseif stim_id >= 19
        stim_id_temp = 3;
    end
    
    stim_id_temp = 3*(run - 1) + stim_id_temp;
    stimsequence(k) = stim_id_temp;
    
    count = count + 1;
    if count == trial_per_run
        run = run + 1;
        count = 0;
    end
end

save([fpath filesep fname], 'run_cell', 'run_dist', 'stimsequence', 'stimulus_times', 'stimsequence_original', '-v7.3')

