



function updateKKandSlurmFiles(exp_path, exp_name, prb_file)
%% update params.prm file
fid = fopen('/media/greg/data/neuro/params.prm', 'r');
i = 1;
tline = fgetl(fid);
txt_cell{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    if strfind(tline, 'prb_file') % update param file with probe file to use
        txt_cell{i} = ['prb_file = ' sprintf('\''') prb_file sprintf('\''')];
    elseif strfind(tline, 'experiment_name') & strcmp(tline(1:3), 'exp') % update param file with experiment name
        txt_cell{i} = ['experiment_name = ' sprintf('\''') exp_name sprintf('\''')]; % no .phy.dat extensions
    else
        txt_cell{i} = tline;
    end
end
fclose(fid);

% Write cell txt_cell into txt file in appropriate electrode directory
fid = fopen([exp_path filesep 'params.prm'], 'w');
for i = 1:numel(txt_cell)
    if txt_cell{i+1} == -1
        fprintf(fid,'%s', txt_cell{i});
        break
    else
        fprintf(fid,'%s\n', txt_cell{i});
    end
end
fclose(fid);

%% update slurm file sk.sh
fid = fopen('/media/greg/data/neuro/sk.sh', 'r');
i = 1;
tline = fgetl(fid);
txt_cell{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    if strfind(tline, '#SBATCH --job-name')
        txt_cell{i} = ['#SBATCH --job-name=' exp_name];
    else
        txt_cell{i} = tline;
    end
end
fclose(fid);

% Write cell txt_cell into txt file in appropriate electrode directory
fid = fopen([exp_path filesep 'sk.sh'], 'w');
for i = 1:numel(txt_cell)
    if txt_cell{i+1} == -1
        fprintf(fid,'%s', txt_cell{i});
        break
    else
        fprintf(fid,'%s\n', txt_cell{i});
    end
end


