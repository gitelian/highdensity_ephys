function createAllMatclustFiles()
%createAllMatclustFiles()
%This function will create matclust files for all ntrodes in one recording
%session.  It assumes that there is one '*.spikes' folder in the current working directory
%that contains the extracted spike information for all ntrodes.  Use
%'extractSpikeBinaryFiles.m' to create these files from the raw .rec file.

currDir = pwd;
filesInDir = dir;
spikesFolder = [];
for i=3:length(filesInDir)
    if filesInDir(i).isdir && ~isempty(strfind(filesInDir(i).name,'.spikes'))
        spikesFolder = filesInDir(i).name;
        break;
    end
end

dotLoc = strfind(spikesFolder,'.');
baseName = spikesFolder(1:dotLoc(end)-1);

if isempty(spikesFolder)
    error('Spikes folder not found in this directory.');
end

ranges = getEpochs(1);  %assumes that there is at least a 1-second gap in data between epochs
names{1} = '1 All points';
for r = 1:31
    if (r <= size(ranges,1))
        st =  timetrans(ranges(r,1),1,1);
        st = st{1};
        et =  timetrans(ranges(r,2),1,1);
        et = et{1};
        names{r+1,1} = [num2str(r+1),'  e',num2str(r),' ', st,'-', et];
    else
        names{r+1,1} = num2str(r+1);
    end
end
ranges = [[ranges(1,1) ranges(end,2)];ranges];

cd(spikesFolder);

datFiles = dir('*.spikes_*.dat');

if (isempty(datFiles))
    cd(currDir);
    error('No spike binary files found in spikes folder.');
end

cd(currDir);
matclustDir = [baseName,'.matclust']; 
mkdir(matclustDir);
matclustDir = fullfile(currDir,matclustDir);

cd(spikesFolder);
for datFileInd = 1:length(datFiles)
    disp(datFiles(datFileInd).name);
    createMatclustFile(datFiles(datFileInd).name,matclustDir);
end
cd(matclustDir);
save times ranges names;


cd(currDir);


function out = timetrans(times,UnitsPerSec,dir)

%out = timestrans(times, UnitsPerSec, dir)

%transforms time inputs from seconds to hours:minutes:seconds or vice versa

%TIMES is times that you want to transform, ie [60 3600] seconds or
%[0:01:00 1:00:00] in hours:min:secs
%UnitsPerSec is units in each sec
%dir is 1 if going from seconds to hours:min:sec, 2 if going from
%hrs:min:sec to sec

if (dir==1)
	
	
	
	for i = 1:length(times)
      if (times(i) > 0)
            hours = floor(times(i)/(60*60*UnitsPerSec));
	        minutes = floor(times(i)/(60*UnitsPerSec))-(hours*60);
	        seconds = floor(times(i)/(UnitsPerSec))-(hours*60*60)-(minutes*60);
      else
            hours = abs(ceil(times(i)/(60*60*UnitsPerSec)));
	        minutes = abs(ceil(times(i)/(60*UnitsPerSec))+(hours*60));
	        seconds = abs(ceil(times(i)/(UnitsPerSec))+(hours*60*60)+(minutes*60));
      end
      if (minutes<10)
          tempmin = ['0',num2str(minutes)];
      else
          tempmin = [num2str(minutes)];
      end
      if (seconds<10)
          tempseconds = ['0',num2str(seconds)];
      else
          tempseconds = [num2str(seconds)];
      end
      out{i,1} = [num2str(hours),':',tempmin,':',tempseconds];
      if (times(i)<0)
          out{i,1} = ['-',out{i,1}];
      end
    end
    
elseif (dir==2)
    for i = 1:length(times)
        t = [0 0 0 0 0 0];
        temptime = times{i};
        colons = findstr(temptime,':');
        startind = length(temptime);
        count = 6;
        for j = length(colons):-1:1
            t(count) = str2num(temptime((colons(j)+1):startind));
            startind = colons(j)-1;
            count = count-1;    
        end
        t(count) = str2num(temptime(1:colons(1)-1));
        out(i,1) = etime(t,[0 0 0 0 0 0])*UnitsPerSec;
    end
end
               

