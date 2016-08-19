These functions are for the Adesnik lab e-phys researchers. They will allow us to extract data out of our high density e-phys acquisition system and into the appropriate format for the KlustaSuite spike sorting package. After spike sorting additional code will convert the .kwik files into a "spikes.mat" structure. The spikes file is organized in the same way as the output of our old spike sorter UltraMegaSort2000. This should allow us to make minimal changes to our workflow while we transition to the new system.

This directory also contains 3rd party code from SpikeGadgets. I did not write anything in TrodesToMatlab but I provid it here for ease of access for my collegues in the Adesnik lab; it can be found here: http://www.spikegadgets.com/software/matlab_toolbox.html. Also, progressbar was written by somebody else and was publically published here: https://www.mathworks.com/matlabcentral/fileexchange/6922-progressbar
