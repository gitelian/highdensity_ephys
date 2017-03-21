

% probe remap
remap_vector = [6, 11, 3, 14, 1, 16, 2, 15, 5, 12, 4, 13, 7, 10, 8, 9];

% convert MCdata cell array into a matrix
for trial = 1:length(MCdata)
    MCdata{1, trial}(:, 1:16) = [];
    MCdata{1, trial}(:, 17) = [];
end

MCdata0 = cell2mat(MCdata');
% MCdata0 = MCdata0(:, 17:32);
clear MCdata

% re-order channels
MCdata1 = MCdata0(:, remap_vector);
clear MCdata0

% Each row is a time point
% Each column is an electrode where col1 = top, col32 = bottom
num_points = size(MCdata1,1)*size(MCdata1,2);
ROWS = size(MCdata1, 1);
COLS = size(MCdata1, 2);

% num_points = 30000*32;
flat_data = zeros(num_points, 1,  'int16');

% fliped_data = fliplr(MCdata1);
% clear MCdata1

count = 1;
for k = 1:ROWS
    for l = 1:COLS
        temp = int16(double(MCdata1(k, l))*10000);
        flat_data(count, 1) = temp;
        count = count + 1;
    end
end

fid2write = fopen('elena16.phy.dat', 'w');
fwrite(fid2write, flat_data, 'int16');
fclose(fid2write);

