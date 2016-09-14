function data_filt = genButterFilter(data, Fc1, Fc2, order, filterType,Fs)

% genButterFilter
% data: timeseries
% Fc1: low cutoff frequency. default = 600;
% Fc2: high cutoff frequency. default = 6000;
% order: filter order. default = 4;
% Fs: sampline frequency used to collect time series. default = 30000
% filterType: 'butter_causal', 'highpass', 'butter_acausal'
% 
%If only one argument is given genButterFilter will use the default values
% designed to filter neural data with a 4th order acausal butterworth
% filter between 600 and 6000Hz.
% 
% When the highpass argument is given any value can be entered for Fc2
% because it will be ignored.
%
% G.Telian
% Ritt Lab
% Boston University
% 2013

if nargin < 2
    %disp('Using default settings: Fc1 = 600, Fc2 = 6000, order = 4, filter_type = butter_causal')
%     order   = 4;    % Order
%     Fc1 = 600;   % First Cutoff Frequency
%     Fc2 = 6000;  % Second Cutoff Frequency
%     h  = fdesign.bandpass('N,F3dB1,F3dB2', order, Fc1, Fc2, 30000);
%     Hd = design(h, 'butter');
%     data_filt = filter(Hd, data);
    
    %construct parameters for filter
    disp('using old school scotts way of filtering')
    Wp = [ 800  8000] * 2 / Fs;
    Ws = [ 600 10000] * 2 / Fs;
    [N,Wn] = buttord( Wp, Ws, 3, 20);
    [B,A] = butter(N,Wn);
    filtfilt( B, A, data)
    return
end

if strcmp(filterType,'highpass')
    h = fdesign.highpass('N,F3db',order,Fc1,Fs);
    Hd = design(h,'butter');
    data_filt = filtfilt(Hd.sosMatrix,Hd.Scalevalues,data);
elseif strcmp(filterType,'butter_acausal')
    h = fdesign.bandpass('N,F3dB1,F3dB2',order,Fc1,Fc2,Fs);
    Hd = design(h,'butter');
    data_filt = filtfilt(Hd.sosMatrix,Hd.ScaleValues,data);
elseif strcmp(filterType,'butter_causal')
    h = fdesign.bandpass('N,F3dB1,F3dB2', order,Fc1,Fc2,Fs);
    Hd = design(h,'butter');
    data_filt = filter(Hd,data);
end
end