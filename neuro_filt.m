function data_filt = neuro_filt(data)

% neuro_filt.m
% data: timeseries
%
% G.Telian
% Adesnik Lab
% UC Berkeley
% 20160909

%construct parameters for filter
Fs = 30000;
disp('using old school scotts way of filtering')
Wp = [ 800  8000] * 2 / Fs;
Ws = [ 600 10000] * 2 / Fs;
[N,Wn] = buttord( Wp, Ws, 3, 20);
[B,A] = butter(N,Wn);
data_filt = filtfilt( B, A, double(data));

end
