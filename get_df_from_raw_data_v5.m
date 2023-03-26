function [output_dF]=get_df_from_raw_data_v5(all_dF,baseline2,params)
% get dF and t after lowpass/ z score etc

%set parameters
LFcut = 2; % cut-off frequency of lowpass filter
order = 4; % N-th order for butterworth filter
Smth=params.Smth;%1;
Lpass=params.Lpass; %1;
Zscore=params.Zscore; %1;
Perc=params.Perc;
fs=params.fs;
output_dF=all_dF;
% clean dF first 0.25 minute
output_dF(1:floor(fs*60*0.25))=0;
%   end
%% z-scored
if Zscore
    % see
    % https://www.sciencedirect.com/science/article/pii/S0091305720307413?via%3Dihub
   % all_dF = (all_dF - nanmedian(all_dF))./mad(all_dF); % normalization using robust z-score
  output_dF = (output_dF - nanmedian(baseline2))./mad(baseline2); % normalization using robust z-score
    % this one created figure 1 in SCN-VIP rep:
    %  all_dF = (all_dF - nanmedian(all_dF))./mad(baseline2); % normalization using robust z-score
    
    
    
    
    % test 030922
    %  all_dF = (all_dF - nanmedian(nanmedian(all_dF)))./mad(mad(baseline2)); % normalization using robust z-score
    %all_dF = (all_dF - nanmedian(nanmedian(baseline2)))./mad(mad(baseline2)); % normalization using robust z-score
    % testing this 030922- I don't understand why it give opposite trend in
    % median dF
    
    
end
%% smooth
if Smth
    tmp = smooth(output_dF,3*double(fs)); % smoothing
    output_dF=tmp';
end
%% Lowpass
if Lpass
    %all_dF.data = zeros(size(all_dF.Data));
    if ~isnan(output_dF)
    output_dF = lowpass(output_dF,LFcut,double(fs),order); % lowpass filter, see below
    end
end
if Perc
    this_Y=prctile(output_dF,8);
    output_dF=output_dF-this_Y;
 end


end


function y = lowpass(x,cutoff,Fs,n)
% Butterworth Lowpass Filter (zero-phase distortion filter)
% This creates n-th order Butterworth lowpass filter and takes input
% signal and creates output, the filtered signal. 
%
% <Usage>
%
% y = lowpass(x,cutoff,Fs,n)
% 
% where x: the unfiltered, raw signal
%       cutoff: cut-off frequency
%       Fs: sampling rate
%       n: the order of Butterworth filter
%       y: the filtered signal
%
% <Example>
%
% y = lowpass(x,100,2000,4);
%
% Coded by Ryan Cho, Oct 21 2013

[b,a] = butter(n,cutoff/(Fs/2),'low');
y = filtfilt(b,a,double(x));
end



  