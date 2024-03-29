function [y] = SynFP_get_dF_time_series(filename,CH,isTTL)
%function [y] = SynFP_get_dF_until062019(filename,STREAM_STORE1,STREAM_STORE2)
%% Fiber Photometry 
%
%% based on FP_Epoch_Averaging_example
% % https://www.tdt.com/support/EXEpocAveragingExampleDR.html
% <html>
% This example goes through fiber photometry analysis using techniques <br>
% such as data smoothing, bleach detrending, and z-score analysis. <br>
% The epoch averaging was done using TDTfilter. <br><br>
% Author Contributions: <br>
% TDT, David Root, and the Morales Lab contributed to the writing and/or conceptualization of the code. <br>
% The signal processing pipeline was inspired by the workflow developed by <a href="https://doi.org/10.1016/j.celrep.2017.10.066">David Barker et al. (2017)</a> for the Morales Lab. <br>
% The data used in the example were provided by David Root. <br><br>
% Author Information: <br>
% David H. Root <br>
% Assistant Professor <br>
% Department of Psychology & Neuroscience <br>
% University of Colorado, Boulder <br>
% Lab Website: <a href="https://www.root-lab.org">https://www.root-lab.org</a> <br>
% david.root@colorado.edu <br><br>
% About the authors: <br>
% The Root lab and Morales lab investigate the neurobiology of reward, aversion, addiction, and depression. <br>
% <br> TDT edits all user submissions in coordination with the contributing
% author(s) prior to publishing.
% </html>
%% AK Jan 2019 - I cut the function and left the very basic df/f data


%% Housekeeping
% Clear workspace and close existing figures. Add SDK directories to Matlab
% path.
close all;  clc;
%[MAINEXAMPLEPATH,name,ext] = fileparts(cd); % \TDTMatlabSDK\Examples
%DATAPATH = fullfile(MAINEXAMPLEPATH, 'ExampleData'); % \TDTMatlabSDK\Examples\ExampleData
%[SDKPATH,name,ext] = fileparts(MAINEXAMPLEPATH); % \TDTMatlabSDK

%DATAPATH='C:\TDT\Synapse\Tanks\';

DATAPATH=[];
STREAM_STORE1=['Ct' CH];
STREAM_STORE2=['GC' CH];
%addpath(genpath(SDKPATH));

%% Importing the Data
% This example assumes you downloaded our example data sets
% (<http://www.tdt.com/support/examples/TDTExampleData.zip link>) and extracted
% it into the \TDTMatlabSDK\Examples\ directory. To import your own data, replace
% |BLOCKPATH| with the path to your own data block.
%
% In Synapse, you can find the block path in the database. Go to Menu > History. 
% Find your block, then Right-Click > Copy path to clipboard.
%BLOCKPATH = fullfile(DATAPATH,'FiPho-180416');
BLOCKPATH = fullfile(DATAPATH,filename);%'VIPGC_25L-190301-150631'

%% Setup the variables for the data you want to extract
% We will extract two different stream stores surrounding the 'PtAB' epoch 
% event. We are interested in a specific event code for the shock onset.
zscore=0;
Lpass=1;
Smt=1;

LFcut = 2; % cut-off frequency of lowpass filter
order = 4; % N-th order for butterworth filter
POWER=0;
FIG=1;

%REF_EPOC = 'PtAB'; % event store name. This holds behavioral codes that are
% read through Ports A & B on the front of the RZ
%SHOCK_CODE = 64959; % shock onset event code we are interested in
% get as input
%STREAM_STORE1 = 'Ct2r'; % name of the 405 store
%STREAM_STORE2 = 'GC2r' ; % name of the 565/490 store

%TRANGE = [-10 20]; % window size [start time relative to epoc onset, window duration]
%BASELINE_PER = [-10 -6]; % baseline period within our window
%ARTIFACT = Inf; % optionally set an artifact rejection level

% Now read the specified data from our block into a Matlab structure.
%data = TDTbin2matAK(BLOCKPATH, 'TYPE', {'epocs', 'scalars', 'streams'});
%data = TDTbin2mat(BLOCKPATH, 'TYPE', {'snips','epocs', 'scalars', 'streams'});
data = TDTbin2mat(BLOCKPATH, 'COMBINE', {STREAM_STORE1,STREAM_STORE2});
%fs=data.streams.(STREAM_STORE1).fs; % the fs of the new FP 
time_series_data=data.snips.(STREAM_STORE2).data ; 
time_series_ctrl=data.snips.(STREAM_STORE1).data;
for ti=1:length(time_series_data)
    this_data=[time_series_data{ti} ;time_series_ctrl{ti}];
    [y{ti}] = fit_ref(this_data);
    y{ti}.data=this_data;
end

% %isTTL=0;
% switch isTTL
%     case 0
%         y.TTL=[];
%         disp('TTL is set to be empty')
%         disp('TTL is set to be empty')
%         disp('TTL is set to be empty')
%         disp('TTL is set to be empty')
%     case 1
%         y.TTL=data.epocs.PC0_.onset;
%         disp('TTL will be saved')
% end

fs=data.snips.(STREAM_STORE1).fs;
for ti=1:length(time_series_data)
    y{ti}.t=[data.snips.(STREAM_STORE1).ts:(1/fs):length(data.snips.(STREAM_STORE1).data{ti})/data.snips.(STREAM_STORE1).fs];
    y{ti}.fs=fs;
    all_dF{ti}.dF=y{ti}.dF;
end


for ti=1:length(time_series_data)
    if zscore
        %% z-scored
        all_dF{ti}.dF = (all_dF{ti}.dF - median(all_dF{ti}.dF))./mad(all_dF{ti}.dF,1); % normalization using robust z-score
        %  all_dF = (all_dF - mean(all_dF))./std(all_dF); % standard z-score
    end
    if Smt
        clear tmp
        tmp = smooth(all_dF{ti}.dF,3*fs); % smoothing
        all_dF{ti}.dF=tmp';
    end
    if Lpass
        %% Lowpass
        %all_dF.data = zeros(size(all_dF.Data));
        all_dF{ti}.dF = lowpass(double(all_dF{ti}.dF),LFcut,fs,order); % lowpass filter, see below
    end
    
    %% perctile
    Y = prctile(all_dF{ti}.dF,2);
    all_dF{ti}.dF=all_dF{ti}.dF-Y;
    clear Y
end
%t = 0:1/fs:length(all_dF.dF)/fs-1/fs;
if FIG
    figure
    for ti=1:length(time_series_data)
       % subplot (length(time_series_data),1,ti)
        plot(y{ti}.t,data.snips.(STREAM_STORE1).data{ti}(1:end-1)+ti*10)
        hold on
        plot(y{ti}.t,data.snips.(STREAM_STORE2).data{ti}(1:end-1)+ti*10)
        hold on
        ylabel('raw data','FontSize',12);
        legend([STREAM_STORE1;STREAM_STORE2])
    end
    
    figure
    for ti=1:length(time_series_data)
      %  subplot(length(time_series_data),1,ti)
        plot(y{ti}.t,all_dF{ti}.dF(1:end-1)+20*ti)
        hold on
    end
    ylabel('dF (Z-score)','FontSize',12);
end
end
%% fit function, to remove ref from signal
function [y] = fit_ref(data)
    B = data(1,:)'; 
    A = [data(2,:)' ones(length(data),1)];
    theta = A\B;
    y.fit400 = theta(1)*data(2,:)+theta(2);
    y.dF = 100*detrend((data(1,:)-y.fit400)./y.fit400);
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
y = filtfilt(b,a,x);
end


