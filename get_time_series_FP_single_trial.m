function [dF,t,analysis]=get_time_series_FP_single_trial(mouse_info, trial_info,analysis_params);
% get time_series data and calculate event and dF 
% read by ' get_time_series_FP_per_mouse'
 % Jan 2022: updated anaysis: 
 % 1) fft
 % 2) df and event analysis skip the first 30 seconds due to artifacts

clear y data
if nargin==0
   %mouse_info.ID='198R'; mouse_info.side='R';trial_info.rig='SynTDT';
   % mouse_info.ID='200LL'; mouse_info.side='R';
% mouse_info.ID='246RL'; mouse_info.side='R';trial_info.rig='SynTDT';
% mouse_info.ID='259R'; mouse_info.side='R';trial_info.rig='SynTDT';
   % mouse_info.ID='260L'; mouse_info.side='L';
%   mouse_info.ID='261RL'; mouse_info.side='R';trial_info.rig='SynTDT';
 %   mouse_info.ID='262R'; mouse_info.side='R'; trial_info.rig='SynTDT';
 %    mouse_info.ID='273RL'; mouse_info.side='R'; trial_info.rig='TDT';
    %   mouse_info.ID='288RL'; mouse_info.side='R';trial_info.rig='SynTDT';% male
     %   mouse_info.ID='313RL'; mouse_info.side='R';trial_info.rig='SynTDT';%female
   mouse_info.ID='366R'; mouse_info.side='L';trial_info.rig='SynTDT';
   
   
    %trial_info.date='102020';%MMDDYY
     trial_info.date='030723';%MMDDYY
    trial_info.sess_num=1;  trial_info.to_remove=[];
    trial_info.estrus=[];
   
    trial_info.show=1;
    trial_info.path='D:\DATA_Glab\fiberphotometry\';
    analysis_params.std_thresh=1.5;
    plot_fft=1;
else
    plot_fft=0;
    trial_info.show=0;
end


rig=trial_info.rig;


load([trial_info.path '\TDT_TimeSeries\VIPGC' mouse_info.ID '_' mouse_info.side 'fiber_' trial_info.date '_TimeSeriesSess' num2str(trial_info.sess_num) '.mat'])
%load('VIPGC198R_Rfiber_063020_TimeSeriesSess8.mat')
data=y;
% switch rig;     case 'TDT';  fs = data{1}.fs; % TDT FP rig; 
%                 case 'SynTDT'; fs = data{1}.fs; %Syn TDT FP rig 
% end
fs = data{1}.fs;

% parameters for basic data analysis dF/F
params.Smth=1;%1;
params.Lpass=1; %1;
params.Zscore=1; %1;
params.Perc=1;
params.fs=fs;


% recording starts at 9am. dark starts at 13:00 the 6th time is complete
% dark
%dark_sessions=[6:17]; 
nS=9;
nE=15;
dark_sessions=[nS:nE]; 
if strcmp(trial_info.date,'071720'); dark_sessions=[nS:9];end % recording stopped in the middle
if strcmp(trial_info.date,'070620')&&strcmp(mouse_info.ID,'198R'); dark_sessions=[nS:7];end %  fiber was out
if strcmp(trial_info.date,'072920')&&strcmp(mouse_info.ID,'198R'); dark_sessions=[nS:11];end %  fiber was out
if strcmp(trial_info.date,'072920')&&strcmp(mouse_info.ID,'200LL'); dark_sessions=[nS:10];end % fiber was out
if strcmp(trial_info.date,'091120')&&strcmp(mouse_info.ID,'246RL'); dark_sessions=[nS:14];end %  fiber was out
if strcmp(trial_info.date,'092020')&&strcmp(mouse_info.ID,'246RL'); dark_sessions=[nS:16];end %  photodiode went down
if strcmp(trial_info.date,'091120')&&strcmp(mouse_info.ID,'247RRL'); dark_sessions=[nS:14];end %  fiber was out
if strcmp(trial_info.date,'091320')&&strcmp(mouse_info.ID,'247RRL'); dark_sessions=[nS:11];end %  fiber was out
if strcmp(trial_info.date,'091420')&&strcmp(mouse_info.ID,'247RRL'); dark_sessions=[nS:13];end %  fiber was out
if strcmp(trial_info.date,'092520')&&strcmp(mouse_info.ID,'247RRL'); dark_sessions=[nS:9];end %  fiber was out
if strcmp(trial_info.date,'111020')&&strcmp(mouse_info.ID,'259R'); dark_sessions=[nS:11];end %  system crashed
if strcmp(trial_info.date,'111520')&&strcmp(mouse_info.ID,'259R'); dark_sessions=[nS:10];end %  system crashed
L=[];
for i=1:length(data)
    L=[L; length(data{i}.dF(1:end-1))]; 
end
minL=min(L);
% get data from file
dF=[];
t=[];
for i=1:length(data)
    dF=[dF; data{i}.dF(1:minL)];
    t=[t; data{i}.t(1:minL)];
end

% calculate baseline for this session 
% baseline1=median(dF(dark_sessions,:),1);
% baseline2=median(dF,1);
% baseline1=nanmedian(dF,1); this is wrong, as it is over all sessions. 

baseline2=nanmedian(dF(dark_sessions,:),1);
% test 030922
%baseline2=dF(dark_sessions,:);
B3 = rmmissing(baseline2);% remove nan
% baseline1=median(dF,1);
% baseline2=median(dF,1);

for hi=1:24 % always creats 24h array, even 
    if ~isempty(find(trial_info.to_remove==hi)) ||  size(dF,1)<hi % in case fiber was out
        dF(hi,:) = nan(1,size(dF,2)); t(hi,:)=nan(1,size(dF,2));
    else
        [dF(hi,:)]=get_df_from_raw_data_v5(dF(hi,:),B3,params);        
    end
end
% dF = (dF - mean(dF))./std(dF); % standard z-score
% 
dF_array=reshape(dF,1,size(dF,1)*size(dF,2));
dF_array=rmmissing(dF_array);% removes nan
analysis_params.peak_thresh=nanmean(dF_array)+analysis_params.std_thresh*std(dF_array);
% calculates event params
for hi=1:size(dF,1)
    if mean(isnan(dF(hi,:)))==0
        [allpks{hi},alllocs{hi},allw{hi},allp{hi}]=findpeaks(dF(hi,floor(30*fs):end),t(hi,floor(30*fs):end),'Annotate','extents','MinPeakProminence', analysis_params.peak_thresh);
        
%         figure
%         findpeaks(dF(hi,:),t(hi,:),'Annotate','extents','MinPeakProminence', analysis_params.peak_thresh);
%         ylim([-20 60] )
    else
        allpks{hi}=[]; alllocs{hi}=[];allw{hi}=[]; allp{hi}=[];
    end
end

for hi=1:size(dF,1)
    if ~isempty(allpks{hi})
        width(hi)=nanmedian(allw{hi});
        height(hi)=nanmedian(allp{hi});
        rate(hi)=length(allpks{hi})/(max(t(hi,:))/60);% event per minute
    else
        height(hi)=nan;
        width(hi)=nan;
        rate(hi)=0;
    end
end
% calculates df integral
for hi=1:size(dF,1)
    int_df(hi)=sum(dF(hi,floor(30*fs):end));
end
 
if  abs(size(dF,2)/600-fs)>0.1; disp('WRONG fs'); end 
 
if trial_info.show==1
    % plot dF
    figure
    for hi=1:size(dF,1)
        ph=plot(t(hi,:),dF(hi,:)+15*hi); hold on
        if ~isempty(find([1:4,17:24]==hi))
            ph.Color=[0.9290 0.6940 0.1250];
        else
            ph.Color=[0 0 0];
        end
    end
    ylim([-20 (hi+3)*15])
    xlabel('Time (sec)')
    ylabel('dF/F /hours')
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus])
    % plot rate/df
    figure
    plotyy([1:24], rate,[1:24], int_df); hold on 
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus])
  
end


%% fft analysis Jan 2022
Y = fft(dF(:,floor(30*fs):end)'); 
%Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
L1=size(Y,1);
P2 = abs(Y/L1);
P1 = P2(1:floor(L1/2)+1,:);
P1(2:end-1,:) = 2*P1(2:end-1,:);
%Define the frequency domain f and plot the single-sided amplitude spectrum P1. The amplitudes are not exactly at 0.7 and 1, as expected, because of the added noise. On average, longer signals produce better frequency approximations.
f = fs*(0:(L1/2))/L1;

GW=8;% gaussian window for smoothdata
B1 = smoothdata(P1,1,'gaussian',GW);

% find the mean frequencies of the obsereved peaks
%GW=[12 180 180];% the first window fits low freq, up to 0.03Hz. the second fits the 0.5 to 2 Hz
%f_limits=[0 0.03 ;0.5 1.1; 1.2 1.8];% define the interval for freq max identification to be avergaed
f_limits=[0.0033 0.03 ;0.03 0.5; 0.5 1.6];% define the interval for freq max identification to be avergaed. minimum is half of 10 minutes (longer can't be detected) 
%f_limits=[0 0.03 ;0.5 1.6];%HZ define the interval for freq max identification to be avergaed
%f_limits=[0 0.03 ;0.03 0.2];%HZ define the interval for freq max identification to be avergaed
f_limits=[0.0033 0.007 ;0.007 0.05; 0.05 0.1; 0.1 0.25; 0.25 0.45; 0.45 1; 1.0 1.35];

for wi=1:size(f_limits,1)
    for di=1:size(B1,2)
        
        f_inds=intersect(find(f>f_limits(wi,1)),find(f<f_limits(wi,2)));
        if ~isempty(f_inds)
            B1_int(wi,di)=sum(B1(f_inds,di));
        else
            B1_int(wi,di)=nan;
        end
    end
end


% define colors
if ~isempty(trial_info.estrus); estrus_array{1}=trial_info.estrus; [ALL_colors,color_ind]=get_estrus_colors(estrus_array); 
else; ALL_colors=[0 0 0; 0.4 0.4 0.4]; color_ind=1;end

if plot_fft
    figure;
    subplot(2,2,1)
    for di=[1:4,17:size(B1,2)]
        % [pks,locs] = findpeaks(B1(:,di)) ;
        ph=plot(f,B1(:,di)) ;ph.Color=[0.9290 0.6940 0.1250]; hold on;
        % plot(f(TF_all(:,di)),B1(TF_all(:,di),di)','r*'); hold on
    end
    for di=[5:16]
        % [pks,locs] = findpeaks(B1(:,di)) ;
        plot(f,B1(:,di),'k') ; hold on;
        
    end
    
    xlim([0 2.5])
    % ylim([0 10])
    xlabel('freq (Hz)'); ylabel('|P1(f)|')
    
    figure
    %subplot(2,2,2)
    for di=[1:4,17:size(B1,2)]
        ph=plot(log10(f),log10(B1(:,di))) ;ph.Color=[0.9290 0.6940 0.1250]; hold on;
    end
    for di=[5:16]
        plot(log10(f),log10(B1(:,di)),'k') ; hold on;
    end 
    f_limits(f_limits==0)=0.0001;
    if size(f_limits,1)==3
        line(log10(f_limits)',[0 0; -1 -1; -2 -2 ]')
    elseif size(f_limits,1)==5
        line(log10(f_limits)',[0 0; -0.5 -0.5; -1 -1; -1.5 -1.5; -2 -2 ]')
    elseif size(f_limits,1)==7
        line(log10(f_limits)',[0 0; -0.25 -0.25; -0.5 -0.5; -0.75 -0.75; -1 -1; -1.5 -1.5; -2 -2 ]')

    end
    xlim([-2.75 1.5])
    % ylim([0 10])
    xlabel('log freq (Hz)'); ylabel('log |P1(f)|')
    
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus ' GW=' num2str(GW)]);
    print(['FFT_' mouse_info.ID '_sess' num2str(trial_info.sess_num) '_Power_vs_Freq'],'-depsc')

    %subplot(2,2,[3 4])
     figure
    XMAX=1.2;
    if size(f_limits,1)>2
        %figure
        for wi=1:size(f_limits,1)
            %   subplot(size(f_limits,1),1,wi)
            ph=plot(B1_int(wi,:)/max(B1_int(wi,:))); hold on;
            %             switch wi
            %                 case 1; ph.Color=ALL_colors(color_ind,:);
            %                 case 2; ph.Color=ALL_colors(color_ind+1,:);
            %             end
            ph.Color=[0.08*wi 0.08*wi 0.08*wi];
        end
        legend(num2str(f_limits))
         ylim([0 XMAX]); ylabel('Normalized integrated Power (a.u.)'); 
    else
        ph1=plot([1:24],B1_int(1,:)/max(B1_int(1,:)));hold on;
        ph1.Color=ALL_colors(color_ind,:);
        ylim([0 XMAX]); ylabel('Normalized integrated Power low freq (a.u.)');
        yyaxis right
        ph2=plot([1:24],B1_int(2,:)/max(B1_int(2,:)));hold on;
        ph2.Color=ALL_colors(color_ind,:)*0.7;
        ylim([0 XMAX]); ylabel('Normalized integrated Power high freq (a.u.)'); 
    end
    xlabel('Time (hours)')
    ph=line([5 5],[0 XMAX]); ph.Color='k';
    ph=line([17 17],[0 XMAX]); ph.Color='k';
    xlim([1 24]);
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus ' GW=' num2str(GW)]);
    annotation('textbox', [0.01, 0.5, 0, 0], 'string', [ { 'freq' 'range:'}'; num2str(f_limits')]) 
       print(['FFT_' mouse_info.ID '_sess' num2str(trial_info.sess_num) '_Power_vs_time'],'-depsc')
end
 % done fft 
 
analysis.med_width=width;
analysis.med_amplitude=height;
analysis.rate=rate;
analysis.int_df=int_df;
analysis.int_fft=B1_int;
analysis.int_fft_limits=f_limits;
analysis.fft_power=B1;
analysis.freq=f;

