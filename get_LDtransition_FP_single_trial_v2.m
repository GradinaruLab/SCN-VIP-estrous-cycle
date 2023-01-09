function [dF,t,analysis]=get_LDtransition_FP_single_trial_v2(mouse_info, trial_info,analysis_params);
% get time_series data and calculate event and dF 
% read by ' get_time_series_FP_per_mouse'
 % Jan 2022: updated anaysis: 
 % 1) fft
 % 2) df and event analysis skip the first 30 seconds due to artifacts

clear y data
if nargin==0
     mouse_info.ID='103R'; mouse_info.side='R';trial_info.rig='TDT';
   % mouse_info.ID='166R'; mouse_info.side='R';trial_info.rig='TDT';
    %mouse_info.ID='198L'; mouse_info.side='R';trial_info.rig='SynTDT';
    %mouse_info.ID='200RR'; mouse_info.side='R';trial_info.rig='SynTDT';
    
    
    trial_info.date='050219';%MMDDYY
    trial_info.sess_num=1;   trial_info.estrus=[];
    trial_info.onset_sess_num=[];
    %trial_info.to_remove=[];
    trial_info.show=1;
    trial_info.path='D:\DATA_Glab\fiberphotometry\';
    analysis_params.std_thresh=1;
    analysis_params.F=1;
    analysis_params.n_windows=15;
    plot_fft=1;
    plot1=1;
else
    plot1=0;
    plot_fft=0;
    trial_info.show=0;
end
do_ACF=0;% autocorr

rig=trial_info.rig;
n_minutes=60;
trial_info.length=2*(n_minutes*60)/60; % time in minutes 

files1=['VIPGC' mouse_info.ID '_' mouse_info.side 'fiber_' trial_info.date '_Sess' num2str(trial_info.sess_num)];
% get the ZT11-ZT12 recording 
load([trial_info.path '\' trial_info.rig '_FP\'  files1 '.mat'])
data=y;
clear y
% get the ZT10 recording 

if ~isempty(trial_info.onset_sess_num)
    files2=['VIPGC' mouse_info.ID '_' mouse_info.side 'fiber_' trial_info.date '_SessOnset' num2str(trial_info.onset_sess_num)];
    load([trial_info.path '\' trial_info.rig '_FP\'  files2 '.mat'])
    data2=y;
    is_onset=1;
    clear y
else
    data2=[];
    is_onset=0;
end
%load('VIPGC198R_Rfiber_063020_TimeSeriesSess8.mat')
%%%%%%%%%%%%%%%%stopped here 06062022

if isfield(data,'fs')
    fs = double(data.fs);
else
    fs=382;% TDT
end

% parameters for basic data analysis dF/F
params.Smth=1;%1;
params.Lpass=1; %1;
params.Zscore=1; %1;
params.Perc=1;
params.fs=fs;
t1=fs*10; %skipping the first 10 seconds, the freq and intensity set
%% pick up just the +- 60 minutes from light off
[light_array]=finds_light_status(files1,data,['Sess' num2str(trial_info.sess_num)],t1,fs);
TRANGE=[-n_minutes*60 n_minutes*60]; % n_minutes before and after dark; in seconds ; 
time_epoc=light_array.light_off;
%data.dF=data.dF(intersect(find(data.t>time_epoc+TRANGE(1)),find(data.t<(time_epoc+TRANGE(2)))));
data.data=data.data(:,intersect(find(data.t>time_epoc+TRANGE(1)),find(data.t<(time_epoc+TRANGE(2)))));
% also put t start to ~zero
data.t=data.t(intersect(find(data.t>time_epoc+TRANGE(1)),find(data.t<(time_epoc+TRANGE(2)))))-time_epoc-TRANGE(1);

% add onset data, if exists
if is_onset
    inds=intersect(find(data2.t<=60*30),find(data.t>0*60));
    data.data=[data2.data(:,inds) data.data];
    data.t=[data2.t(inds)-60*60 data.t];  
   % data.data=data.data(:,find(data.t>0.5*60));
    % data.t=data.t(find(data.t>0.5*60));
end
% correct the raw data for both onset and sess, or just sess
 [data.dF] = fit_ref(data);
% plot(data.t,dF)
%  calculate dF/F 
baseline2=data.dF(4*end/5:end);
B3 = rmmissing(baseline2);% remove nan

[dF_F]=get_df_from_raw_data_v6(data.dF,B3,params);    

 % check how the data looks like
 if plot1
     figure
     plot(data.t,data.dF); hold on % before data processing 
     plot(data.t,dF_F); hold on % after 
     ylim([-20 35])
 end
% recording starts at ZT10 for 2 hours 
% devide data to intervals of X minutes. Data is 150 or 120 minutes total 
% ZT11 to 12 start times % adds the ZT10 start times 
switch analysis_params.n_windows
    case 3
        bin_size=n_minutes; % minutes
        t_start=[-60:bin_size:trial_info.length-bin_size+1]*60; % in seconds
        t_end=[-30 60:bin_size:trial_info.length]*60; % in seconds
    case 15
        bin_size=10; % minutes
        t_start=[-60:bin_size:trial_info.length-bin_size+1]*60; % in seconds
        t_end=[-50:bin_size:trial_info.length]*60; % in seconds
end

for hi=1:length(t_start)
    inds=intersect(find(data.t>t_start(hi)),find(data.t<=t_end(hi)));
    if  isempty(inds) % will skip -1800 to 0, and also sessions that there are no onsetSess
        divided_data{hi}.dF=[];
        divided_data{hi}.t=[];
    else
        if inds(1)==1
            divided_data{hi}.dF=dF_F(inds);
            divided_data{hi}.t=data.t(inds);
        else
            divided_data{hi}.dF=dF_F(inds-1);
            divided_data{hi}.t=data.t(inds-1);
        end
    end
end
clear new_divided_data
new_divided_data=divided_data;

% finds the largest length that fits all / might not be needed for the DL transition 
L=[];T=[];
for i=1:length(new_divided_data)
    L=[L; length(new_divided_data{i}.dF)]; 
   % T=[T; max(new_divided_data{i}.t)]; 
end
maxL=max(L(L~=0));
% adds nan to arrays that are not long enough at the end
for i=1:length(new_divided_data)
    if length(new_divided_data{i}.dF)<maxL
        new_divided_data{i}.dF=[new_divided_data{i}.dF nan(1,maxL-length(new_divided_data{i}.dF))];
        new_divided_data{i}.t=[new_divided_data{i}.t nan(1,maxL-length(new_divided_data{i}.t))];
    end
    T=[T; max(new_divided_data{i}.t)];
end

% interpolate the data to a constant fs, new_fs (TDT and SynTDT has different
% sampling rate)

% get data from file% interpolate the data to a constant fs
clear t_intrlp dF_intrlp
for i=1:length(new_divided_data)
    if ~isempty(new_divided_data{i}.dF) && ~isempty(find(~isnan(new_divided_data{i}.dF))) % empty or nan
        new_fs=382;% Hz, per sect
        if abs(new_fs-fs)<0.001
            dF_intrlp{i}=new_divided_data{i}.dF;
            t_intrlp{i}=new_divided_data{i}.t;
        else % if not the same fs, need to interpolate the data
            t_intrlp{i}=t_start(i):(1/new_fs):t_end(i) ;
            this_df=new_divided_data{i}.dF;
            TTT=new_divided_data{i}.t(~isnan(this_df)); % just to check it
            dF_intrlp{i}=interp1(new_divided_data{i}.t(~isnan(this_df)),this_df(~isnan(this_df)),t_intrlp{i});
        end
    end
end

% put the new interpolated data into a new array, dF_raw and t_raw
dF_raw=[];t_raw=[]; 
MAXL=max(cellfun(@length,dF_intrlp));
for i=1:length(dF_intrlp)
    if length(dF_intrlp{i})<MAXL
        dF_raw=[dF_raw; [dF_intrlp{i} nan(1,MAXL-length(dF_intrlp{i}))]];
        t_raw=[t_raw; [t_intrlp{i} nan(1,MAXL-length(t_intrlp{i}))]];
    else
        dF_raw=[dF_raw; dF_intrlp{i}];
        t_raw=[t_raw; t_intrlp{i}];
    end  
end

% check the interpolation: 
check=0;
if check
    figure
    for i=1:length(new_divided_data)
        plot( new_divided_data{i}.t,  new_divided_data{i}.dF);hold on
        plot( t_raw(i,:),  dF_raw(i,:)+10);hold on
    end
end
dF=dF_raw;
t=t_raw;

% calculated peak_thresh based on the Z11-12 session  
switch analysis_params.n_windows
    case 3
        if length(dF)>2; inds=2; else; inds=1; end
    case 15
        inds=[13:18];
end
analysis_params.peak_thresh=analysis_params.F*nanmedian(nanmedian(dF(inds,:)))+analysis_params.std_thresh*nanstd(nanstd(dF(inds,:)));

% calculates event params
plot2=1;
if plot2
    figure
end
for hi=1:size(dF,1)
   
    if sum(isnan(dF(hi,:)))<length(dF(hi,:))
        
        [allpks{hi},alllocs{hi},allw{hi},allp{hi}]=findpeaks(dF(hi,~isnan(dF(hi,:))),t(hi,~isnan(dF(hi,:))),'Annotate','extents','MinPeakProminence', analysis_params.peak_thresh);
        %[allpks{hi},alllocs{hi},allw{hi},allp{hi}]=findpeaks(dF(hi,floor(5*fs):end),t(hi,floor(5*fs):end),'Annotate','extents','MinPeakProminence', analysis_params.peak_thresh);
        % DO NOT delete this. use to check if needed
        if plot2
            subplot(1,ceil(size(dF,1)),hi)
            findpeaks(dF(hi,~isnan(dF(hi,:))),t(hi,~isnan(dF(hi,:))),'Annotate','extents','MinPeakProminence', analysis_params.peak_thresh);hold on
            ylim([-20 60] )
            title([mouse_info.ID ' sess ' num2str(trial_info.sess_num) ' ' trial_info.date])
        end
    else
        allpks{hi}=[]; alllocs{hi}=[];allw{hi}=[]; allp{hi}=[];
    end
end



for hi=1:size(dF,1)
    if ~isempty(allpks{hi})
        width(hi)=nanmedian(allw{hi});
        height(hi)=nanmedian(allp{hi});
        rate(hi)=length(allpks{hi})/(abs(max(t(hi,:))-min(t(hi,:)))/60);% event per minute, t periods are 600 minutes- 10 minutes*60 sec
    else
        height(hi)=nan;
        width(hi)=nan;
        rate(hi)=nan;
    end
end
% calculates df integral
for hi=1:size(dF,1)
    int_df(hi)=sum(dF(hi,floor(5*new_fs):end)); % skip the first 10 samples
end
 
if  abs(size(dF,2)/(bin_size*60)-new_fs)>new_fs*0.1; disp([mouse_info.ID ' sess ' num2str(trial_info.sess_num) ': WRONG fs']); end 
disp([ 'dF size = ' num2str(size(dF,2))]) 

A_length=size(dF,1);
if trial_info.show==1
    % plot dF
    figure
    for hi=1:size(dF,1)
        ph=plot(t(hi,:),dF(hi,:)); hold on
        if ~isempty(find([1:6]==hi))
            ph.Color=[0.9290 0.6940 0.1250];
        else
            ph.Color=[0 0 0];
        end
    end
    ylim([-10 20])
    xlabel('Time (sec)')
    ylabel('dF/F /hours')
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus])
    % plot rate/df
    figure
    plotyy([1:A_length], rate,[1:A_length], int_df); hold on 
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus])
  
end


%% fft analysis Jan 2022
% skip the first 30 points 
clear L1 P2 P1 f B1
dF(isnan(dF))=0;
Y = fft(dF(:,floor(30*new_fs):end)'); 
%Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
L1=size(Y,1);
P2 = abs(Y/L1);
P1 = P2(1:floor(L1/2)+1,:);
P1(2:end-1,:) = 2*P1(2:end-1,:);
%Define the frequency domain f and plot the single-sided amplitude spectrum P1. The amplitudes are not exactly at 0.7 and 1, as expected, because of the added noise. On average, longer signals produce better frequency approximations.
f = new_fs*(0:(L1/2))/L1;

GW=8;% gaussian window for smoothdata
B1 = smoothdata(P1,1,'gaussian',GW);

% find the mean frequencies of the obsereved peaks
%GW=[12 180 180];% the first window fits low freq, up to 0.03Hz. the second fits the 0.5 to 2 Hz
%f_limits=[0 0.03 ;0.5 1.1; 1.2 1.8];% define the interval for freq max identification to be avergaed
f_limits=[0.0005 0.03 ;0.03 0.1; 0.1 0.35];% define the interval for freq max identification to be avergaed
%f_limits=[0 0.03 ;0.5 1.6];%HZ define the interval for freq max identification to be avergaed
%f_limits=[0 0.03 ;0.03 0.2];%HZ define the interval for freq max identification to be avergaed


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
    if size(B1,2)>11; n_color=8; else n_color=3; end
    figure;
    subplot(2,2,1)
    for di=[1:n_color]
        % [pks,locs] = findpeaks(B1(:,di)) ;
        ph=plot(f,B1(:,di)) ;ph.Color=[0.9290 0.6940 0.1250]; hold on;
        % plot(f(TF_all(:,di)),B1(TF_all(:,di),di)','r*'); hold on
    end
    for di=[n_color+1:size(B1,2)]
        % [pks,locs] = findpeaks(B1(:,di)) ;
        plot(f,B1(:,di),'k') ; hold on;
        
    end
    
    xlim([0 2.5])
    % ylim([0 10])
    xlabel('freq (Hz)'); ylabel('|P1(f)|')
    
    %figure
    subplot(2,2,2)
    for di=[1:n_color]
        ph=plot(log10(f),log10(B1(:,di))) ;ph.Color=[0.9290 0.6940 0.1250]; hold on;
    end
    for di=[n_color+1:size(B1,2)]
        plot(log10(f),log10(B1(:,di)),'k') ; hold on;
    end 
    f_limits(f_limits==0)=0.0001;
    line(log10(f_limits)',[0 0; -1 -1; -2 -2]')
    xlim([-2.75 1.5])
    % ylim([0 10])
    xlabel('log freq (Hz)'); ylabel('log |P1(f)|')
    
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus ' GW=' num2str(GW)]);
    print(['FFT_' mouse_info.ID '_sess' num2str(trial_info.sess_num) '_Power_vs_Freq'],'-depsc')

    subplot(2,2,[3 4])
    % figure
    XMAX=50;
    if size(f_limits,1)>2
        %figure
        for wi=1:size(f_limits,1)
            %   subplot(size(f_limits,1),1,wi)
            ph=plot(B1_int(wi,:)); hold on;
            switch wi
                case 1; ph.Color=ALL_colors(color_ind,:);
                case 2; ph.Color=ALL_colors(color_ind+1,:);
            end
        end
        ylim([0 XMAX]); ylabel('int P (a.u.)'); 
    else
        ph1=plot([1:A_length],B1_int(1,:));hold on;
        ph1.Color=ALL_colors(color_ind,:);
        ylim([0 XMAX]); ylabel('int P low freq (a.u.)');
        yyaxis right
        ph2=plot([1:A_length],B1_int(2,:));hold on;
        ph2.Color=ALL_colors(color_ind,:)*0.7;
        ylim([0 XMAX]); ylabel('int P high freq (a.u.)'); 
    end
    xlabel('Time (hours)')
    ph=line([5 5],[0 XMAX]); ph.Color='k';
    ph=line([17 17],[0 XMAX]); ph.Color='k';
    xlim([1 A_length]);
    title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus ' GW=' num2str(GW)]);
    annotation('textbox', [0.01, 0.5, 0, 0], 'string', [ { 'freq' 'range:'}'; num2str(f_limits')]) 
       print(['FFT_' mouse_info.ID '_sess' num2str(trial_info.sess_num) '_Power_vs_time'],'-depsc')
end
 % done fft 
 
 %% autocorrelation - to find dominant frequencies
 if do_ACF
     for hi=1:A_length
         t2=t(hi,30*fs:end);
         y=double(dF(hi,30*fs:end)');
         y2=medfilt1(y,100);
         if ~(sum(isnan(y2))==length(y2))
             figure
             title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus ' hour=' num2str(hi)]);
             subplot(2,1,1)
             plot(t2,y,t2,y2); hold on
             subplot(2,1,2)
             
             % autocorr(y2,'NumLags',40,'NumSTD',2)
             parcorr(y2,'NumLags',10,'NumSTD',2)
         end
     end
     %02/15/22 produce mainly h=0,1,2 above threshold- does it make sense?
 end
 % done with autocorr
%%
analysis.new_fs=new_fs;
analysis.med_width=width;
analysis.med_amplitude=height;
analysis.rate=rate;
analysis.int_df=int_df;
analysis.int_fft=B1_int;
analysis.int_fft_limits=f_limits;
analysis.fft_power=B1;
analysis.freq=f;
end

%% fit function, to remove ref from signal
function [dF] = fit_ref(input)
    B = input.data(1,:)';
    A = [input.data(2,:)' ones(length(input.data),1)];
    theta = A\B;
    fit400 = theta(1)*input.data(2,:)+theta(2);
    dF = 100*detrend((input.data(1,:)-fit400)./fit400);
end
