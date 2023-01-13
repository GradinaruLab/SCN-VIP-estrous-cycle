function [output] = get_compass_data_v3(exp)
%% read csv files produced with COMPASS ('processing' )
% measurment is done every 10 seconds
% plots individual days and mean 
clear mouse_ind mouseID
bar_plot=0;
rate=6; % number of measuremnts per minute
if nargin==0
    %   exp='VIP_C57_comparison July2021'
   exp='VIP_ChR2_DD'
   % exp='VIP_ChR2_DD_2'
    %exp='Esr1_GFAP_Esr1_KD_1';
    %exp='VIP_cre_DL'
    %exp='VIP_cre_DD'
     %    exp='VIP_cre_DD_red_light'
    %     exp='VIP_cre_LD 102021'
    %     exp='VIP_cre_DD_30min_blue'
     
    %exp='VIP_cre_DD_completeDD_Blue'
    %exp='VIP_cre_DD_30min@8pm'
    %exp='VIP_cre_DD_30min@8pm_60min@530am'
    %exp='VIP_cre_DD_completeDD'
    %exp='VIP_cre_DD_30min@8pm2'
    % exp='VIP_cre_DD_30min@8pm_60min@12am';
end
mean_Fig=0;

switch exp
    case 'VIP_C57_comparison July2021'
        filename='210714_0920';
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        exp_title='VIP C57 DD+L1+L2';
        off_limit=4300;
        light_off=3; %8am
    case 'VIP_ChR2_DD' % 4R 5R 6L
        filename='210126_1000'; % DD+light at 5:30-6am+447at9:30
        filename='210202_0940'; % DD+light at 5:30-6am+447at9:30
        %filename='210222_1727'; % DD+light at 5:30-6am+447at15:30
       % filename='210304_1531'; %DD+light at 5:30-6am+447at15:30
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        mouse_ind(3)=3; existEEG(3)=0; mouseID{3}='female';
        exp_title='VIP ChR2 DD+L1+O2';
        off_limit=4300;
        light_off=5.5; %5.5am
    case 'VIP_ChR2_DD_2'
        % 30R, (31L/32R),35R
        filename='210322_0938'; % VIP-cre DL 12/12 3pm to 3am dark.
         filename='210426_0701'; % VIP-cre DD+light at 3:00-3:30am+EXCI@ZT4
          filename='210518_1224'; % VIP-cre DD+light at 3:00-3:30am+EXCI@ZT10
            filename='210611_1257'; % VIP-cre DL DD+light at 3:00-3:30am+ZT10
        %          filename='210714_0920'; % VIP-cre DL 12/12 3pm to 3am dark.
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        mouse_ind(3)=3; existEEG(3)=0; mouseID{3}='female';
        exp_title='VIP ChR2 DD+L1+O2';
        off_limit=4300;
        light_off=3; %8am
      case 'VIP_cre_DD_30min@8pm2'
        filename='201113_0956'; % VIP-cre DL 12/12 8am to 8pm dark. 8-8:30pm light
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        exp_title='VIP cre DD 30min@8pm 2 ';
        off_limit=4300;
        light_off=8; %8am
      

       case 'VIP_cre_DD_30min@8pm_60min@530am'
        filename='200922_0915'; % VIP-cre DL 12/12 8am to 8pm dark. 8-8:30pm light
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        exp_title='VIP cre DD 30min/60min light at 8pm/530am';
        off_limit=4300;
        light_off=8; %8am
    case 'VIP_cre_DD_30min@8pm_60min@12am'
        filename='201120_1000'; % VIP-cre DL 12/12 8am to 8pm dark. 8-8:30pm light
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        exp_title='VIP cre DD 30min/60min light at 8pm/12am';
        off_limit=4300;
        light_off=8; %8am
    case 'VIP_cre_DD_30min@8pm'
        filename='200901_0945'; % VIP-cre DL 12/12 8am to 8pm dark. 8-8:30pm light
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        exp_title='VIP cre DD 30min light at 8pm';
        off_limit=4300;
        light_off=8; %8am
    case 'VIP_cre_DL'
        filename='200630_0915'; % VIP-cre DL 12/12 8am to 8pm dark
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        off_limit=4300;
         exp_title='VIP cre DL';
         light_off=8; %8am
    case {'VIP_cre_DD', 'VIP_cre_DD_red_light'}
        filename='200724_0930'; % VIP-cre DD
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        exp_title='VIP cre DD';
        off_limit=5700;
        light_off=8; %8am

     case {'VIP_cre_DD_completeDD'}
        filename='201015_0918'; % VIP-cre DD
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
        exp_title='VIP cre DD complete DD';
        off_limit=5700;
        light_off=8; %8am
     case { 'VIP_cre_LD 102021'}
        filename='211019_1603'; % VIP-cre DD
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female';
        mouse_ind(2)=3; existEEG(2)=0; mouseID{2}='female';
        %mouse_ind(3)=3; existEEG(3)=0; mouseID{3}='female';
        exp_title='VIP cre LD (blue light)';
        off_limit=5700;
        light_off=15; %8am
    case 'VIP_cre_DD_30min_blue'
        filename='211118_1703'; % VIP-cre DL 12/12 3am to 3pm light. 3-3:30pm light
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female_blue';
        mouse_ind(2)=3; existEEG(2)=0; mouseID{2}='female_blue';
        %mouse_ind(3)=3; existEEG(2)=0; mouseID{3}='female_blue';
        exp_title='VIP cre DD 30min blue';
        off_limit=4300;
        light_off=15; %3pm
     case 'VIP_cre_DD_completeDD_Blue'
        filename='211211_1803'; % VIP-cre DL 12/12 3am to 3pm light. 3-3:30pm light
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female_blue';
        mouse_ind(2)=3; existEEG(2)=0; mouseID{2}='female_blue';
        %mouse_ind(3)=3; existEEG(3)=0; mouseID{3}='female_blue';
        exp_title='VIP cre DD (complete, after blue)';
        off_limit=4300;
        light_off=15; %3pm
    case 'Esr1_GFAP_Esr1_KD'
        filename='200304_0908'; %  lower numbers with 1. kigher numbers are 2
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female_KO';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
         exp_title='Esr1 GFAP Esr1 KD 1';
         off_limit=4300; % check on that
         light_off=8; %8am
     case 'Esr1_GFAP_Esr1_KD_2'
        filename='200609_0933'; %  - Esr1 KO . KO female had skin issues
        mouse_ind(1)=1; existEEG(1)=0; mouseID{1}='female_KO';
        mouse_ind(2)=2; existEEG(2)=0; mouseID{2}='female';
         exp_title='Esr1 GFAP Esr1 KD 2';
         off_limit=4300;% check on that
         light_off=8; %8am
end
%

%mouse_ind(3)=3; existEEG(3)=0;  mouseID{3}='nan';
One_day_COMP=24*60*6;

%comp_path='D:\DATA_Glab\Behavioral\COMPASS\';
comp_path='C:\Users\anatk\Documents\Data_Glab_home_work\Behavioral\COMPASS\';
existEEG=0;
if existEEG
    load('D:\DATA_Glab\Behavioral\COMPASS\Behavioral_States.mat')
    Wake=Behavioral_States(1,:);
    NREM=Behavioral_States(2,:);
    REM=Behavioral_States(3,:);
    One_day_EEG=length(Wake);
end
%COMPASS_data=csvread([comp_path filename '.csv'],1,2);
T=readtable([comp_path filename '.csv']);
COMPASS_data=table2array(T(:,3:end));

for ti=1:6
    if strcmp(filename,'200901_0945') || strcmp(filename,'211019_1603')|| strcmp(filename,'211118_1703')%when reads, use different headers
        this_text=T.Time(ti);
    else
        this_text=T.Var1(ti);
    end
    n=strfind(this_text{1},'T');
    time_hour(ti)=str2num(this_text{1}(n+1:n+2));
    time_minutes(ti)=str2num(this_text{1}(n+4:n+5));
end
same_minutes=find(time_minutes==time_minutes(1));
% add nan arrays to make it start at 'light off'
%each minute there are 6 recording, so need to add nan arrays up to that
add_array_length=rate-length(same_minutes);% fill one minute
add_array_length=add_array_length+(60-time_minutes(1))*rate; % fill one hour
add_array_length=add_array_length+(time_hour(1)-1-light_off)*60*rate;
added_array=nan(add_array_length,size(COMPASS_data,2));
COMPASS_data_2=cat(1,added_array,COMPASS_data);
clear COMPASS_data

switch exp
    case 'VIP_cre_DD'
       %  COMPASS_data=COMPASS_data_2(1:end,:);% technitians turned on red light at the begining of the experiment. this data is removed

        COMPASS_data=COMPASS_data_2(One_day_COMP*7:One_day_COMP*(7+21),:);% technitians turned on red light at the begining of the experiment. this data is removed
    case 'VIP_cre_DD_red_light'
        %COMPASS_data=COMPASS_data_2(1:95016+add_array_length,:);% technitians turned on red light at the begining of the experiment. this data is removed      
        COMPASS_data=COMPASS_data_2(One_day_COMP*7:One_day_COMP*(7+21),:);% technitians turned on red light at the begining of the experiment. this data is removed      
       % COMPASS_data=COMPASS_data_2(One_day_COMP*1:end,:);% technitians turned on red light at the begining of the experiment. this data is removed      

        exp_title='VIP cre DD R light';
    otherwise
        COMPASS_data=COMPASS_data_2;
end
clear PIR

% get PIR data 
for di=1:floor(length(COMPASS_data)/One_day_COMP)
    ALL_PIR(:,:,di)=COMPASS_data((di-1)*One_day_COMP+1:One_day_COMP*di,1:6);
    ALL_PIR_nonan(:,:,di)=COMPASS_data((di-1)*One_day_COMP+1:One_day_COMP*di,1:6);
    ALL_LDR(:,:,di)=COMPASS_data((di-1)*One_day_COMP+1:One_day_COMP*di,7);
end
if length(COMPASS_data)/One_day_COMP-floor(length(COMPASS_data)/One_day_COMP)>0.3 % the last day is not a full day
    tmp_d=COMPASS_data((di+1-1)*One_day_COMP+1:end,1:6);
    nan_tmp=nan(One_day_COMP-size(tmp_d,1),6);
    tmp2=[tmp_d' nan_tmp'];
    ALL_PIR(:,:,di+1)=tmp2';
end

PIR=nanmean(ALL_PIR,3);
LDR=nanmean(ALL_LDR,3);
% get light status

LDR=LDR/max(LDR);


logLDR(find(LDR>0.5))=true(1);
logLDR(find(LDR<=0.5))=false(1);
%B=repmat(logLDR,50,1);
imwrite( repmat(logLDR,50,1), 'NameOfFile.tiff')
LDR_tiff=imread('NameOfFile.tiff');

if bar_plot
    % This figure shows daily rythms of all detectors
    for i=1:length(mouse_ind)
        figure ('name',[filename ' ' mouseID{i}])
        subplot (9,1,1);
        imshow(LDR_tiff);
        for di=1:size(ALL_PIR,3)
            subplot (size(ALL_PIR,3)+1,1,1+di);
            bar(ALL_PIR(:,mouse_ind(i),di),'FaceColor',[0.35,0,0.55]) ; hold on
            xlim([0 One_day_COMP])
            %legend({'motion sensor'},'fontsize',12)
            set(gca, 'Xticklabel',[])
            set(gca, 'Yticklabel',[])
            ylabel('PIR sensor','fontsize',15)
        end
        for di=1:size(ALL_PIR,3)
            if existEEG
                subplot (size(ALL_PIR,3)+7,1,7+di) %% check on that!!!
                bar(Wake,'FaceColor',[0.4,0,0.5]); hold on;
                bar(NREM,'FaceColor',[0.5,0.5,0.5]); hold on;
                bar(REM,'FaceColor',[0,0.7,0.9]); hold on;
                legend({'Wake','NREM','REM'},'fontsize',12)
                xlim([0 One_day_EEG])
                set(gca, 'Xtick',[0 One_day_EEG/4 One_day_EEG/2 One_day_EEG*3/4 One_day_EEG])
                set(gca, 'Xticklabel',{'0', '6', '12' ,'18' ,'24'},'fontsize',12)
                set(gca, 'Yticklabel',[])
                xlabel('Time (hours)','fontsize',15)
                ylabel('EEG/EMG','fontsize',15)
            end
        end
    end
end

% find onsets/offset, using envelop_hilbert function 
SamplingFrequency=360;
Smooth_window=SamplingFrequency*1;
DURATION=SamplingFrequency*1;
threshold_style=1;% 0 is manually detected
gr=0;% show figure



% circular plots and mean hour  
for i=1:length(mouse_ind)
    This_PIR(:,:)=ALL_PIR(:,mouse_ind(i),:);
    for di=1:size(This_PIR,2)
        data=This_PIR(:,di);
        [circ_params] = my_circle_plot(data,exp);% find the mean value per detector       
        Hour_mean(i,di)=circ_params.Hour_mean;
        activity_onsets(i,di)=circ_params.onset;
         activity_offsets(i,di)=circ_params.offset;
         activity_period(i,di)=circ_params.offset-circ_params.onset;
    end
end
Choose_cage=2;
figure
plot(Hour_mean(Choose_cage,:),[1:size(This_PIR,2)],'*-k'); hold on
plot(activity_onsets(Choose_cage,:),[1:size(This_PIR,2)],'*-b'); hold on
plot(activity_offsets(Choose_cage,:),[1:size(This_PIR,2)],'*-r'); hold on
plot(activity_period(Choose_cage,:),[1:size(This_PIR,2)],'*-g'); hold on
set(gca, 'YDir','reverse') 
xlim([0 24])
title(exp_title)

clear This_PIR
for i=1:length(mouse_ind)
    This_PIR(:,:)=ALL_PIR_nonan(:,mouse_ind(i),:);
    This_PIR1=reshape(This_PIR,size(This_PIR,1)*size(This_PIR,2),1);
    %alarm = envelop_hilbert(This_PIR1,Smooth_window,threshold_style,DURATION,gr);
    M=max(This_PIR1);
   % figure; bar(This_PIR1);hold on;  plot(alarm*M);
    %if length(alarm)>length(This_PIR1); alarm=alarm(1:length(This_PIR1));end
    %onsets(:,:)=reshape(alarm,size(This_PIR,1),size(This_PIR,2));
    %subplot(1,length(mouse_ind),i)
    figure
    for di=1:size(This_PIR,2)
       % tmp=find(onsets(:,di)>0);
      %  ind_on=tmp(min(find(tmp>7000))); if isempty(ind_on); ind_on=size(onsets,1); end
       
       %ind_off=tmp(max(find(tmp<off_limit))); if isempty(ind_off); ind_off=0; end
        subplot(size(This_PIR,2),1,di)
        plot([0:24/(size(This_PIR,1)-1):24],This_PIR(:,di),'k'); hold on
%        plot(M*onsets(:,di)); hold on
        ph=plot([activity_onsets(i,di) activity_onsets(i,di)],[0 M]); set(ph,'LineWidth',3,'color','r'); hold on
        kh=plot([Hour_mean(i,di) Hour_mean(i,di)],[0 M]); set(kh,'LineWidth',3,'color','b'); hold on
        lh=plot([activity_offsets(i,di) activity_offsets(i,di)],[0 M]); set(lh,'LineWidth',3,'color','g'); hold on
        
        %all_ind_on(i,di)=ind_on;
        %all_ind_off(i,di)=ind_off;
        set(gca, 'Yticklabel',[])
        set(gca, 'Xtick',[0 One_day_COMP/4 One_day_COMP/2 One_day_COMP*3/4 One_day_COMP]); set(gca, 'Xticklabel',[])
%        if di==size(onsets,2);
            set(gca, 'Xticklabel',{'0', '6', '12' ,'18' ,'24'},'fontsize',12)
            xlabel('Time (hours)','fontsize',10);
 %       end
        if di==1; title(exp_title); end;
    end
   
end



% plot as map

figure
for i=1:length(mouse_ind)
    clear This_PIR2 This_PIR
    subplot(1,length(mouse_ind),i)
    This_PIR(:,:)=ALL_PIR(:,mouse_ind(i),:);
%     if ~isEven(size(This_PIR,2)); This_PIR=[This_PIR nan(size(This_PIR,1),1)];end
%     This_PIR2=reshape(This_PIR(:,:),[size(This_PIR(:,:),1)*2,size(This_PIR(:,:),2)/2]);
    B=cat(2,This_PIR,zeros(size(This_PIR,1),1));
    A=cat(2,zeros(size(This_PIR,1),1),This_PIR);
    This_PIR2=[A;B];
    clims = [0 90];
    imagesc(This_PIR2',clims)
    %imagesc(This_PIR')
    colormap('hot')
    colorbar;
    xlabel('Time (hours)','fontsize',15)
    ylabel('Days','fontsize',15)
    set(gca, 'Xtick',[0 One_day_COMP/4 One_day_COMP/2 One_day_COMP*3/4 One_day_COMP One_day_COMP*5/4 One_day_COMP*3/2 One_day_COMP*7/4 One_day_COMP*2])
    set(gca, 'Xticklabel',{'0', '6', '12' ,'18' ,'24', '30', '36' ,'42' ,'48'},'fontsize',12)
    title(exp_title)
end

% 
% % plot as circle
% figure
% for i=1:length(mouse_ind)
%     subplot(1,length(mouse_ind),i)
%     This_PIR(:,:)=ALL_PIR(:,mouse_ind(i),:);
%   
%     binned_PIR=PIR2binned(This_PIR);
% %     for di=1:size(binned_PIR,2)
% %         binned_PIR_rad{di} = circ_ang2rad(binned_PIR{di});       % convert to radians
% %             subplot(5,ceil(size(binned_PIR,2)/5),di)
% %             circ_plot(binned_PIR_rad{di},'hist',[],20,true,true,'linewidth',2,'color','r')
% %     end
% %     fprintf('\nTHE CIRCSTAT TOOLBOX EXAMPLE\n\nDescriptive Statistics\n')
% end


%% part2: plot data (generate figure 1)
if mean_Fig
figure(1)
%     subplot(2,2,1)
%     circ_plot(alpha_rad,'pretty','bo',true,'linewidth',2,'color','r'),
%
for i=1:size(Hour_mean,1)
    figure ('name',[filename ' ' mouseID{i}])
    
    subplot (5,1,1);
    imshow(LDR_tiff);
    
    subplot (5,1,2:5);
    bar(PIR(:,mouse_ind(i)),'FaceColor',[0.35,0,0.55]) ; hold on
    
    xlim([0 One_day_COMP])
    legend({'motion sensor'},'fontsize',12)
    set(gca, 'Xticklabel',[])
    set(gca, 'Yticklabel',[])
    ylabel('PIR sensor','fontsize',15)
    set(gca, 'Xtick',[0 One_day_COMP/4 One_day_COMP/2 One_day_COMP*3/4 One_day_COMP])
    set(gca, 'Xticklabel',{'0', '6', '12' ,'18' ,'24'},'fontsize',12)
    set(gca, 'Yticklabel',[])
    xlabel('Time (hours)','fontsize',15)
    
end
title(exp_title)
end

figure ('name',[filename ' ' mouseID{i}])

subplot (4,1,1);
imshow(LDR_tiff);

for i=1:length(mouse_ind)
    subplot (4,1,i+1);
    bar(PIR(:,mouse_ind(i)),'FaceColor',[0.35,0,0.55]) ; hold on
    xlim([0 One_day_COMP])
    legend({'motion sensor'},'fontsize',12)
    set(gca, 'Xticklabel',[])
    set(gca, 'Yticklabel',[])
    ylabel('PIR sensor','fontsize',15)
    set(gca, 'Xtick',[0 One_day_COMP/4 One_day_COMP/2 One_day_COMP*3/4 One_day_COMP])
    set(gca, 'Xticklabel',{'0', '6', '12' ,'18' ,'24'},'fontsize',12)
    set(gca, 'Yticklabel',[])
    xlabel('Time (hours)','fontsize',15)
    
end
title(exp_title)

%output.all_ind_on=all_ind_on % hour_mean is a better method. based on
%circular statistics
%output.all_ind_off=all_ind_off;
output.exp_title=exp_title;
output.hour_mean_activity=Hour_mean;
output.activity_onsets=activity_onsets;
output.activity_offsets=activity_offsets;
output.activity_periods=activity_period;


