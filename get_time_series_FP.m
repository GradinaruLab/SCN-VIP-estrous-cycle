function get_time_series_FP
%% this function is used to analyse the time dependent time-series FP data, 10 minutes per hour
% Taken with SynapseTDT

%% get experimental information: 
my_path='D:\DATA_Glab\fiberphotometry\';
[NUM,TXT]=xlsread([my_path 'FP_VIP_GC4.xlsx']);
by_estrous=0; % one means by P E M D 
%%%%%%%%%%%%%%%%%%
% Females. 1-6
ind=0;
ind =ind+1; mouse_info{ind}.ID='198R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='200LL';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';%
ind=ind+1;mouse_info{ind}.ID='246RL';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Female';%%rates detected are very low- 
ind=ind+1; mouse_info{ind}.ID='247RRL';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='259R'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='260L'; mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='261RL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female'; %X=8; % has 3 completed cycles, remove the last 8 sessions to look at complete 3 sets
ind=ind+1; mouse_info{ind}.ID='313RL';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
%mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;
%mouse_info{ind}.sex='Female';% signal was very low

n_females=ind;
% % Males 7-12
% ind=ind+1;mouse_info{ind}.ID='262R'; mouse_info{ind}.side='R'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male'; %male
% ind=ind+1; mouse_info{ind}.ID='273RL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
% ind=ind+1; mouse_info{ind}.ID='286R'; mouse_info{ind}.side='R'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
% ind=ind+1;mouse_info{ind}.ID='287L';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male not good
% ind=ind+1; mouse_info{ind}.ID='288RL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
% ind=ind+1; mouse_info{ind}.ID='296R'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
% n_males=6;
% %OVX 13-16
% ind=ind+1; mouse_info{ind}.ID='247RRL_OVX';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
% ind=ind+1; mouse_info{ind}.ID='246RL_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
% ind=ind+1;mouse_info{ind}.ID='261RL_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
% ind=ind+1; mouse_info{ind}.ID='259R_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX'; % OVX
% n_OVX=4;


if by_estrous %  by P E M D
    for idi=1:n_females
        mouse_info{idi}.sex='Female2';
    end
end
    

% get data for each mouse
% for idi=1:length(mouse_info)
%     [output{idi}] = get_time_series_FP_per_mouse (mouse_info{idi});
% end
% get data for each mouse
for idi=1:length(mouse_info)
    switch mouse_info{idi}.sex
        case 'Female2'
                if ~exist ([my_path 'time_series_output_general2_' mouse_info{idi}.ID '.mat' ])
                    disp(['get data for ' mouse_info{idi}.ID])
                    output= get_time_series_FP_per_mouse (mouse_info{idi});
                else
                    load([my_path 'time_series_output_general2_' mouse_info{idi}.ID '.mat' ])% load 'output'
                end
        case 'Female'
                if ~exist ([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])
                    disp(['get data for ' mouse_info{idi}.ID])
                    output= get_time_series_FP_per_mouse (mouse_info{idi});
                else
                    load([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])% load 'output'
                end
%         case {'Male','OVX'}
%             if ~exist ([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])
%                 disp(['get data for ' mouse_info{idi}.ID])
%                 output = get_time_series_FP_per_mouse (mouse_info{idi});
%             else
%                 load([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])%all_output{idi}=output;
%             end 
    end
    output.new_estrus_states=estrus_to_receptive(output.estrus_states);
    all_output{idi}=output;
end
clear output
output=all_output; 
clear all_output
%%%%%%%%%%%%%%%    

if by_estrous
    estrus_states_titles={'P','E','M','D','OVX'};
    for i=1:length(mouse_info)
        mouse_info{i}.analysis_type='estrous_cycle2';
    end
else
    estrus_states_titles={'P-2','P-1','P+0','P+1','P+2','OVX'};
    for i=1:length(mouse_info)
        mouse_info{i}.analysis_type='estrous_cycle';
    end
end
[ALL_colors,color_ind]=get_estrus_colors(estrus_states_titles);


L=[];
for idi=1:length(mouse_info)
    L=[L size(output{idi}.mean_dF_over_all_samples,2)];
end


% arrange data
clear med_dF_dark_light_over_estrus_samples med_rate_dark_light_over_estrus_samples med_width_dark_light_over_estrus_samples
clear mean_dF_over_all_samples mean_dF_over_estrus_samples mean_dF_by_state mean_rate_over_estrus_samples
clear median_dF_over_all_samples median_dF_over_estrus_samples median_dF_by_state median_rate_over_estrus_samples
for idi=1:length(mouse_info)
    if size(output{idi}.mean_dF_over_estrus_samples,1)<size(estrus_states_titles,2) % no OVX for this female- add nan arrays to match size 
        output{idi}.mean_dF_over_estrus_samples=cat(1, output{idi}.mean_dF_over_estrus_samples(:,:,1:min(L)),nan(1,24,min(L)));
        output{idi}.mean_rate_over_estrus_samples=cat(1,output{idi}.mean_rate_over_estrus_samples,nan(1,24));
        output{idi}.mean_width_over_estrus_samples=cat(1,output{idi}.mean_width_over_estrus_samples,nan(1,24));
        output{idi}.median_dF_over_estrus_samples=cat(1, output{idi}.median_dF_over_estrus_samples(:,:,1:min(L)),nan(1,24,min(L)));
        output{idi}.median_rate_over_estrus_samples=cat(1,output{idi}.median_rate_over_estrus_samples,nan(1,24));
        output{idi}.median_width_over_estrus_samples=cat(1,output{idi}.median_width_over_estrus_samples,nan(1,24));
        output{idi}.mean_dF_by_state=cat(1,output{idi}.mean_dF_by_state,nan(1,24));
        output{idi}.median_dF_by_state=cat(1,output{idi}.median_dF_by_state,nan(1,24));
        output{idi}.median_dF_dark_light=[output{idi}.median_dF_dark_light nan nan];% add two nans for OVX light and dark phasese
        output{idi}.median_rate_dark_light=[output{idi}.median_rate_dark_light nan nan];% add two nans for OVX light and dark phasese
        output{idi}.median_width_dark_light=[output{idi}.median_width_dark_light nan nan];% add two nans for OVX light and dark phasese
    end
    mean_dF_over_all_samples(idi,:,:)=output{idi}.mean_dF_over_all_samples(:,1:min(L));
    median_dF_over_all_samples(idi,:,:)=output{idi}.median_dF_over_all_samples(:,1:min(L));
    %a=output{idi}.mean_dF_over_estrus_samples;
    mean_dF_over_estrus_samples(idi,:,:,:)=output{idi}.mean_dF_over_estrus_samples(:,:,1:min(L));% ID, estrus, hour, time
    median_dF_over_estrus_samples(idi,:,:,:)=output{idi}.median_dF_over_estrus_samples(:,:,1:min(L));% ID, estrus, hour, time
    %b=output{idi}.mean_dF_by_state;
    mean_dF_by_state(idi,:,:)=output{idi}.mean_dF_by_state;% ID, estrus, hour
    mean_rate_over_estrus_samples(idi,:,:)=output{idi}.mean_rate_over_estrus_samples;% ID, estrus, hour
    mean_width_over_estrus_samples(idi,:,:)=output{idi}.mean_width_over_estrus_samples;% ID, estrus, hour

    median_dF_by_state(idi,:,:)=output{idi}.median_dF_by_state;% ID, estrus, hour
    median_rate_over_estrus_samples(idi,:,:)=output{idi}.median_rate_over_estrus_samples;% ID, estrus, hour
    median_width_over_estrus_samples(idi,:,:)=output{idi}.median_width_over_estrus_samples;% ID, estrus, hour
    
    med_dF_dark_light_over_estrus_samples(idi,:)=output{idi}.median_dF_dark_light;% ID, estrus/dark/light
    med_rate_dark_light_over_estrus_samples(idi,:)=output{idi}.median_rate_dark_light;% ID,  estrus/dark/light
    med_width_dark_light_over_estrus_samples(idi,:)=output{idi}.median_width_dark_light;% ID,  estrus/dark/light

end

% average property over specific time window
% results an array of ID, estrus
clear med_dF_over_estrus_samples_window med_rate_over_estrus_samples_window med_width_over_estrus_samples_window

CF=0.8/length(mouse_info);% color factor
if by_estrous
    full_time_estrus_states_titles={'P D','P L','E D','E L','M D','M L','Di D','Di L','OVX L', 'OVX D'};
else
    full_time_estrus_states_titles={'P-2 D','P-1 L','P-1 D','P+0 L','P+0 D','P+1 L','P+1 D','P+2 L','P+2 D' 'OVX L', 'OVX D'};
end
% test the results
figure
subplot(1,3,1)
for idi=1:length(mouse_info)
    plot(med_dF_dark_light_over_estrus_samples(idi,:)); hold on
end
ylabel('df')
subplot(1,3,2)
for idi=1:length(mouse_info)
    plot(med_rate_dark_light_over_estrus_samples(idi,:)); hold on
end
ylabel('rate')
subplot(1,3,3)
for idi=1:length(mouse_info)
    plot(med_width_dark_light_over_estrus_samples(idi,:)); hold on
end
ylabel('width')

data_to_explore=med_rate_dark_light_over_estrus_samples;
data_to_explore=med_dF_dark_light_over_estrus_samples;
Light=[2:2:size(data_to_explore,2)];
disp ('rates all light phase, different estrous stages (+-sem): ')
data_light=nanmedian(data_to_explore(:,Light));
data_light_sem=nanstd(data_to_explore(:,Light))/sqrt(size(data_to_explore,1));
disp (num2str(data_light))
disp(num2str(data_light_sem))
disp([num2str(mean(data_light)) '+-' num2str(mean(data_light_sem))] )

Dark=[1:2:size(data_to_explore,2)];
disp ('data for all light phase, different estrous stages (+-sem): ')
data_dark=nanmedian(data_to_explore(:,Dark));
data_dark_sem=nanstd(data_to_explore(:,Dark))/sqrt(size(data_to_explore,1));
disp (num2str(data_dark))
disp(num2str(data_dark_sem))
disp([num2str(mean(data_dark)) '+-' num2str(mean(data_dark_sem))] )



% statistics
figure; [p,tbl,stats] = kruskalwallis(med_dF_dark_light_over_estrus_samples,[],'off');
c_dF= multcompare(stats,'CType','hsd');c_dF(:,end+1)=c_dF(:,6)<0.05;% Tukey's honest significant difference criterion correction
figure; [p,tbl,stats_r] = kruskalwallis(med_rate_dark_light_over_estrus_samples,[],'off');
c_r = multcompare(stats_r,'CType','hsd');c_r(:,end+1)=c_r(:,6)<0.05;
% figure; [p,tbl,stats_w] = kruskalwallis(med_width_dark_light_over_estrus_samples,[],'off');
% c_w = multcompare(stats_w,'CType','hsd');c_w(:,end+1)=c_w(:,6)<0.05;
% plot the median dF and rate over light/dark
%dF
% set a new color index for dark- light 
color_ind2=[color_ind(1)]; 
for i=2:length(color_ind)
    color_ind2=[color_ind2 color_ind(i) color_ind(i)];
end  
figure
subplot(2,1,1)
for sti=1:length(color_ind2)
    % plot bars
    tmp=nanmean(med_dF_dark_light_over_estrus_samples(:,sti));
    bh=bar(sti, tmp); hold on;
    set(bh,'FaceColor', ALL_colors(color_ind2(sti),:))
    % error bar
    sem=nanstd(med_dF_dark_light_over_estrus_samples(:,sti))/sqrt(length(med_dF_dark_light_over_estrus_samples(:,sti)));
    lh=line([sti sti],[tmp+sem tmp-sem]); lh.Color='k'; hold on;
end
for idi=1:length(mouse_info) 
    lh1=plot(med_dF_dark_light_over_estrus_samples(idi,:),'-*'); hold on;
    set(lh1,'Color', [CF*idi CF*idi CF*idi])
end
xticklabels(full_time_estrus_states_titles)
ylabel ('mean dF')
ylim([0 15.5])
%rates
subplot(2,1,2)
for sti=1:length(color_ind2)
    tmp=nanmean(med_rate_dark_light_over_estrus_samples(:,sti));
    bh=bar(sti,tmp); hold on;
    set(bh,'FaceColor', ALL_colors(color_ind2(sti),:))
    % error bar
    sem=nanstd(med_rate_dark_light_over_estrus_samples(:,sti))/sqrt(length(med_rate_dark_light_over_estrus_samples(:,sti)));
    lh=line([sti sti],[tmp+sem tmp-sem]); lh.Color='k'; hold on;
end
%bar(nanmedian(med_rate_dark_light_over_estrus_samples)); hold on;
for idi=1:length(mouse_info) 
    lh1=plot(med_rate_dark_light_over_estrus_samples(idi,:),'-*'); hold on;
    set(lh1,'Color', [CF*idi CF*idi CF*idi])
end
xticklabels(full_time_estrus_states_titles)
ylabel ('mean rates')
ylim([0 0.7])
%width
% subplot(3,1,3)
% bar(median(med_width_dark_light_over_estrus_samples)); hold on;
% for idi=1:length(mouse_info) 
%     lh1=plot(med_width_dark_light_over_estrus_samples(idi,:),'-*'); hold on;
%     set(lh1,'Color', [CF*idi CF*idi CF*idi])
% end
% xticklabels(full_time_estrus_states_titles)
% ylabel ('median width')
% prabola fit dF/F

%% plot specific window data
clear lh1
% shift data: original data taken at strating at 9am, while at 13 light
% turned off. Data shifted that 1am (Light) will be the start time
% setting the first one: 
by='MEAN'
switch by
    case 'MEDIAN'
        dF_to_plot=median_dF_by_state;
        rate_to_plot=median_rate_over_estrus_samples;
        width_to_plot=median_width_over_estrus_samples;
    case 'MEAN'
        dF_to_plot=mean_dF_by_state;
        rate_to_plot=mean_rate_over_estrus_samples;
        width_to_plot=mean_width_over_estrus_samples;
end
dF_by_state_DL(:,1,:)=cat(3,dF_to_plot(:,1,17:24),dF_to_plot(:,1,1:16));
rate_over_estrus_samples_DL(:,1,:)=cat(3,rate_to_plot(:,1,17:24),rate_to_plot(:,1,1:16));
width_over_estrus_samples_DL(:,1,:)=cat(3,width_to_plot(:,1,17:24),width_to_plot(:,1,1:16));
% tricky point here- the first 4 hours are fake- to make them nan 
dF_by_state_DL(:,1,1:8)=nan;
rate_over_estrus_samples_DL(:,1,1:8)=nan;
width_over_estrus_samples_DL(:,1,1:8)=nan;
% set all the other estrous states: 
for esi=1:4
    dF_by_state_DL(:,esi+1,:)=cat(3,dF_to_plot(:,esi,17:24),dF_to_plot(:,esi+1,1:16));
    rate_over_estrus_samples_DL(:,esi+1,:)=cat(3,rate_to_plot(:,esi,17:24),rate_to_plot(:,esi+1,1:16));
    width_over_estrus_samples_DL(:,esi+1,:)=cat(3,width_to_plot(:,esi,17:24),width_to_plot(:,esi+1,1:16));
end

% now shift the OVX to fit 24h DL
dF_by_state_DL(:,6,:)=cat(3,dF_to_plot(:,6,17:24),dF_to_plot(:,6,1:16));
rate_over_estrus_samples_DL(:,6,:)=cat(3,rate_to_plot(:,6,17:24),rate_to_plot(:,6,1:16));
width_over_estrus_samples_DL(:,6,:)=cat(3,width_to_plot(:,6,17:24),width_to_plot(:,6,1:16));

%window=[1:24];
for i=1:2:13
    window=[i:i+1];% LH onset
    med_dF_over_estrus_samples_window(:,:)=nanmedian(dF_by_state_DL(:,:,window),3);% ID, estrus, hour
    med_rate_over_estrus_samples_window(:,:)=nanmedian(rate_over_estrus_samples_DL(:,:,window),3);% ID, estrus, hour
    med_width_over_estrus_samples_window(:,:)=nanmedian(width_over_estrus_samples_DL(:,:,window),3);% ID, estrus, hour
    
    mean_dF_over_estrus_samples_window(:,:)=nanmean(dF_by_state_DL(:,:,window),3);% ID, estrus, hour
    mean_rate_over_estrus_samples_window(:,:)=nanmean(rate_over_estrus_samples_DL(:,:,window),3);% ID, estrus, hour
    mean_width_over_estrus_samples_window(:,:)=nanmean(width_over_estrus_samples_DL(:,:,window),3);% ID, estrus, hour
    
    %%% pay attaention here -choose median or mean
    figure
    subplot(2,1,1)
    bar(nanmedian(mean_dF_over_estrus_samples_window)); hold on;
    for idi=1:size(med_dF_over_estrus_samples_window,1)
        lh1=plot(med_dF_over_estrus_samples_window(idi,:),'-*'); hold on;
        set(lh1,'Color', [CF*idi CF*idi CF*idi])
    end
    xticklabels(estrus_states_titles)
    ylim([0 25])
    ylabel ('median/mean dF')
    title(['window : ' num2str(window(1)) ' to ' num2str(window(end))])
    %rates
    subplot(2,1,2)
    bar(nanmedian(med_rate_over_estrus_samples_window)); hold on;
    for idi=1:size(med_rate_over_estrus_samples_window,1)
        lh1=plot(med_rate_over_estrus_samples_window(idi,:),'-*'); hold on;
        set(lh1,'Color', [CF*idi CF*idi CF*idi])
    end
    xticklabels(estrus_states_titles)
    ylim([0 1])
    ylabel ('median/mean rates')
    title(['window : ' num2str(window(1)) ' to ' num2str(window(end))])
    
    
    % statistics
%     figure; [p2,tbl,stats_df] = kruskalwallis(med_dF_over_estrus_samples_window,[],'off');
%     c_dF2= multcompare(stats_df,'CType','hsd');
%     figure; [p1,tbl,stats_r] = kruskalwallis(med_rate_over_estrus_samples_window,[],'off');
%     c_r2 = multcompare(stats_r,'CType','hsd');
end

% compare time windows % Figure 3
clear window mean_rate_over_estrus_samples_window median_rate_over_estrus_samples_window
window{1}=[4:5];
window{2}=[8:9];
window{3}=[10:11];
window{4}=[14:15];
for wi=1:length(window)
    switch by
        case 'MEDIAN'
            median_rate_over_estrus_samples_window{wi}(:,:)=nanmedian(rate_over_estrus_samples_DL(:,:,window{wi}),3);% ID, estrus, hour
        case 'MEAN'
            median_rate_over_estrus_samples_window{wi}(:,:)=nanmean(rate_over_estrus_samples_DL(:,:,window{wi}),3);% ID, estrus, hour
    end
end
figure
for wi=1:length(window)
    subplot(length(window),1,wi)
    for sti=1:length(color_ind)
        bh=bar(sti, nanmedian(median_rate_over_estrus_samples_window{wi}(:,sti))); hold on;
        set(bh,'FaceColor', ALL_colors(color_ind(sti),:))
    end
    for idi=1:size(median_rate_over_estrus_samples_window{wi},1)
        lh1=plot(median_rate_over_estrus_samples_window{wi}(idi,:),'-*'); hold on;
        set(lh1,'Color', [CF*idi CF*idi CF*idi])
        
    end
    xticklabels(estrus_states_titles)
    ylabel ([by ' rates'])
    ylim([0 1])
    title(['window : ' num2str(window{wi}(1)) ' to ' num2str(window{wi}(end))])
end
legend(estrus_states_titles)


% plot rates
clear tmp
figure

for sti=1:length(estrus_states_titles)
    subplot(length(estrus_states_titles),1,sti)
    for idi=1:length(mouse_info)
        tmp(:)=rate_over_estrus_samples_DL(idi,sti,:);
        lh=plot(tmp); hold on
        set(lh,'Color', ALL_colors(color_ind(sti),:))
    end
    clear tmp
    ylim([0 0.9])
    xlim ([0 25])
    title ('rates')
    
end

% plot width
figure
clear tmp

for sti=1:length(estrus_states_titles)
    subplot(length(estrus_states_titles),1,sti)
    for idi=1:length(mouse_info)
        tmp(:)=width_over_estrus_samples_DL(idi,sti,:);
        lh=plot(tmp); hold on
        set(lh,'Color', ALL_colors(color_ind(sti),:))
    end
    clear tmp
    ylim([0 30])
    title ('width')
end



% Figure 3
figure
% plot mean rates
clear tmp
for sti=1:length(estrus_states_titles)
    subplot(length(estrus_states_titles),1,sti)
    % plot bar of mean/median
    tmp(:)=nanmean(rate_over_estrus_samples_DL(:,sti,:));
    lh=bar(tmp); hold on
    set(lh,'FaceColor', ALL_colors(color_ind(sti),:))
   % plot individual points
    for idi=1:length(mouse_info)
        tmp2(:)=rate_over_estrus_samples_DL(idi,sti,:);
        lh=plot(tmp2,'.'); hold on
        set(lh,'Color', [0 0 0])
    end
   % plot error bars
    for hi=1:size(rate_over_estrus_samples_DL,3)
        sem=nanstd(rate_over_estrus_samples_DL(:,sti,hi))/sqrt(length(rate_over_estrus_samples_DL(:,sti,hi)));
        lh2=line([hi hi],[tmp(hi)+sem tmp(hi)-sem]); hold on
        lh2.Color='k';
    end
    title (estrus_states_titles{sti})
    ylabel('median rates')
    ylim([0 1.4])
end
legend (estrus_states_titles)

% plot mean width
figure
subplot(2,1,2)
for sti=1:length(estrus_states_titles)
    subplot(length(estrus_states_titles),1,sti)
    tmp(:)=nanmedian(width_over_estrus_samples_DL(:,sti,:));
    lh=bar(tmp); hold on
    set(lh,'FaceColor', ALL_colors(color_ind(sti),:))
    % plot individual point 
    for idi=1:length(mouse_info)
        tmp2(:)=width_over_estrus_samples_DL(idi,sti,:);
        lh=plot(tmp2,'.'); hold on
        set(lh,'Color', [0 0 0])
    end
    % plot error bars
    for hi=1:size(width_over_estrus_samples_DL,3)
        sem=nanstd(width_over_estrus_samples_DL(:,sti,hi))/sqrt(length(width_over_estrus_samples_DL(:,sti,hi)));
        lh2=line([hi hi],[tmp(hi)+sem tmp(hi)-sem]); hold on
        lh2.Color='k';
    end
    title (estrus_states_titles{sti})
    ylabel('median width')
    ylim([0 75])
end


% plot dF
figure
for sti=1:length(estrus_states_titles)
    subplot(length(estrus_states_titles),1,sti)
    for idi=1:length(mouse_info)
        tmp(:)=dF_by_state_DL(idi,sti,:);
        lh=plot(tmp); hold on
        set(lh,'Color', ALL_colors(color_ind(sti),:))
    end
    ylim([0 15])
    ylabel ('dF')
    title (estrus_states_titles{sti})
end

% plot mean dF
figure
for sti=1:length(estrus_states_titles)
    subplot(length(estrus_states_titles),1,sti)
    tmp(:)=nanmedian(dF_by_state_DL(:,sti,:));
    bh=bar(tmp); hold on
    set(bh,'FaceColor', ALL_colors(color_ind(sti),:))
    
    for idi=1:length(mouse_info)
        tmp2(:)=dF_by_state_DL(idi,sti,:);
        lh=plot(tmp2,'.'); hold on
        set(lh,'Color', [0 0 0])
    end
    for hi=1:size(dF_by_state_DL,3)
        sem=nanstd(dF_by_state_DL(:,sti,hi))/sqrt(length(dF_by_state_DL(:,sti,hi)));
        lh2=line([hi hi],[tmp(hi)+sem tmp(hi)-sem]); hold on
        lh2.Color='k';
    end
    ylim([-0.5 15])
   ylabel ('Median dF')
    title (estrus_states_titles{sti})
end

%legend (estrus_states_titles)
% statistics:
list={'rate_over_estrus_samples_DL'; 'dF_by_state_DL'};
p_names={'rates'; 'dF'};
for pi=1:length(list)
    eval(['P=' list{pi} ';'])
    G=[]; % define groups for kruskalwallis
    data_to_plot=[];
    for sti=1:size(P,2)
        data_to_plot=cat(1,data_to_plot,P(:,sti,:));%(idi,sti,hi);id, state, hour
        data_to_plot2=reshape(data_to_plot, size(data_to_plot,1), size(data_to_plot,3));
        G=[G; sti*ones(size(P,1),1)];
    end
    figure;
    for hi=1:24
        subplot(2,12,hi)
        [p,tbl,stats_r] = kruskalwallis(data_to_plot2(:,hi),G,'off');
        c_r2{hi} = my_multcompare(stats_r,'CType','dunn-sidak');
        title([ ' hour=' num2str(hi)])
        ylabel(p_names{pi})
    end
   
    for hi=1:24
        h{hi}=find(c_r2{hi}(:,6)<0.05);
        if ~isempty(h{hi}); disp('found some significance!!!!'); end
        % if isempty(h{hi}); disp('no significance :-('); end
    end
end
%%
% prabola fit dF/F
clear beta_dF beta_rate tmp
zt_fit=[2:11];
all_data_to_plot={'dF',  'rate'};
figure
for di=1:length(all_data_to_plot)
    clear  ind ind_max
    switch all_data_to_plot{di}
        case 'dF'
            data_to_plot=dF_by_state_DL; 
        case 'rate'
            data_to_plot=rate_over_estrus_samples_DL;
    end
    subplot(length(all_data_to_plot),1,di)
    for sti=2:length(estrus_states_titles) % starts with 2 as P-2 doesn't have the full 24h data 
        %subplot(length(estrus_states_titles),1,sti)
        clear beta_dF 
        tmp(:)=nanmean(data_to_plot(:,sti,:));
        lh=plot(zt_fit,tmp(zt_fit+1),'o'); hold on
        set(lh,'Color', ALL_colors(color_ind(sti),:))
        Y=tmp(zt_fit+1)';
        x=[zt_fit]';
        A=[ones(length(x),1) x x.^2];
        beta_dF(sti,:)=A\Y;
        lh2=plot(x, beta_dF(sti,1)+beta_dF(sti,2)*x+beta_dF(sti,3)*x.^2);
        
        set(lh2,'Color', ALL_colors(color_ind(sti),:))
        % calculate R square
        SSR=sum((tmp(zt_fit+1)'-(beta_dF(sti,1)+beta_dF(sti,2)*x+beta_dF(sti,3)*x.^2)).^2);
        SST=sum((tmp(zt_fit+1)'-mean(tmp(zt_fit+1)')).^2);
        R_sqr(sti-1)=1-SSR/SST;
        [val(sti-1),ind(sti-1)]=min(beta_dF(sti,1)+beta_dF(sti,2)*x+beta_dF(sti,3)*x.^2); % look for min ZT
        [val_max(sti-1),ind_max(sti-1)]=max(beta_dF(sti,1)+beta_dF(sti,2)*x+beta_dF(sti,3)*x.^2); % look for min ZT
        leg_est{sti-1}=sprintf('Estimated (y=%.4f+%.4fx+%.4fx^2',beta_dF(sti,1),beta_dF(sti,2),beta_dF(sti,3));
        legend('Data',leg_est{sti})
    end
    ylabel(['parabolic fit to ' all_data_to_plot{di} ])
    xlabel('Time (ZT)')
    xlim([min(zt_fit)-0.5 max(zt_fit)+0.5])
    disp(['Min parabola ' all_data_to_plot{di}  ' is at ZT ' num2str(mean(ind)) ' +- ' num2str(std(ind)/sqrt(length(ind)))])
    disp(['Max parabola ' all_data_to_plot{di}  ' is at ZT ' num2str(mean(ind_max(ind_max>mean(ind)))) ' +- ' num2str(std(ind_max(ind_max>mean(ind)))/sqrt(length(ind_max)))])
     disp(['Parabola Rsqr ' all_data_to_plot{di}  ' is ' num2str(mean(R_sqr)) ' +- ' num2str(std(R_sqr)/sqrt(length(R_sqr)))])
end

% figure
% for idi=1:length(mouse_info)
%     for hi=1:size(output{idi}.mean_dF_over_all_samples,1)
%         plot(output{idi}.mean_dF_over_all_samples(hi,:)+25*hi); hold on
%     end
%     ylim([-20 (hi+3)*25])
% end
% title('all mice')


% plot mean dF per ID
CF=0.8/length(mouse_info);% color factor
figure
for sti=1:length(estrus_states_titles)
    subplot(length(estrus_states_titles),1,sti)
     for idi=1:length(mouse_info)
        tmp(:)=dF_by_state_DL(idi,sti,:);
        lh=plot(tmp,'-*'); hold on
        set(lh,'Color', [CF*idi CF*idi CF*idi])
    end
    ylim([-2 20])
   title ([ estrus_states_titles{sti} ', median dF'])
end
legend (estrus_states_titles)


for sti=1:length(estrus_states_titles)
    all_df_by_state(sti,:)=nanmedian(dF_by_state_DL(:,sti,:));
    %histcounts(med_all_df_by_state(sti,:),24);
end

1