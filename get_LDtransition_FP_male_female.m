function get_LDtransition_FP_male_female
%% this function is used to analyse the time dependent time-series FP data, 10 minutes per hour
% Taken with SynapseTDT

%% get experimental information: 
my_path='D:\DATA_Glab\fiberphotometry\LDtransition\';

%%%%%%%%%%%%%%%%%%
%% get experimental information: 
% Females. 1-6
ind=0;
minimum_trials=1;
% females w/o OVX
%ind=ind+1; mouse_info{ind}.ID='68RL';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='93N';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='108L';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='110LL';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';%
% females with OVX+ OVX with hormones
ind=ind+1; mouse_info{ind}.ID='107R'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';
%ind=ind+1; mouse_info{ind}.ID='115L'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
%ind=ind+1; mouse_info{ind}.ID='A116R';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='119LL';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='122R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='123L';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='128R'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';

% % females w/o OVX
%ind=ind+1; mouse_info{ind}.ID='161RR'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='166R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='175L';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='176RL'; mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='198L'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='200RR'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
% % 

n_females=ind;
% Males 7-12
ind=ind+1; mouse_info{ind}.ID='25L'; mouse_info{ind}.side='L'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male'; %male
ind=ind+1; mouse_info{ind}.ID='60N'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
ind=ind+1; mouse_info{ind}.ID='62L'; mouse_info{ind}.side='R'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
ind=ind+1; mouse_info{ind}.ID='103R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male not good
ind=ind+1; mouse_info{ind}.ID='106LL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male

n_males=ind-n_females;

% get data for each mouse
% for idi=1:length(mouse_info)
%     [output{idi}] = get_time_series_FP_per_mouse (mouse_info{idi});
% end
% get data for each mouse
tic
for idi=1:length(mouse_info)
    mouse_info{idi}.minimum_trials=minimum_trials;
    switch mouse_info{idi}.sex
        case 'Female'
            if ~exist ([my_path 'output_estrous_cycle_' mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])
                disp(['get data for ' mouse_info{idi}.ID])
                output= get_LDtransition_FP_per_mouse (mouse_info{idi});
            else
                load([my_path 'output_estrous_cycle_' mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])% load 'output'
            end
        case {'Male'}
            if ~exist ([my_path 'output_male_female_' mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])
                disp(['get data for ' mouse_info{idi}.ID])
                output = get_LDtransition_FP_per_mouse (mouse_info{idi});
            else 
                load([my_path 'output_male_female_' mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])%all_output{idi}=output;
            end 
    end
    output.new_estrus_states=estrus_to_receptive(output.estrus_states);
    all_output{idi}=output;
end
toc
clear output
output=all_output; 
clear all_output
fs=output{1}.analysis{1}.new_fs;
%%%%%%%%%%%%%%%    
for i=1:length(mouse_info)
    mouse_info{i}.analysis_type='male_female';
end

estrus_states_titles={'Female','Male'};
[ALL_colors,color_ind]=get_estrus_colors(estrus_states_titles);

L=[];
for idi=1:length(mouse_info)
    L=[L size(output{idi}.mean_dF_over_all_samples,2)];
end

% arrange data
close all
clear med_dF_dark_light_over_estrus_samples med_rate_dark_light_over_estrus_samples nan_array
clear mean_dF_over_all_samples mean_dF_over_estrus_samples mean_dF_by_state mean_rate_over_estrus_samples median_width_over_estrus_samples
clear median_dF_over_all_samples median_dF_over_estrus_samples median_dF_by_state median_rate_over_estrus_samples
full_size=size(output{idi}.mean_rate_over_estrus_samples,2);
%for idi=12:16
for idi=1:length(mouse_info)
    %
            %
    %     mean_dF_over_estrus_samples(idi,:,:,:)=output{idi}.mean_dF_over_estrus_samples(:,:,1:min(L));% ID, estrus, hour, time
    %     %median_dF_over_estrus_samples(idi,:,:,:)=output{idi}.median_dF_over_estrus_samples(:,:,1:min(L));% ID, estrus, hour, time
    %
    %     mean_dF_by_state(idi,:,:)=output{idi}.mean_dF_by_state;% ID, estrus, hour
    %     mean_rate_over_estrus_samples(idi,:,:)=output{idi}.mean_rate_over_estrus_samples;% ID, estrus, hour
    %     %     mean_width_over_estrus_samples(idi,:,:)=output{idi}.mean_width_over_estrus_samples;% ID, estrus, hour
    %
    %     %median_dF_by_state(idi,:,:)=output{idi}.median_dF_by_state;% ID, estrus, hour
    %     % median_rate_over_estrus_samples(idi,:,:)=output{idi}.median_rate_over_estrus_samples;% ID, estrus, hour
    %     % median_width_over_estrus_samples(idi,:,:)=output{idi}.median_width_over_estrus_samples;% ID, estrus, hour
    switch mouse_info{idi}.sex
        case {'Male','male'}
            med_dF_dark_light_over_estrus_samples(idi,:)=output{idi}.mean_dF_dark_light;% ID, estrus/dark/light
            med_rate_dark_light_over_estrus_samples(idi,:)=output{idi}.mean_rate_dark_light;% ID,  estrus/dark/light
            
            mean_dF_over_all_samples(idi,:,:)=output{idi}.mean_dF_over_all_samples(:,1:min(L));
            median_dF_over_all_samples(idi,:,:)=output{idi}.median_dF_over_all_samples(:,1:min(L));
            
            G(idi)=2; % male
        case {'Female'} % take proestrous only 
            P_ind=3; OVX_ind=6; E_ind=4;
            if sum(~isnan(output{idi}.mean_dF_dark_light(P_ind,:)))>0 %|| sum(~isnan(output{idi}.mean_dF_dark_light(E_ind,:)))>0
                med_dF_dark_light_over_estrus_samples(idi,:)=output{idi}.mean_dF_dark_light(P_ind,:);% ID, estrus/dark/light
                med_rate_dark_light_over_estrus_samples(idi,:)=output{idi}.mean_rate_dark_light(P_ind,:);% ID,  estrus/dark/light
                
                mean_dF_over_all_samples(idi,:,:)=output{idi}.mean_dF_over_all_samples(:,1:min(L));
                median_dF_over_all_samples(idi,:,:)=output{idi}.median_dF_over_all_samples(:,1:min(L));
                
                G(idi)=1; % female
            end
%             if sum(~isnan(output{idi}.mean_dF_dark_light(OVX_ind,:)))>0
%                 med_dF_dark_light_over_estrus_samples(idi,:)=output{idi}.mean_dF_dark_light(P_ind,:);% ID, estrus/dark/light
%                 med_rate_dark_light_over_estrus_samples(idi,:)=output{idi}.mean_rate_dark_light(P_ind,:);% ID,  estrus/dark/light
%                 G(idi)=2; % OVX
%             end
    end
    %     med_width_dark_light_over_estrus_samples(idi,:)=output{idi}.median_width_dark_light;% ID,  estrus/dark/light
end

clear med_dF_over_estrus_samples_window med_rate_over_estrus_samples_window med_width_over_estrus_samples_window

CF=0.8/length(mouse_info);% color factor
full_time_estrus_states_titles={'P-2 D','P-1 L','P-1 D','P+0 L','P+0 D','P+1 L','P+1 D','P+2 L','P+2 D' 'OVX L', 'OVX D'};

b=0.15;
X=[1-b 1+b; 2-b 2+b; 3-b 3+b];

% arrange traces
clear df
dF_mat=[];
dt=output{1}.t(end)-output{1}.t(1);
for idi=1:size(mean_dF_over_all_samples,1)
    all_dF{idi}=[];
    all_t{idi}=[];
    for hi=7:size(mean_dF_over_all_samples,2)
        df(:)=mean_dF_over_all_samples(idi,hi,10*fs:end-fs);% id, hour, sampling points
        all_dF{idi}=[all_dF{idi} df];
        all_t{idi}=[ all_t{idi} output{1}.t(10*fs:end-fs)+(hi-1)*dt];
    end
    dF_mat=[dF_mat ; all_dF{idi}];
end
dt=all_t{1};

figure
for i=1:2
subplot(2,1,i)
% female
for idi=1:size(mean_dF_over_all_samples,i); hold on
    if G(idi)==i
        plot(dt,dF_mat(idi,:),'k')
    end
end
ph=plot(dt,nanmean(dF_mat(find(G==i),:),1));hold on
ph.Color=ALL_colors(color_ind(i),:);
end


% plot light/dark
% dF
figure
for sti=1:length(color_ind)
    bh=bar(sti, nanmean(med_dF_dark_light_over_estrus_samples(G==sti,2:3))); hold on;
    set(bh,'FaceColor', ALL_colors(color_ind(sti),:))
end
for gi=1:3
      ph=plot([X(gi,1)*ones(length(find(G==gi)),1) X(gi,2)*ones(length(find(G==gi)),1)],med_dF_dark_light_over_estrus_samples(G==gi,2:3),'.'); hold on
    for pi=1:length(ph)
        ph(pi).Color=ALL_colors(color_ind(gi),:);
        ph(pi).Color=[0 0 0];
    end
    
end
ylabel('dF/F')
%ylim([0 5]);
%rates: 
figure
for sti=1:length(color_ind)
    bh=bar(sti, nanmedian(med_rate_dark_light_over_estrus_samples(G==sti,2:3))); hold on;
    set(bh,'FaceColor', ALL_colors(color_ind(sti),:))
end
for gi=1:3
      ph=plot([X(gi,1)*ones(length(find(G==gi)),1) X(gi,2)*ones(length(find(G==gi)),1)],med_rate_dark_light_over_estrus_samples(G==gi,2:3),'.'); hold on
    for pi=1:length(ph)
        ph(pi).Color=ALL_colors(color_ind(gi),:);
        ph(pi).Color=[0 0 0];
    end  
end
ylabel('Event rates (events/min)')


% statistics
clear c_dF_MF c_rate_MF
% compare males/females
for k=2:3
    figure; [p,tbl,stats] = kruskalwallis([med_dF_dark_light_over_estrus_samples(find(G==1),k)' med_dF_dark_light_over_estrus_samples(find(G==2),k)'],G(G>0),'off');
    c_dF_MF{k}= multcompare(stats);% 
      figure; [p,tbl,stats] = kruskalwallis([med_rate_dark_light_over_estrus_samples(find(G==1),k)' med_rate_dark_light_over_estrus_samples(find(G==2),k)'],G(G>0),'off');
    c_rate_MF{k}= multcompare(stats);% 
end



% % plot the median dF and rate over light/dark
% %dF
% data_to_plot_names={'med_dF_dark_light_over_estrus_samples', 'med_rate_dark_light_over_estrus_samples'};
% ZTs={'10' '11' '12'}; YLIMs=[7 2]; YLABELs={'Median dF (a.u.)' ,'Median event rates (events/min)'};
% for pi=1:length(data_to_plot_names)
%     eval(['data_to_plot=' data_to_plot_names{pi} ';']);
%     figure
%     for k=1:3
%         subplot(1,3,k)
%         bar(nanmedian(data_to_plot(:,:,k))); hold on;
%         for idi=1:length(mouse_info)
%             lh1=plot(data_to_plot(idi,:,k),'*'); hold on;
%             set(lh1,'Color', [CF*idi CF*idi CF*idi])
%         end
%         xticklabels(estrus_states_titles)
%         ylabel (YLABELs{pi})
%         ylim([0 YLIMs(pi)])
%         title (['ZT ' ZTs{k}])
%     end
% end
% 
% %% plot dF and rates all states 
% data_to_plot_names={'mean_dF_by_state','mean_rate_over_estrus_samples'};
% YLIMs=[12 1.7]; YLABELs={'Mean dF (a.u.)' ,'Mean event rates (events/min)'};
% for pi=1:length(data_to_plot_names)
%     eval(['data_to_plot=' data_to_plot_names{pi} ';']);
%     figure
%     % plot mean rates/dF
%     for sti=1:length(estrus_states_titles)
%         subplot(length(estrus_states_titles),1,sti)
%         tmp(:)=nanmedian(data_to_plot(:,sti,:));
%         lh=bar(tmp); hold on
%         set(lh,'FaceColor', ALL_colors(color_ind(sti),:))
%         for idi=1:length(mouse_info)
%             tmp(:)=data_to_plot(idi,sti,:);
%             lh=plot(tmp,'*'); hold on
%             set(lh,'Color', [0.7 0.7 0.7])
%         end
%         title (estrus_states_titles{sti})
%         ylabel (YLABELs{pi})
%         ylim([0 YLIMs(pi)])
%     end
%     legend (estrus_states_titles)
% end
% 
% % arrange data to plot by receptive state
% receptive_ind=[1 ,2; 3 ,5; 6 6; 7 7; 8 8];
% receptive_states_titles={'NR','RE','OVX','OVX+PR','OVX+Esr'};
% [ALL_colors,rec_color_ind]=get_estrus_colors(receptive_states_titles);
% for pi=1:size(receptive_ind,1)
%     mean_dF_by_receptive(:,pi,:)=mean(mean_dF_by_state(:,receptive_ind(pi,1):receptive_ind(pi,2),:),2);
%     mean_rate_over_receptive(:,pi,:)=mean(mean_rate_over_estrus_samples(:,receptive_ind(pi,1):receptive_ind(pi,2),:),2);
% end
% 
% %% plot dF and rates by receptive 
% data_to_plot_names={'mean_dF_by_receptive','mean_rate_over_receptive'};
% YLIMs=[12 1.7]; YLABELs={'Mean dF (a.u.)' ,'Mean event rates (events/min)'};
% for pi=1:length(data_to_plot_names)
%     eval(['data_to_plot=' data_to_plot_names{pi} ';']);
%     figure
%     % plot mean rates/dF
%     for sti=1:length(receptive_states_titles)
%         subplot(length(receptive_states_titles),1,sti)
%         tmp(:)=nanmedian(data_to_plot(:,sti,:));
%         lh=bar(tmp); hold on
%         set(lh,'FaceColor', ALL_colors(rec_color_ind(sti),:))
%         for idi=1:length(mouse_info)
%             tmp(:)=data_to_plot(idi,sti,:);
%             lh=plot(tmp,'*'); hold on
%             set(lh,'Color', [0.7 0.7 0.7])
%         end
%         title (receptive_states_titles{sti})
%         ylabel (YLABELs{pi})
%         ylim([0 YLIMs(pi)])
%     end
%     legend (receptive_states_titles)
% end
% 
% 
% % statistics
% clear c_dF2 c_r2 c_dF22
% c_dF22=[]; c_r22=[];
% for k=[1:3,7:18]
%     figure; [p,tbl,stats] = kruskalwallis(mean_dF_by_receptive(:,:,k),[],'off');
%     c_dF2= multcompare(stats,'CType','hsd');% Tukey's honest significant difference criterion correction
%     tmp2=cat(2, c_dF2,c_dF2(:,6)<0.05);
%     c_dF22=[c_dF22; cat(2,tmp2,k*ones(size(c_dF2,1),1))];
%    clear tmp2
%     figure; [p,tbl,stats_r] = kruskalwallis(mean_rate_over_receptive(:,:,k),[],'off');
%     c_r2 = multcompare(stats_r,'CType','hsd');
%     tmp2=cat(2, c_r2,c_r2(:,6)<0.05);
%     c_r22=[c_r22; cat(2,tmp2,k*ones(size(c_r2,1),1))];
% %     figure; [p,tbl,stats_w] = kruskalwallis(med_width_dark_light_over_estrus_samples(:,:,k),[],'off');
% %     c_w = multcompare(stats_w,'CType','hsd');
% end
% 
% 
% %% plot specific window data
% % choose between estrous stages or receptive/non receptive
% clear lh1
% by_receptive=1;
% switch by_receptive
%     case 0
%         dF_to_plot=mean_dF_by_state;
%         rate_to_plot=mean_rate_over_estrus_samples;
%         titles_str=estrus_states_titles;
%         color_ind2=color_ind;
%     case 1
%         dF_to_plot=mean_dF_by_receptive;
%         rate_to_plot=mean_rate_over_receptive;
%         titles_str=receptive_states_titles;
%         color_ind2=rec_color_ind;
% end
% %width_to_plot=median_width_over_estrus_samples;
% clear med_dF_over_estrus_samples_window med_rate_over_estrus_samples_window mean_dF_over_estrus_samples_window mean_rate_over_estrus_samples_window c_r2
% % for paper I used window=i, i=2
% for i=[1,7:16]
%     window=[i:i+2];% 
%     med_dF_over_estrus_samples_window(:,:)=nanmedian(dF_to_plot(:,:,window),3);% ID, estrus, hour
%     med_rate_over_estrus_samples_window(:,:)=nanmedian(rate_to_plot(:,:,window),3);% ID, estrus, hour
%     %med_width_over_estrus_samples_window(:,:)=nanmedian(width_to_plot(:,:,window),3);% ID, estrus, hour
%     
%     mean_dF_over_estrus_samples_window(:,:)=nanmean(dF_to_plot(:,:,window),3);% ID, estrus, hour
%     mean_rate_over_estrus_samples_window(:,:)=nanmean(rate_to_plot(:,:,window),3);% ID, estrus, hour
%     %mean_width_over_estrus_samples_window(:,:)=nanmean(width_to_plot(:,:,window),3);% ID, estrus, hour
%     
%     %%% pay attaention here - median or mean?
%     %med_rate_over_estrus_samples_window=med_rate_over_estrus_samples_window([1,2,4:6],:);
%     %med_dF_over_estrus_samples_window=med_dF_over_estrus_samples_window([1,2,4:6],:);
%     figure
%     subplot(2,1,1)
%     %bar(nanmedian(mean_dF_over_estrus_samples_window)); hold on;
%     for sti=1:length(color_ind2)
%         bh=bar(sti, nanmedian(mean_dF_over_estrus_samples_window(:,sti))); hold on;
%         set(bh,'FaceColor', ALL_colors(color_ind2(sti),:))
%     end
%     for idi=1:size(med_dF_over_estrus_samples_window,1)
%         lh1=plot(med_dF_over_estrus_samples_window(idi,:),'*'); hold on;
%         set(lh1,'Color', [CF*idi CF*idi CF*idi])
%     end
%     xticks([1:length(color_ind2)])
%     xticklabels(titles_str)
%     ylim([0 12])
%     ylabel ('median/mean dF')
%     title(['window : ' num2str(window(1)) ' to ' num2str(window(end))])
%     %rates
%     subplot(2,1,2)
%     %bar(nanmedian(med_rate_over_estrus_samples_window)); hold on;
%     for sti=1:length(color_ind2)
%         bh=bar(sti, nanmedian(med_rate_over_estrus_samples_window(:,sti))); hold on;
%         set(bh,'FaceColor', ALL_colors(color_ind2(sti),:))
%     end
%     for idi=1:size(med_rate_over_estrus_samples_window,1)
%         lh1=plot(med_rate_over_estrus_samples_window(idi,:),'*'); hold on;
%         set(lh1,'Color', [CF*idi CF*idi CF*idi])
%     end
%     xticks([1:length(color_ind2)])
%     xticklabels(titles_str)
%     ylim([0 1.7])
%     ylabel ('median/mean rates')
%     title(['window : ' num2str(window(1)) ' to ' num2str(window(end))])
%     
%     
%     % statistics
%     %if i==2
%     figure; [p1,tbl,stats_r] = kruskalwallis(med_rate_over_estrus_samples_window,[],'off');
%     c_r2{i} = multcompare(stats_r,'CType','hsd');
%     %end
% end
% 1
% % compare time windows
% % window1=[1:3];
% % window2=[6:8];
% % window3=[10:12];
% % mean_rate_over_estrus_samples_window1(:,:)=nanmean(rate_over_estrus_samples_DL(:,:,window1),3);% ID, estrus, hour
% % mean_rate_over_estrus_samples_window2(:,:)=nanmean(rate_over_estrus_samples_DL(:,:,window2),3);% ID, estrus, hour
% % mean_rate_over_estrus_samples_window3(:,:)=nanmean(rate_over_estrus_samples_DL(:,:,window3),3);% ID, estrus, hour
% % figure
% % % window 1
% % subplot(3,1,1)
% % bar(nanmedian(mean_rate_over_estrus_samples_window1)); hold on;
% % for idi=1:size(mean_rate_over_estrus_samples_window1,1) 
% %     lh1=plot(mean_rate_over_estrus_samples_window1(idi,:),'-*'); hold on;
% %     set(lh1,'Color', [CF*idi CF*idi CF*idi])
% % end
% % xticklabels(estrus_states_titles)
% % ylabel ('median/mean rates')
% % ylim([0 0.8])
% % title(['window : ' num2str(window1(1)) ' to ' num2str(window1(end))])
% % %window 2
% % subplot(3,1,2)
% % bar(nanmedian(mean_rate_over_estrus_samples_window2)); hold on;
% % for idi=1:size(mean_rate_over_estrus_samples_window2,1) 
% %     lh1=plot(mean_rate_over_estrus_samples_window2(idi,:),'-*'); hold on;
% %     set(lh1,'Color', [CF*idi CF*idi CF*idi])
% % end
% % xticklabels(estrus_states_titles)
% % ylabel ('median/mean rates')
% % ylim([0 0.8])
% % title(['window : ' num2str(window2(1)) ' to ' num2str(window2(end))])
% % %window 3
% % subplot(3,1,3)
% % bar(nanmedian(mean_rate_over_estrus_samples_window3)); hold on;
% % for idi=1:size(mean_rate_over_estrus_samples_window3,1) 
% %     lh1=plot(mean_rate_over_estrus_samples_window3(idi,:),'-*'); hold on;
% %     set(lh1,'Color', [CF*idi CF*idi CF*idi])
% % end
% % xticklabels(estrus_states_titles)
% % ylabel ('median/mean rates')
% % ylim([0 0.8])
% % title(['window : ' num2str(window3(1)) ' to ' num2str(window3(end))])
% %%%
% % 
% % % plot rates
% % clear tmp
% % figure
% % 
% % for sti=1:length(estrus_states_titles)
% %     subplot(length(estrus_states_titles),1,sti)
% %     for idi=1:length(mouse_info)
% %         tmp(:)=rate_over_estrus_samples_DL(idi,sti,:);
% %         lh=plot(tmp); hold on
% %         set(lh,'Color', ALL_colors(color_ind(sti),:))
% %     end
% %     clear tmp
% %     ylim([0 0.75])
% %     title ('rates')
% % end
% % % plot width
% % figure
% % clear tmp
% % 
% % for sti=1:length(estrus_states_titles)
% %     subplot(length(estrus_states_titles),1,sti)
% %     for idi=1:length(mouse_info)
% %         tmp(:)=width_over_estrus_samples_DL(idi,sti,:);
% %         lh=plot(tmp); hold on
% %         set(lh,'Color', ALL_colors(color_ind(sti),:))
% %     end
% %     clear tmp
% %     ylim([0 30])
% %     title ('width')
% % end
% % 
% % 
% % 
% % 
% % figure
% % % plot mean rates
% % 
% % for sti=1:length(estrus_states_titles)
% %     subplot(length(estrus_states_titles),1,sti)
% %     tmp(:)=nanmedian(rate_over_estrus_samples_DL(:,sti,:));
% %     lh=bar(tmp); hold on
% %     set(lh,'FaceColor', ALL_colors(color_ind(sti),:))
% %     for idi=1:length(mouse_info)
% %         tmp(:)=rate_over_estrus_samples_DL(idi,sti,:);
% %         lh=plot(tmp,'*'); hold on
% %         set(lh,'Color', [0.7 0.7 0.7])
% %     end
% %     title (estrus_states_titles{sti})
% %     ylabel('median rates')
% %     ylim([0 0.75])
% % end
% % legend (estrus_states_titles)
% % 
% % % plot mean width
% % figure
% % subplot(2,1,2)
% % for sti=1:length(estrus_states_titles)
% %     subplot(length(estrus_states_titles),1,sti)
% %     tmp(:)=nanmedian(width_over_estrus_samples_DL(:,sti,:));
% %     lh=bar(tmp); hold on
% %     set(lh,'FaceColor', ALL_colors(color_ind(sti),:))
% %     for idi=1:length(mouse_info)
% %         tmp(:)=width_over_estrus_samples_DL(idi,sti,:);
% %         lh=plot(tmp,'*'); hold on
% %         set(lh,'Color', [0.7 0.7 0.7])
% %     end
% %     title (estrus_states_titles{sti})
% %     ylabel('median width')
% %     ylim([0 9])
% % end
% % 
% % 
% % % plot dF
% % figure
% % for sti=1:length(estrus_states_titles)
% %     subplot(length(estrus_states_titles),1,sti)
% %     for idi=1:length(mouse_info)
% %     tmp(:)=dF_by_state_DL(idi,sti,:);
% %     lh=plot(tmp); hold on
% %     set(lh,'Color', ALL_colors(color_ind(sti),:))
% %     end
% %     ylim([0 15])
% %     ylabel ('dF')
% %     title (estrus_states_titles{sti})
% % end
% % 
% % 
% % 
% % 
% % % plot mean dF
% % figure
% % for sti=1:length(estrus_states_titles)
% %     subplot(length(estrus_states_titles),1,sti)
% %     tmp(:)=nanmedian(dF_by_state_DL(:,sti,:));
% %     lh=bar(tmp); hold on
% %     set(lh,'FaceColor', ALL_colors(color_ind(sti),:))
% %     for idi=1:length(mouse_info)
% %         tmp(:)=dF_by_state_DL(idi,sti,:);
% %         lh=plot(tmp,'*'); hold on
% %         set(lh,'Color', [0.7 0.7 0.7])
% %     end
% %     ylim([-0.5 15])
% %    ylabel ('median dF')
% %     title (estrus_states_titles{sti})
% % end
% % 
% % %legend (estrus_states_titles)
% % 
% % 
% % figure
% % for idi=1:length(mouse_info)
% %     for hi=1:size(output{idi}.mean_dF_over_all_samples,1)
% %         plot(output{idi}.mean_dF_over_all_samples(hi,:)+25*hi); hold on
% %     end
% %     ylim([-20 (hi+3)*25])
% % end
% % title('all mice')
% % 
% % 
% % % plot mean dF per ID
% % CF=0.8/length(mouse_info);% color factor
% % figure
% % for sti=1:length(estrus_states_titles)
% %     subplot(length(estrus_states_titles),1,sti)
% %      for idi=1:length(mouse_info)
% %         tmp(:)=dF_by_state_DL(idi,sti,:);
% %         lh=plot(tmp,'-*'); hold on
% %         set(lh,'Color', [CF*idi CF*idi CF*idi])
% %     end
% %     ylim([-2 20])
% %    title ([ estrus_states_titles{sti} ', median dF'])
% % end
% % legend (estrus_states_titles)
% % 
% % 
% % for sti=1:length(estrus_states_titles)
% %     all_df_by_state(sti,:)=nanmedian(dF_by_state_DL(:,sti,:));
% %     %histcounts(med_all_df_by_state(sti,:),24);
% % end
% 
% 1
% 
