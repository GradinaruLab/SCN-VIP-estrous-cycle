function [output] = get_time_series_FP_per_mouse (mouse_info);
%% this function is used to analyse the time dependent time-series FP data, 10 minutes per hour
% Taken with SynapseTDT. per mouse 
% used in get_time_series_FP

close all
if nargin==0
    
  %  mouse_info.ID='198R';mouse_info.side='R';OVX=0;% female
 % mouse_info.ID='200LL';mouse_info.side='R';OVX=0;%% female
  mouse_info.ID='246RL';mouse_info.side='R';OVX=1;%% female
  %mouse_info.ID='247RRL';mouse_info.side='L';OVX=1;%% female % 
 %mouse_info.ID='259R'; mouse_info.side='R'; OVX=1;% female
 % mouse_info.ID='260L'; mouse_info.side='L';% OVX=1;%female
   %mouse_info.ID='261RL'; mouse_info.side='R';% OVX=1;%female
  %   mouse_info.ID='313RL'; mouse_info.side='R'; OVX=1;% female
     mouse_info.analysis_type='estrous_cycle';
    
 %    mouse_info.ID='247RRL_OVX';mouse_info.side='L';% OVX
    % mouse_info.ID='246RL_OVX'; mouse_info.side='R';% OVX
    % mouse_info.ID='261RL_OVX'; mouse_info.side='R';% OVX
    %  mouse_info.ID='259R_OVX'; mouse_info.side='R'; % OVX
    
    % mouse_info.ID='262R'; mouse_info.side='R';% male, SynTDT
    % mouse_info.ID='273RL'; mouse_info.side='R';% male
    % mouse_info.ID='286R'; mouse_info.side='R';% male
    % mouse_info.ID='287L'; mouse_info.side='R';% male to exlude?
    % mouse_info.ID='288RL'; mouse_info.side='R';OVX=0;% male
    % mouse_info.ID='296R'; mouse_info.side='R';% male
    % mouse_info.analysis_type='male_female';
    
    
    mouse_info.load_new_trials=1;
   % mouse_info.analysis_type='male_female';
 
    % mouse_info.analysis_type='estrous_cycle2';
    trial_info.show=1;% show each 24h trial
    show_fft=1;
else
     trial_info.show=0;% show each 24h trial 
      show_fft=0;
      % used for FFT analysis function 
      switch mouse_info.sex
          case 'Female'
              mouse_info.analysis_type='estrous_cycle';
          case 'Female2'
               mouse_info.analysis_type='estrous_cycle2';
          case {'OVX','Male'}
              mouse_info.analysis_type='male_female';
      end
end

switch mouse_info.ID
    case {'198R' ,'200LL','260L','262R'}
        OVX=0;
    case {'246RL','247RRL','259R','261RL','313RL'}
        OVX=1;
end

%% get experimental information:
my_path='D:\DATA_Glab\fiberphotometry\';
switch mouse_info.analysis_type
    case 'male_female'
        [NUM,TXT]=xlsread([my_path 'FP_VIP_GC_male_female2.xlsx']);
    case {'estrous_cycle','estrous_cycle2'}
        [NUM,TXT]=xlsread([my_path 'FP_VIP_GC4.xlsx']);
end
%% load analysis parameters
% par_file_name='ParametersFP';
% [NUMpar,TXTpar,RAWpar]=xlsread([my_path par_file_name '.xlsx']);
analysis_params.std_thresh=1.2; % was 1.5 until 07/28/22
trial_info.path='D:\DATA_Glab\fiberphotometry\';

 cc=0;
 FFT_cc=1;
%OVX=0;
 
All_IDS=TXT(2:end,1);
All_Genders=upper(TXT(2:end,2));
%All_constructs=upper(TXT(2:end,5));
All_genotype=TXT(2:end,5);
All_dates=get_dates(NUM);
All_SnameNnums=TXT(2:end,13); %Sess(LD) DL test6R onset
[All_Sess,All_Snames]=get_Snames(All_SnameNnums);
All_estrus_states=TXT(2:end,20);
All_estrus_states2=TXT(2:end,23);
All_ages=TXT(2:end,21);
All_rigs=TXT(2:end,22);
All_includes=NUM(1:end,14);
All_sides=TXT(2:end,14);
All_repeats_to_remove=TXT(2:end,24);


this_ID_ind=find(strcmp(All_IDS, ['VIPGC' mouse_info.ID]));
include_ind=find(All_includes==1);
sess_ind=intersect(this_ID_ind,include_ind);

switch mouse_info.analysis_type
    case 'male_female'
        this_filename=[my_path '\TDT_TimeSeries\' mouse_info.ID '_dF_male_female.mat'];
    case 'estrous_cycle'
        this_filename=[my_path 'TDT_TimeSeries\' mouse_info.ID '_dF_estrous.mat'];
    case 'estrous_cycle2'
        this_filename=[my_path 'TDT_TimeSeries\' mouse_info.ID '_dF_estrous2.mat'];
        All_estrus_states=All_estrus_states2;
end

% switch mouse_info.ID
%     case {'247RRL';'246RL';'261RL';'259R'};  OVX=1;
%     otherwise  OVX=0;
% end
        

%% load the relevant FP data.
if exist(this_filename) && mouse_info.load_new_trials==0
    load(this_filename)
elseif  ~exist([my_path '\TDT_TimeSeries\' mouse_info.ID '_dF.mat']) || mouse_info.load_new_trials==1
    % if file doesn't exist- load the original data
    %  data will be z-scored using
    % dF = dF - median(dF))./mad(baseline); % normalization using robust
    % z-score. baseline is the median of the recording taken during the dark
    % phase (11 repaets, ignoring the first hour)
    
   for i=1:length(sess_ind)
  %for i=1:12
        k=sess_ind(i);
        trial_info.date=All_dates{k};
        trial_info.sess_num=All_Sess(k);
        trial_info.estrus=All_estrus_states{k};
        trial_info.to_remove=str2num(All_repeats_to_remove{k});
        trial_info.rig=All_rigs{k};
        
       [df,t,analysis{i}]=get_time_series_FP_single_trial(mouse_info, trial_info,analysis_params);
        
        if i==1; dF1(i,:,:)=df; end
        if size(df,2)==size(dF1,3)
            dF1(i,:,:)=df;
            t2(i,:,:)=t;
        end
        if size(df,2)>size(dF1,3)
            dF1(i,:,:)=df(:,1:size(dF1,3));
            t2(i,:,:)=t(:,1:size(dF1,3));
        end
        if size(df,2)<size(dF1,3)
            dF1=dF1(:,:,1:size(df,2));
            t2=t2(:,:,1:size(df,2));
            dF1(i,:,:)=df;
            t2(i,:,:)=t;
        end
        
    end
    t1(:,:)=nanmean(t2,1);
    t=mean(t1,1);
    clear t2 t1
%     switch mouse_info.analysis_type
%         case 'male_female'
%             save([my_path '\TDT_TimeSeries\' mouse_info.ID '_dF_male_female'],'dF1','t','analysis')
%         case 'estrous_cycle'
%             save([my_path '\TDT_TimeSeries\' mouse_info.ID '_dF_estrous'],'dF1','t','analysis')
%     end
end

estrus_states=All_estrus_states(sess_ind);

switch mouse_info.analysis_type
    case 'male_female'
        for ei=1:length(estrus_states)
           if strfind(estrus_states{ei},'P'); estrus_states{ei}='Female'; end
        end
        estrus_states_titles={'Female','OVX','Male'};
    case 'estrous_cycle'
        estrus_states_titles={'P-2','P-1','P+0','P+1','P+2','OVX'};
        if ~OVX
            estrus_states_titles=estrus_states_titles(1:5);
        end
    case 'estrous_cycle2'
        estrus_states_titles={'P','E','M','D','OVX'};
        if ~OVX
            estrus_states_titles=estrus_states_titles(1:4);
        end
end
[ALL_colors,color_ind]=get_estrus_colors(estrus_states_titles);
[G,ID]=findgroups(estrus_states);% the order is shown in ID
% make an array of the order of the sessions, based on the estrous cycle
order=[];
new_estrus_list=[];
for idi=1:length(ID)
    order=[order find(G==idi)'];
    new_estrus_list=[new_estrus_list estrus_states(find(G==idi))'];
end
[G2,~]=findgroups(new_estrus_list);


% after fft was included in the analysis ;Feb 2022
if FFT_cc
    % the minimum frequency to sum on is 0.0033Hz, which is 5 minutes
    % (1/5*60), which is half cycle in 10 minutes
     newf='_newF';
    %newf=[];
    switch newf
        case isempty(newf)
            new_f_limits=[0.0033 0.03; 0.03 0.1; 0.1 0.35; 0.35 0.65; 0.65 1.0; 1.0 1.35 ; 1.35 1.65; 1.65 2.3; 2.3 4];
        case '_newF'
            new_f_limits=[0.0033 0.007; 0.007 0.05; 0.05 0.1; 0.1 0.25; 0.25 0.45; 0.45 1.0; 1.0 1.35];
    end
     new_f_limits=flip(new_f_limits,1);
     %  f_limits=[0.0033 0.007 ;0.007 0.05; 0.05 0.25; 0.25 0.45; 0.45 2];     
%     switch mouse_info.analysis_type
%         case 'estrous_cycle'
            for fi=1:size(new_f_limits,1)
                eval(['all_fft_freq_range_array' num2str(fi) '=[];']);
            end
            for fi=1:size(new_f_limits,1)
                for i=1:length(analysis) % all sessions
                    f_limits=analysis{i}.int_fft_limits;
                    eval(['all_fft_freq_range_' num2str(fi) '(i,:)=analysis{i}.int_fft(' num2str(fi) ',:);']); % 0 to 0.03 Hz lower frequencies
                    eval(['all_fft_freq_range_array' num2str(fi) '=[all_fft_freq_range_array' num2str(fi) ' analysis{i}.int_fft(' num2str(fi) ',:)];']);
                end
            end
            
            %plot freqyency dependancy 
            if show_fft
%                 figure
%                 for i=1:size(all_fft_freq_range_1,1)
%                     
%                     ph{i}=plot((i-1)*24+[1:24],all_fft_freq_range_1(i,:));hold on
%                     if ~isempty(estrus_states{i})
%                         this_estrus{1}=estrus_states{i};
%                     end
%                 end
%                
                
                figure
                %plot(nanmedian(all_fft_freq_range_1,1),nanmedian(all_fft_freq_range_2,1),'-*'); hold on
                % plot(nanmedian(all_fft_freq_range_1,1),nanmedian(all_fft_freq_range_3,1),'-*'); hold on           %             plot(nanmedian(all_fft_freq_range_3,1),nanmedian(all_fft_freq_range_2,1),'-*'); hold on
                 plot(nanmedian(all_fft_freq_range_3,1),nanmedian(all_fft_freq_range_2,1),'-*'); hold on
                for i=1:size(all_fft_freq_range_2,1);
                    % for i=1:10
                    plot(all_fft_freq_range_3(i,:),all_fft_freq_range_2(i,:),'-*'); hold on
                end
                xlabel(['int. power fft; ' num2str(f_limits(3,:)) ' Hz'])
                ylabel(['int. power fft; ' num2str(f_limits(2,:)) ' Hz'])
                
            end
            
            % plot the power fft
            % first define the legend 
               switch mouse_info.analysis_type
                   case {'estrous_cycle', 'estrous_cycle2'}
                       ES=unique(estrus_states);ES=ES(~cellfun('isempty', ES));% remove OVX if doesn't exist 
                       for esi=1:length(ES)
                           if ~isempty(find(strcmp(estrus_states,estrus_states_titles{esi})))
                               legend_estrus_inds(esi)=min(find(strcmp(estrus_states,estrus_states_titles{esi})));
                           else
                               legend_estrus_inds(esi)=[];
                           end
                       end
                   case 'male_female'
                       if ~isempty(find(strcmp(estrus_states,estrus_states_titles{1})))
                           legend_estrus_inds=min(find(strcmp(estrus_states,estrus_states_titles{1})));
                       else
                           legend_estrus_inds=[];
                       end
               end

            %data_to_plot=all_fft_freq_range_1; range_title='0-0.03 Hz';YLIM_Val=[0 50];
            if  show_fft
                data_to_plot=all_fft_freq_range_2; range_title=[ f_limits(2,:) ' Hz']; YLIM_Val=[2 80];
                figure
                legend_included=[];
                for i=1:size(data_to_plot,1)
                    
                    ph{i}=plot((i-1)*24+[1:24],data_to_plot(i,:));hold on
                    if ~isempty(estrus_states{i})
                        this_estrus{1}=estrus_states{i};
                        [ALL_colors_fft,color_ind2]=get_estrus_colors(this_estrus);
                    else
                        ALL_colors_fft=[0 0 0]; color_ind2=1;
                    end
                    ph{i}.Color=ALL_colors(color_ind2,:);
                end
                legend_included=[legend_included ph{legend_estrus_inds}];
                legend(legend_included,estrus_states_titles(1:length(legend_estrus_inds)))
                title([mouse_info.ID ' ' range_title ' fft'])
                ylabel('int. power fft')
                xlabel('Time (hour)')
                ylim(YLIM_Val)
            end
           
            % calcualte Time (time intervals)correlation and plot it
            ac_estrus_inds=[];
            ac_OVX_inds=[];
             estrus_inds=[];
            switch mouse_info.analysis_type
                case 'estrous_cycle'
                    ac_estrus_inds=find(strcmp(estrus_states,'P+0'));
                    % find the indexes of full estrous cycle
                    estrus_inds(:,1)=find(strcmp(estrus_states,'P-2'));
                    estrus_inds(:,2)=estrus_inds(:,1)+3;
                case 'estrous_cycle2'
                    ac_estrus_inds=find(strcmp(estrus_states,'P'));
                    % find the indexes of full estrous cycle
                    estrus_inds(:,1)=find(strcmp(estrus_states,'P'))-2;
                    estrus_inds(:,2)=estrus_inds(:,1)+4;
            end
            ac_OVX_inds=find(strcmp(estrus_states,'OVX'));
            if ~isempty(find(strcmp(estrus_states,'OVX')))
                ovxi=find(strcmp(estrus_states,'OVX'));
                estrus_inds=[estrus_inds; [ovxi(1) ovxi(end)]];
            end
            
            
            if show_fft
               % data_str_to_plot= {'all_fft_freq_range_array1' 'all_fft_freq_range_array2' 'all_fft_freq_range_array3'};
               for fi=1:size(new_f_limits,1)
                   data_str_to_plot{fi}= ['all_fft_freq_range_array' num2str(fi)];
               end
                    
                figure
                for pi=1:length(data_str_to_plot)
                    eval(['data_to_plot=' data_str_to_plot{pi} ';']);
                    bh=subplot(length(data_str_to_plot),1,pi);
                    autocorr(double(data_to_plot),'NumLags',length(data_to_plot)-round(length(data_to_plot)*0.12));hold on;
                    if ~isempty(ac_estrus_inds)
                        ph=line([ac_estrus_inds*24 ac_estrus_inds*24],[0 1]);
                        for i=1:length(ph); ph(i).Color='r'; end
                    end
                    if ~isempty(ac_OVX_inds)
                        ph=line([ac_OVX_inds*24 ac_OVX_inds*24],[0 1]);
                        for i=1:length(ph); ph(i).Color='k'; end
                    end
                    numbers=(get(bh, 'XTick')/24);
                    for ni=1:length(numbers); num_str{ni}=num2str(numbers(ni),'%0.1f');end
                    set(bh,'XTickLabel',num_str)
                    xlabel('Lag (Days)') % switched to days
                    title([mouse_info.ID  ' ' num2str(new_f_limits(pi,:)) ' Hz fft'])
                    xlim([0 30*24])
                    ylim([-0.6 0.6])
%                     
%                     xlim([0 8*24])
%                     ylim([-0.1 0.1])
                end
               
                numbers=(get(bh, 'XTick')/24);
                for ni=1:length(numbers); num_str{ni}=num2str(numbers(ni),'%0.1f');end
                set(bh,'XTickLabel',num_str)
            end
            
            
            % creates an FFT matrix of power
            for i=1:length(analysis) % all sessions
                B1=analysis{i}.fft_power;
                f=analysis{i}.freq;
                   
                for wi=1:size(new_f_limits,1)
                    for di=1:size(B1,2)
                        f_inds=intersect(find(f>new_f_limits(wi,1)),find(f<new_f_limits(wi,2)));
                        if ~isempty(f_inds)
                            B1_int(wi,di)=sum(B1(f_inds,di));
                        else
                            B1_int(wi,di)=nan;
                        end
                    end
                end
                FFT_POWER_INT_by_freq{i}=B1_int;
            end
            % shift FFT to light - dark 
            tmp_FFT{1}=cat(2,FFT_POWER_INT_by_freq{1}(:,17:24),FFT_POWER_INT_by_freq{1}(:,1:16));
            for si=2:length(analysis) % all sessions
                tmp_FFT{si}=cat(2,FFT_POWER_INT_by_freq{si-1}(:,17:24),FFT_POWER_INT_by_freq{si}(:,1:16));
               %FFT_POWER_INT_by_freq{si}=;
            end  
            
            if show_fft
                % plot the fft power integrals
                freq_inds=find(new_f_limits(:,2)<=1.35);
                clims=[0 60];
                figure
                L=length(tmp_FFT) ;St=1; k=0;
               % L=10; St=10; k=0;
                for i=St:St+L-1% all sessions
                    k=k+1;
                    subplot(5,ceil(L/5),k)
                    %image( FFT_POWER_INT_by_freq{i},'CDataMapping','scaled')
                    imagesc(tmp_FFT{i}(freq_inds,:),clims)
                    colorbar
                    title([mouse_info.ID '  ' estrus_states{i} ])
                    yticks([1: size(tmp_FFT{i},1)])
                    yticklabels(num2str(new_f_limits(freq_inds,:)))
                    xlabel('Time (hours)')
                    ylabel('Int. Power fft')
                end
            end
            
  %  end
end
for i=1:length(estrus_states_titles)
    estrus_states_ind{i}=find(strcmp(estrus_states,estrus_states_titles{i}));
end

clear mean_over_all_samples mean_dF_over_estrus_samples rate_over_estrus_samples mean_dF_over_hour width_over_estrus_samples mean_rate_over_estrus_samples mean_width_over_estrus_samples
mean_dF_over_all_samples(:,:)=nanmean(dF1,1);
median_dF_over_all_samples(:,:)=nanmedian(dF1,1);
mean_dF_over_hour(:,:)=nanmean(dF1,3);
median_dF_over_hour(:,:)=nanmedian(dF1,3);
if trial_info.show
    figure
    subplot(2,1,1)
    title (mouse_info.ID)
    for hi=1:size(mean_dF_over_all_samples,1)
        %plot(mean_dF_over_all_samples(hi,:)+hi*10); hold on
      %  plot(mean_dF_over_all_samples(hi,:)); hold on
       plot(median_dF_over_all_samples(hi,:)); hold on
    end
    ylabel('all dF over hours')
    subplot(2,1,2)
     
    %    plot([1:24],mean_dF_over_hour'); hold on
    plot([1:24],median_dF_over_hour'); hold on
    ylabel('median dF over hours')
    xlabel (mouse_info.ID)
end

switch mouse_info.analysis_type
    case {'estrous_cycle','estrous_cycle2'}
    for sti=1:length(estrus_states_ind)
        mean_dF_over_estrus_samples(sti,:,:)=nanmean(dF1(estrus_states_ind{sti},:,:),1);
        median_dF_over_estrus_samples(sti,:,:)=nanmedian(dF1(estrus_states_ind{sti},:,:),1);
        k=0;
        %  rate_over_estrus_samples(sti,1,:)=nan;
        % width_over_estrus_samples(sti,1,:)=nan;
        for st2i=1:length(estrus_states_ind{sti})
            ind2=estrus_states_ind{sti}(st2i);
            k=k+1;
            rate_over_estrus_samples(sti,k,:)=analysis{ind2}.rate;
            width_over_estrus_samples(sti,k,:)=analysis{ind2}.med_width;
        end
        if size(rate_over_estrus_samples,1)<sti  % for females that were not OVX
            mean_rate_over_estrus_samples(sti,:)=nan;
            mean_width_over_estrus_samples(sti,:)=nan;
            median_rate_over_estrus_samples(sti,:)=nan;
           median_width_over_estrus_samples(sti,:)=nan;
        else
            mean_rate_over_estrus_samples(sti,:)=nanmean(rate_over_estrus_samples(sti,:,:),2);
            mean_width_over_estrus_samples(sti,:)=nanmean(width_over_estrus_samples(sti,:,:),2);
            median_rate_over_estrus_samples(sti,:)=nanmedian(rate_over_estrus_samples(sti,:,:),2);
            median_width_over_estrus_samples(sti,:)=nanmedian(width_over_estrus_samples(sti,:,:),2);
        end
    end
    if trial_info.show
        figure
        for sti=1:length(estrus_states_ind)
            subplot(1,length(estrus_states_ind),sti)
            temp_dF(:,:)=mean_dF_over_estrus_samples(sti,:,:);
            for hi=1:size(mean_dF_over_all_samples,1)
                plot(temp_dF(hi,:)+10*hi); hold on
            end
            clear temp_dF
        end
    end
    case 'male_female'
        sti=find(~cellfun(@isempty,estrus_states_ind));
       mean_dF_over_estrus_samples(:,:)=nanmean(dF1(estrus_states_ind{sti},:,:),1);
        median_dF_over_estrus_samples(:,:)=nanmedian(dF1(estrus_states_ind{sti},:,:),1);
        k=0;   
    
    for st2i=1:length(estrus_states_ind{sti})
        ind2=estrus_states_ind{sti}(st2i);
        k=k+1;
        rate_over_estrus_samples(k,:)=analysis{ind2}.rate;
        width_over_estrus_samples(k,:)=analysis{ind2}.med_width;
    end
    mean_rate_over_estrus_samples(:)=nanmean(rate_over_estrus_samples(:,:),1);
    mean_width_over_estrus_samples(:)=nanmean(width_over_estrus_samples(:,:),1);
    median_rate_over_estrus_samples(:)=nanmedian(rate_over_estrus_samples(:,:),1);
    median_width_over_estrus_samples(:)=nanmedian(width_over_estrus_samples(:,:),1);
 
end
  

switch mouse_info.analysis_type
    case {'estrous_cycle' , 'estrous_cycle2'}
        for sti=1:length(estrus_states_ind)
            mean_dF_by_state(sti,:)=nanmean(mean_dF_over_hour(estrus_states_ind{sti},:),1);
            median_dF_by_state(sti,:)=nanmedian(mean_dF_over_hour(estrus_states_ind{sti},:),1);
        end
%         if trial_info.show
%             figure
%             plot(mean_dF_by_state')
%             title([ mouse_info.ID ' mean dF by state'])
%             xlabel('Time (hours)')
%         end
    case 'male_female'
        mean_dF_by_state(:)=nanmean(mean_dF_over_hour(estrus_states_ind{sti},:),1);
         median_dF_by_state(:)=nanmedian(mean_dF_over_hour(estrus_states_ind{sti},:),1);
end

%% calculates medain of dark/light
switch mouse_info.analysis_type
    case 'male_female'
        full_time_estrus_states_titles={'Female D','Female L','OVX D','OVX L','Male D','Male L'};
    case 'estrous_cycle'
        full_time_estrus_states_titles={'P-2 D','P-1 L','P-1 D','P+0 L','P+0 D','P+1 L','P+1 D','P+2 L','P+2 D','OVX D','OVX L'};
     case 'estrous_cycle2'
        full_time_estrus_states_titles={'P D','P L','E D','E L','M D','M L','Di D','Di L','OVX D','OVX L'};

end

%% gets the value of each parameter for the total dark or light phase 
% 05/12/22- change to take median instead of mean 
% 11/20/22- change to mean 
clear tmp tmp2 tmp3
tmp=mean_dF_by_state;%  estrus, hour
%tmp=median_dF_by_state;%  estrus, hour
full_time_dF=reshape(tmp',1,size(tmp,1)*size(tmp,2));
tmp2=mean_rate_over_estrus_samples;%  estrus, hour
%tmp2=median_rate_over_estrus_samples;%  
full_time_rate=reshape(tmp2',1,size(tmp2,1)*size(tmp2,2));
tmp3=mean_width_over_estrus_samples;%  estrus, hour
%tmp3=median_width_over_estrus_samples;%  estrus, hour
full_time_width=reshape(tmp3',1,size(tmp3,1)*size(tmp3,2));

clear dF_dark_light rate_dark_light width_dark_light
dF_dark_light=[];
rate_dark_light=[];
width_dark_light=[];

switch mouse_info.analysis_type
    case {'male_female' }
        light_off=[5:17];
        light_on=[1:4,18:24];
        
        dF_dark_light=[nanmedian(full_time_dF(light_off)) nanmedian(full_time_dF(light_on))];
        rate_dark_light=[nanmedian(full_time_rate(light_off)) nanmedian(full_time_rate(light_on))];
        width_dark_light=[nanmedian(full_time_width(light_off)) nanmedian(full_time_width(light_on))];
        
    case {'estrous_cycle','estrous_cycle2'}
        % for estous cycle 
        switch mouse_info.analysis_type
            case 'estrous_cycle'
                if OVX; L2=5+24; else L2=5; end% remove the first 5 hours, but also the last 24 hours, if OVX is included
                ind_end=length(full_time_dF)-L2;%
                light_off=[5:24:ind_end];% first 4 hours are light
                light_on=[5+12:24:ind_end];
                %light_off=24+[5:24:ind_end];% first 4 hours are light
                %light_on=24+[5+12:24:ind_end];
                light_switch=sort([light_off, light_on]);
                
                for ti=1:length(light_switch)-1% start with dark
                    dF_dark_light(ti)=nanmedian(full_time_dF(light_switch(ti):light_switch(ti+1)));
                    rate_dark_light(ti)=nanmedian(full_time_rate(light_switch(ti):light_switch(ti+1)));
                    width_dark_light(ti)=nanmedian(full_time_width(light_switch(ti):light_switch(ti+1)));
                end
            case 'estrous_cycle2'
                ind_end=length(full_time_dF)-5-24;% 
                light_off=[5:24:ind_end];% first 4 hours are light
                light_on=[5+12:24:ind_end];
                 for ti=1:length(light_off) % shift each trial 
                    light_off_ind=[light_off(ti):light_on(ti)];
                    light_on_ind=[light_off(ti)-4:light_off(ti)-1,light_on(ti)+1:light_on(ti)+7];
                    dF_dark_light=[dF_dark_light nanmedian(full_time_dF(light_off_ind)) nanmedian(full_time_dF(light_on_ind))];
                    rate_dark_light=[rate_dark_light nanmedian(full_time_rate(light_off_ind)) nanmedian(full_time_rate(light_on_ind))];
                    width_dark_light=[width_dark_light nanmedian(full_time_width(light_off_ind)) nanmedian(full_time_width(light_on_ind))];
                 end

        end

        % for OVX- the first 24 indexes comes from OVX 
        if OVX
            L_OVX=length(full_time_dF);
            OVX_light_off=L_OVX-24+[5:16];
            OVX_light_on=L_OVX-24+[1:4,17:24];
            
            OVX_dF_dark_light=[ nanmedian(full_time_dF(OVX_light_on)) nanmedian(full_time_dF(OVX_light_off))];
            OVX_rate_dark_light=[ nanmedian(full_time_rate(OVX_light_on)) nanmedian(full_time_rate(OVX_light_off))];
            OVX_width_dark_light=[ nanmedian(full_time_width(OVX_light_on)) nanmedian(full_time_width(OVX_light_off))];
            
            dF_dark_light=[ dF_dark_light OVX_dF_dark_light];
            rate_dark_light=[ rate_dark_light OVX_rate_dark_light];
            width_dark_light=[ width_dark_light OVX_width_dark_light];

        end
end




%% now plot
% 
% figure
% for hi=1:size(mean_dF_over_all_samples,1)
%     plot(t,mean_dF_over_all_samples(hi,:)+25*hi); hold on
% end
% ylim([-20 (hi+3)*25])
% ylabel('dF')
% title(['VIPGC' mouse_info.ID ])
step=15;
switch mouse_info.analysis_type
    case 'male_female'
        figure
        for hi=1:size(mean_dF_over_estrus_samples,1)
            %         tmp0(:,:)=dF1(estrus_states_ind{sti},hi,:);
            %         ph0=plot(tmp0); hold on
            %         set(ph0,'Color',[0.8 0.8 0.8])
            %tmp1(:)=mean_dF_over_estrus_samples(hi,:);
             tmp1(:)=median_dF_over_estrus_samples(hi,:);
            ph=plot(t,tmp1+step*hi); hold on
            set(ph,'Color',ALL_colors(color_ind(1),:))
        end
        ylim([-20 (hi+3)*step])
        title(['VIPGC' mouse_info.ID ' ' estrus_states_titles{1}])
        ylabel('dF')
    case {'estrous_cycle' ,'estrous_cycle2'}
    figure
    estrus_states_ind=estrus_states_ind(~cellfun('isempty',estrus_states_ind));
    for sti=1:length(estrus_states_ind)
        subplot(1,length(estrus_states_ind),sti)
        for hi=1:size(mean_dF_over_estrus_samples,2)
            %         tmp0(:,:)=dF1(estrus_states_ind{sti},hi,:);
            %         ph0=plot(tmp0); hold on
            %         set(ph0,'Color',[0.8 0.8 0.8])
            %tmp1(:)=mean_dF_over_estrus_samples(sti,hi,:);
            tmp1(:)=median_dF_over_estrus_samples(sti,hi,:);
            
            ph=plot(t,tmp1+step*hi); hold on
            set(ph,'Color',ALL_colors(color_ind(sti),:))
        end
        ylim([-20 (hi+3)*step])
        title(['VIPGC' mouse_info.ID ' ' estrus_states_titles{sti}])
        ylabel('dF')
    end
end
% plot dF over 10 minutes by state
switch mouse_info.analysis_type
    case 'male_female'
        figure
        sti=1;
        subplot(1,3,1)
        %lh=plot(mean_dF_over_hour(estrus_states_ind{sti},:)','o'); hold on
        lh=plot(median_dF_over_hour(estrus_states_ind{sti},:)','o'); hold on
        set(lh,'Color',ALL_colors(color_ind(sti),:))
        %lh2=plot( mean_dF_by_state(:)); hold on
         lh2=plot( median_dF_by_state(:)); hold on
        set(lh2,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        legend (estrus_states_titles)
        ylabel('dF')
        title(['VIPGC' mouse_info.ID ])
        
        % plot rates
        subplot(1,3,2)
        
       % ph=plot(mean_rate_over_estrus_samples(:));hold on
        ph=plot(median_rate_over_estrus_samples(:));hold on
        set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        legend (estrus_states_titles)
        ylabel('rates')
        title(['VIPGC' mouse_info.ID ])
        
        % plot width
        subplot(1,3,3)
       % ph=plot(mean_width_over_estrus_samples(:));hold on
        ph=plot(median_width_over_estrus_samples(:));hold on
        set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        legend (estrus_states_titles)
        ylabel('width')
        title(['VIPGC' mouse_info.ID ])
        
    case {'estrous_cycle','estrous_cycle2'}
        figure
        subplot(1,3,1)
        for sti=1:length(estrus_states_ind)
            
            %lh=plot(mean_dF_over_hour(estrus_states_ind{sti},:)','o'); hold on
            lh=plot(median_dF_over_hour(estrus_states_ind{sti},:)','o'); hold on
            set(lh,'Color',ALL_colors(color_ind(sti),:))
            %lh2=plot( mean_dF_by_state(sti,:)); hold on
            lh2=plot( median_dF_by_state(sti,:)); hold on
            set(lh2,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        end
        legend (estrus_states_titles)
        ylabel('dF')
        title(['VIPGC' mouse_info.ID ])
        
        % plot rates
        subplot(1,3,2)
        for sti=1:length(estrus_states_ind)
            %ph=plot(mean_rate_over_estrus_samples(sti,:));hold on
            ph=plot(median_rate_over_estrus_samples(sti,:));hold on
            set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        end
        legend (estrus_states_titles)
        ylabel('rates')
        title(['VIPGC' mouse_info.ID ])
        
        % plot width
        subplot(1,3,3)
        for sti=1:length(estrus_states_ind)
            %ph=plot(mean_width_over_estrus_samples(sti,:));hold on
            ph=plot(median_width_over_estrus_samples(sti,:));hold on
            set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        end
        legend (estrus_states_titles)
        ylabel('width')
        title(['VIPGC' mouse_info.ID ])
end

output.t=t;
%if ~isempty(estrus_states_ind{1})
output.mean_dF_over_estrus_samples=mean_dF_over_estrus_samples;
output.mean_rate_over_estrus_samples=mean_rate_over_estrus_samples;
output.mean_width_over_estrus_samples=mean_width_over_estrus_samples;
output.median_dF_over_estrus_samples=median_dF_over_estrus_samples;
output.median_rate_over_estrus_samples=median_rate_over_estrus_samples;
output.median_width_over_estrus_samples=median_width_over_estrus_samples;
%else
%   output.mean_dF_over_estrus_samples=[];
%  output.mean_rate_over_estrus_samples=[];
%  output.mean_width_over_estrus_samples=[];
%end
output.mean_dF_over_all_samples=mean_dF_over_all_samples;
output.median_dF_over_all_samples=median_dF_over_all_samples;
output.mean_dF_by_state=mean_dF_by_state;
output.median_dF_by_state=median_dF_by_state;
output.median_dF_dark_light=dF_dark_light;
output.median_rate_dark_light=rate_dark_light;
output.median_width_dark_light=width_dark_light;
if FFT_cc
    output.FFT_POWER_INT_by_freq=FFT_POWER_INT_by_freq;
end
output.estrus_states=estrus_states;
output.new_f_limits=new_f_limits;

switch mouse_info.analysis_type
    case 'male_female'
        save([my_path 'time_series_output' newf '_male_female_' mouse_info.ID], 'output')
         save([my_path 'time_series_output' newf '_general_' mouse_info.ID], 'output')
    case 'estrous_cycle'
        save([my_path 'time_series_output' newf '_estrous_cycle_' mouse_info.ID], 'output')
         save([my_path 'time_series_output' newf '_general_' mouse_info.ID], 'output')
      case 'estrous_cycle2'
        save([my_path 'time_series_output' newf '_estrous_cycle2_' mouse_info.ID], 'output')    
         save([my_path 'time_series_output' newf '_general2_' mouse_info.ID], 'output')
end

save([my_path 'mean_outputs\time_series_output' newf '_general_' mouse_info.ID], 'output')

 disp (['done ' mouse_info.ID])