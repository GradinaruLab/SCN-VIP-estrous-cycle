function [output] = get_LDtransition_FP_per_mouse (mouse_info);
%% this function is used to analyse the time dependent time-series FP data, 10 minutes per hour
% Taken with SynapseTDT. per mouse 
% used in XXXX
%%%%%%need to add. OVX+ESr/PR 
close all
if nargin==0
    %females:
    % mouse_info.ID='68RL';mouse_info.side='L';OVX=0;mouse_info.sex='Female';% female. Has enough repeats?
    %mouse_info.ID='93N';mouse_info.side='R';OVX=0;mouse_info.sex='Female';% female. Has enough repeats?
    %mouse_info.ID='108L';mouse_info.side='R';OVX=0;mouse_info.sex='Female';% female
    %  mouse_info.ID='110LL';mouse_info.side='L';OVX=0;mouse_info.sex='Female';% female
    
    %mouse_info.ID='107R';mouse_info.side='R';OVX=1;mouse_info.sex='Female';% female. also has hormonal manipulation, but don't have estrous cycle
   % mouse_info.ID='115L';mouse_info.side='R';OVX=1;mouse_info.sex='Female';% female also has hormonal manipulation
    % mouse_info.ID='A116R';mouse_info.side='L';OVX=1;mouse_info.sex='Female';% female. also has hormonal manipulation
    %mouse_info.ID='119LL';mouse_info.side='R';OVX=1;mouse_info.sex='Female';% female.also has hormonal manipulation
    % mouse_info.ID='122R';mouse_info.side='R';OVX=1;mouse_info.sex='Female';% female. also has hormonal manipulation
    %mouse_info.ID='123L';mouse_info.side='R';OVX=1;mouse_info.sex='Female';% female. also has hormonal manipulation. problematic fs
    % mouse_info.ID='128R';mouse_info.side='R';OVX=1;mouse_info.sex='Female';% female. also has hormonal manipulation
    
   %mouse_info.ID='161RR';mouse_info.side='R';OVX=0;mouse_info.sex='Female';% female.dones't have enough repeats. 
  %  mouse_info.ID='166R';mouse_info.side='R';OVX=0;mouse_info.sex='Female';% female.
   % mouse_info.ID='175L';mouse_info.side='L';OVX=0;mouse_info.sex='Female';% female
    %mouse_info.ID='176RL';mouse_info.side='L';OVX=0;mouse_info.sex='Female';% female
     mouse_info.ID='198L';mouse_info.side='R';OVX=0;mouse_info.sex='Female';% female
   % mouse_info.ID='200RR';mouse_info.side='R';OVX=0;mouse_info.sex='Female';%% female
    
   % males
%  mouse_info.ID='25L';mouse_info.side='L';OVX=0; mouse_info.sex='Male';% male
 %  mouse_info.ID='60N';mouse_info.side='R';OVX=0; mouse_info.sex='Male';% male. has old sessions
 %mouse_info.ID='62L';mouse_info.side='R';OVX=0; mouse_info.sex='Male';% male. has old sessions
  %mouse_info.ID='103R';mouse_info.side='R';OVX=0;mouse_info.sex='Male'; % male
  % mouse_info.ID='106LL';mouse_info.side='R';OVX=0; mouse_info.sex='Male';% male
   % mouse_info.ID='113L';mouse_info.side='R';OVX=0;mouse_info.sex='Male'; % male only has DL 
   
   % GFP
   % mouse_info.ID='GFP6N';mouse_info.side='R';OVX=0;mouse_info.sex='Female'; % female
   % mouse_info.ID='GFP7R';mouse_info.side='R';OVX=0;mouse_info.sex='Female'; % female
   % mouse_info.ID='GFP8N';mouse_info.side='R';OVX=0;mouse_info.sex='Male'; % male
   % mouse_info.ID='GFP9R';mouse_info.side='R';OVX=0;mouse_info.sex='Male'; % male
   mouse_info.minimum_trials=2;
    mouse_info.load_new_trials=1;
    %  mouse_info.analysis_type='male_female';
   % mouse_info.analysis_type='estrous_cycle';
    
    trial_info.show=1;% show each 24h trial
    show_fft=1;
else
     trial_info.show=0;% show each 24h trial 
      show_fft=0;
end
% used for FFT analysis function
switch mouse_info.sex
    case 'Female'
        mouse_info.analysis_type='estrous_cycle';
    case {'OVX','Male','male'}
        mouse_info.analysis_type='male_female';
end

switch mouse_info.ID  
    case {'107R', '115L','A116R','119LL','122R','123L','128R'}
        OVX=1;
    otherwise
        OVX=0;
end

%% get experimental information:
my_path='D:\DATA_Glab\fiberphotometry\';
T=readtable([my_path 'fiberphotometryVIPGC3_Aug2022.xlsx']);

%% load analysis parameters
analysis_params.minimum_trials=mouse_info.minimum_trials;
analysis_params.n_windows=15;
analysis_params.std_thresh=0.4; % 
analysis_params.F=1;
newF='newF'
trial_info.path='D:\DATA_Glab\fiberphotometry\';

%cc=0;
FFT_cc=1;
 
All_IDS=T.ID;
All_Genders=T.gender;
%All_genotype=T.Virus_cross;
All_dates=get_dates_table(T.month,T.day,T.year);
All_SnameNnums=T.processedFileName; %Sess(LD) DL test6R onset
[All_Sess,All_Snames]=get_Snames(All_SnameNnums);
All_estrus_states=T.estrus;
All_estrus_states2=T.estrusStates;
%All_ages=T.age;
All_rigs=T.rig;
All_includes=T.include;
%All_sides=T.side;
clear T


sess_ind_tmp=intersect(find(strcmp(All_IDS, ['VIPGC' mouse_info.ID])),find(All_includes==1));
sess_ind=intersect(sess_ind_tmp,find(strcmp(All_Snames,'Sess')));
onset_ind=intersect(sess_ind_tmp,find(strcmp(All_Snames,'SessOnset')));

this_filename=[my_path 'LDtransition\output_' mouse_info.analysis_type '_' newF mouse_info.ID '_MT' num2str(analysis_params.minimum_trials) '.mat'];

% switch mouse_info.ID
%     case {'247RRL';'246RL';'261RL';'259R'};  OVX=1;
%     otherwise  OVX=0;
% end  

%% load the relevant FP data.
if exist(this_filename) && mouse_info.load_new_trials==0
    load(this_filename)
    analysis=output.analysis;
    dF1=output.dF1;
    t=output.t;
elseif  ~exist([my_path '\LDtransition\' mouse_info.ID '_dF_' mouse_info.analysis_type '.mat' ]) || mouse_info.load_new_trials==1
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
        %trial_info.to_remove=str2num(All_repeats_to_remove{k});
        trial_info.rig=All_rigs{k};
        if ~isempty(onset_ind)
            trial_info.onset_sess_num=All_Sess(k);
        else
            trial_info.onset_sess_num=[];
        end
       [df,t,analysis{i}]=get_LDtransition_FP_single_trial_v2(mouse_info, trial_info,analysis_params);
        
        if i==1; dF1(i,:,:)=df; end % dF dimensions are : session, time-frame, time
        if size(df,2)==size(dF1,3)
            dF1(i,:,:)=df;
            t2(i,:,:)=t;
        end
        % size is matched in the single_trial function 
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
    t=nanmean(t1,1);
    clear t2 t1

 %save([my_path '\LDtransition\' mouse_info.ID '_dF_' mouse_info.analysis_type],'dF1','t','analysis')
end
estrus_states=All_estrus_states(sess_ind);

switch mouse_info.analysis_type
    case 'male_female'
        for ei=1:length(estrus_states)
            % pick a few by choosing P as representative
           if strfind(estrus_states{ei},'P'); estrus_states{ei}='Female'; end
        end
        estrus_states_titles={'Female','OVX','male'};
    case 'estrous_cycle'
        estrus_states_titles={'P-2','P-1','P+0','P+1','P+2','OVX','OVX+Esr','OVX+PR'};
%         if ~OVX
%             estrus_states_titles=estrus_states_titles(1:5);
%         end
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
    switch newF
        case 'newF'
             new_f_limits=[0.0005 0.007; 0.007 0.05; 0.05 0.1; 0.1 0.25; 0.25 0.45; 0.45 1.0 ; 1.0 1.35];
        otherwise
            new_f_limits=[0.0005 0.03; 0.03 0.1; 0.1 0.35; 0.35 0.65; 0.65 1.0; 1.0 1.35 ; 1.35 1.65; 1.65 2.3; 2.3 4];
    end
     new_f_limits=flip(new_f_limits,1);
            
%     switch mouse_info.analysis_type
%         case 'estrous_cycle'
            all_fft_freq_range_array1=[];
            all_fft_freq_range_array2=[];
            all_fft_freq_range_array3=[];
            for i=1:length(analysis) % all sessions
                f_limits=analysis{i}.int_fft_limits;
                all_fft_freq_range_1(i,:)=analysis{i}.int_fft(1,:);% 0 to 0.03 Hz lower frequencies
                all_fft_freq_range_2(i,:)=analysis{i}.int_fft(2,:);%  frequencies
                all_fft_freq_range_3(i,:)=analysis{i}.int_fft(3,:);% frequencies
                all_fft_freq_range_array1=[all_fft_freq_range_array1 analysis{i}.int_fft(1,:)];
                all_fft_freq_range_array2=[all_fft_freq_range_array2 analysis{i}.int_fft(2,:)];
                all_fft_freq_range_array3=[all_fft_freq_range_array3 analysis{i}.int_fft(3,:)];
            end
            
            %plot freqyency dependancy  `
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
                   case 'estrous_cycle'
                       ES=unique(estrus_states);ES=ES(~cellfun('isempty', ES));% remove OVX if doesn't exist 
                       ES=ES(~strcmp(ES,'OVX+Esr'));
                       ES=ES(~strcmp(ES,'OVX+PR'));
                       for esi=length(ES): -1:1
                       %for esi=length(ES): -1:1
                           if ~isempty(find(strcmp(estrus_states,estrus_states_titles{esi})))
                               legend_estrus_inds(esi)=min(find(strcmp(estrus_states,estrus_states_titles{esi})));
                           else
                               legend_estrus_inds(esi)=nan;
                           end
                       end
                   case 'male_female'
                       if ~isempty(find(strcmp(estrus_states,estrus_states_titles{1})))
                           legend_estrus_inds=min(find(strcmp(estrus_states,estrus_states_titles{1})));
                       else
                           legend_estrus_inds=nan;
                       end
               end

            %data_to_plot=all_fft_freq_range_1; range_title='0-0.03 Hz';YLIM_Val=[0 50];
            if  show_fft
                data_to_plot=all_fft_freq_range_2; range_title=[ f_limits(2,:) ' Hz']; YLIM_Val=[2 20];
                figure
                legend_included=[];
                for i=1:size(data_to_plot,1)
                    
                    ph{i}=plot((i-1)*size(dF1,2)+[1:size(dF1,2)],data_to_plot(i,:));hold on
                    if ~isempty(estrus_states{i})
                        this_estrus{1}=estrus_states{i};
                        [ALL_colors_fft,color_ind2]=get_estrus_colors(this_estrus);
                    else
                        ALL_colors_fft=[0 0 0]; color_ind2=1;
                    end
                    ph{i}.Color=ALL_colors(color_ind2,:);
                end
                legend_estrus_inds=legend_estrus_inds(~isnan(legend_estrus_inds));
                legend_included=[legend_included ph{legend_estrus_inds}];
                legend(legend_included,estrus_states_titles(1:length(legend_estrus_inds)))
                title([mouse_info.ID ' ' range_title ' fft'])
                ylabel('int. power fft')
                xlabel('Time (hour)')
                ylim(YLIM_Val)
            end
           
            % calcualte autocorrelation and plot it
            ac_estrus_inds=[];
            ac_OVX_inds=[];
            ac_estrus_inds=find(strcmp(estrus_states,'P+0'));
            ac_OVX_inds=find(strcmp(estrus_states,'OVX'));
            % find the indexes of full estrous cycle
            estrus_inds(:,1)=find(strcmp(estrus_states,'P-2'));
            estrus_inds(:,2)=estrus_inds(:,1)+3;
            if ~isempty(find(strcmp(estrus_states,'OVX')))
                ovxi=find(strcmp(estrus_states,'OVX'));
                estrus_inds=[estrus_inds; [ovxi(1) ovxi(end)]];
            end
            
            
            if show_fft
                data_str_to_plot= {'all_fft_freq_range_array1' 'all_fft_freq_range_array2' 'all_fft_freq_range_array3'};
                figure
                for pi=1:length(data_str_to_plot)
                    eval(['data_to_plot=' data_str_to_plot{pi} ';']);
                    subplot(length(data_str_to_plot),1,pi)
                    autocorr(double(data_to_plot),'NumLags',length(data_to_plot)-round(length(data_to_plot)*0.12));hold on;
                    if ~isempty(ac_estrus_inds)
                        ph=line([ac_estrus_inds*24 ac_estrus_inds*24],[0 1]);
                        for i=1:length(ph); ph(i).Color='r'; end
                    end
                    if ~isempty(ac_OVX_inds)
                        ph=line([ac_OVX_inds*24 ac_OVX_inds*24],[0 1]);
                        for i=1:length(ph); ph(i).Color='k'; end
                    end
                    xlabel('Lag (hours)')
                    title([mouse_info.ID  ' ' num2str(f_limits(pi,:)) ' Hz fft'])
                    xlim([0 12*24])
                    ylim([-0.6 0.6])
                end
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
%             % shift FFT to light - dark 
%             tmp_FFT{1}=cat(2,FFT_POWER_INT_by_freq{1}(:,17:24),FFT_POWER_INT_by_freq{1}(:,1:16));
%             for si=2:length(analysis) % all sessions
%                 tmp_FFT{si}=cat(2,FFT_POWER_INT_by_freq{si-1}(:,17:24),FFT_POWER_INT_by_freq{si}(:,1:16));
%                %FFT_POWER_INT_by_freq{si}=;
%             end  
            tmp_FFT=FFT_POWER_INT_by_freq;
            if show_fft
                % plot the fft power integrals
                clims=[0 10];
                figure
                L=length(tmp_FFT) ;
               % L=10;
                for i=1:L% all sessions
                    subplot(5,ceil(L/5),i)
                    %image( FFT_POWER_INT_by_freq{i},'CDataMapping','scaled')
                    imagesc(tmp_FFT{i},clims)
                    colorbar
                    title([mouse_info.ID '  ' estrus_states{i} ])
                    yticks([1: size(tmp_FFT{i},1)])
                    yticklabels(num2str(new_f_limits))
                    xlabel('Time (hours)')
                    ylabel('Int. Power fft')
                end
            end
            
  %  end
end
for i=1:length(estrus_states_titles)
    estrus_states_ind{i}=find(strcmp(estrus_states,estrus_states_titles{i}));
end
dF1(:,:,1:300)=0;% clean the first few seconds

clear mean_over_all_samples mean_dF_over_estrus_samples rate_over_estrus_samples mean_dF_over_hour width_over_estrus_samples mean_rate_over_estrus_samples mean_width_over_estrus_samples
mean_dF_over_all_samples(:,:)=nanmean(dF1,1);
median_dF_over_all_samples(:,:)=nanmedian(dF1,1);
mean_dF_over_hour(:,:)=nanmean(dF1,3);
median_dF_over_hour(:,:)=nanmedian(dF1,3);
if trial_info.show
    figure
    subplot(2,1,1)
    for hi=1:size(mean_dF_over_all_samples,1)
        %plot(mean_dF_over_all_samples(hi,:)+hi*10); hold on
      %  plot(mean_dF_over_all_samples(hi,:)); hold on
       plot(t,median_dF_over_all_samples(hi,:)); hold on
    end
     ylabel('median dF all samples')
    subplot(2,1,2)
    
    %    plot([1:24],mean_dF_over_hour'); hold on
    plot([1:size(dF1,2)],median_dF_over_hour'); hold on
    ylabel('median dF')
end

clear rate_over_estrus_samples width_over_estrus_samples mean_dF_over_estrus_samples median_dF_over_estrus_samples
clear median_width_over_estrus_samples median_rate_over_estrus_samples mean_width_over_estrus_samples mean_rate_over_estrus_samples
switch mouse_info.analysis_type
    case 'estrous_cycle'
    for sti=1:length(estrus_states_ind)
        if length(estrus_states_ind{sti})>= analysis_params.minimum_trials
            mean_dF_over_estrus_samples(sti,:,:)=nanmean(dF1(estrus_states_ind{sti},:,:),1);
            median_dF_over_estrus_samples(sti,:,:)=nanmedian(dF1(estrus_states_ind{sti},:,:),1);
        else
            mean_dF_over_estrus_samples(sti,:,:)=nan(size(dF1,2),size(dF1,3));
            median_dF_over_estrus_samples(sti,:,:)=nan(size(dF1,2),size(dF1,3));
        end
    
            
        k=0;

        % in case no recording for a specific state or not enough
        if length(estrus_states_ind{sti})<analysis_params.minimum_trials
            L=1;
        else
            L=length(estrus_states_ind{sti});
        end
        ind2=[];
        for st2i=1:L
            if ~isempty(estrus_states_ind{sti})
                ind2=estrus_states_ind{sti}(st2i);
            end
             k=k+1;
            if ~isempty(ind2)
                rate_over_estrus_samples(sti,k,:)=analysis{ind2}.rate;
                width_over_estrus_samples(sti,k,:)=analysis{ind2}.med_width;
            else      
                rate_over_estrus_samples(sti,k,:)=nan(1,size(analysis{1}.rate,2));
                width_over_estrus_samples(sti,k,:)=nan(1,size(analysis{1}.rate,2));
            end
        end
        % indroduce nan-s if  
        % doesn't exist or not enough repeats
        if length(estrus_states_ind{sti})<analysis_params.minimum_trials  %
            mean_rate_over_estrus_samples(sti,:)=nan(1,size(rate_over_estrus_samples,3));
            mean_width_over_estrus_samples(sti,:)=nan(1,size(rate_over_estrus_samples,3));
            median_rate_over_estrus_samples(sti,:)=nan(1,size(rate_over_estrus_samples,3));
            median_width_over_estrus_samples(sti,:)=nan(1,size(rate_over_estrus_samples,3));
        else
            mean_rate_over_estrus_samples(sti,:)=nanmean(rate_over_estrus_samples(sti,:,:),2);
            mean_width_over_estrus_samples(sti,:)=nanmean(width_over_estrus_samples(sti,:,:),2);
            median_rate_over_estrus_samples(sti,:)=nanmedian(rate_over_estrus_samples(sti,:,:),2);
            median_width_over_estrus_samples(sti,:)=nanmedian(width_over_estrus_samples(sti,:,:),2);
        end
    end
    if trial_info.show
        figure
        for sti=1:size(mean_dF_over_estrus_samples,1)
            subplot(1,size(mean_dF_over_estrus_samples,1),sti)
            temp_dF(:,:)=mean_dF_over_estrus_samples(sti,:,:);
            for hi=1:size(mean_dF_over_all_samples,1)
                plot(temp_dF(hi,:)+10*hi); hold on
            end
            clear temp_dF
        end
    end
    case 'male_female' %%% no limitation here on number of repeats- to update if needed!!!!
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
  

clear by_state_matrix mean_dF_by_state median_dF_by_state by_state_matrix
switch mouse_info.analysis_type
    case 'estrous_cycle'
        for sti=1:length(estrus_states_ind)
            if length(estrus_states_ind{sti})>=analysis_params.minimum_trials
                mean_dF_by_state(sti,:)=nanmean(mean_dF_over_hour(estrus_states_ind{sti},:),1);
                median_dF_by_state(sti,:)=nanmedian(mean_dF_over_hour(estrus_states_ind{sti},:),1);
                by_state_matrix(sti,1:size(mean_dF_over_hour,2))=estrus_states_titles(sti);
            else
                mean_dF_by_state(sti,:)=nan(1,size(mean_dF_over_hour,2));
                median_dF_by_state(sti,:)=nan(1,size(mean_dF_over_hour,2));
                by_state_matrix(sti,1:size(mean_dF_over_hour,2))=estrus_states_titles(sti); 
            end
        end
        if trial_info.show
            figure
            plot(mean_dF_by_state')
            title([ mouse_info.ID ' mean dF by state'])
            xlabel('Time (hours)')
        end
    case 'male_female'
        mean_dF_by_state(:)=nanmean(mean_dF_over_hour(estrus_states_ind{sti},:),1);
         median_dF_by_state(:)=nanmedian(mean_dF_over_hour(estrus_states_ind{sti},:),1);
         by_state_matrix(1,1:size(mean_dF_over_hour,2))=estrus_states_titles(sti);
end

%% calculates medain of dark/light
switch mouse_info.analysis_type
    case 'male_female'
        full_time_estrus_states_titles={'Female D','Female L','OVX D','OVX L','Male D','Male L'};
    case 'estrous_cycle'
        full_time_estrus_states_titles={'P-2 D','P-1 L','P-1 D','P+0 L','P+0 D','P+1 L','P+1 D','P+2 L','P+2 D','OVX D','OVX L','OVX+Esr1 D','OVX+Esr1 L','OVX+PR D','OVX+PR L'};
end

%% gets the value of each parameter for the total dark or light phase -
%edit to fit ZT10 to 13-060922
% 05/12/22- change to take median instead of mean 
clear tmp tmp2

% adjust array length in case there is no 'onset'
% if length(median_dF_by_state)<13
%    median_dF_by_state=[nan nan  median_dF_by_state];
%    median_rate_over_estrus_samples=[nan nan  median_rate_over_estrus_samples];
%    median_width_over_estrus_samples=[nan nan  median_width_over_estrus_samples];
% end


tmp=median_dF_by_state;%  estrus, hour
full_time_dF=reshape(tmp',1,size(tmp,1)*size(tmp,2));
tmp2=mean_rate_over_estrus_samples;%  estrus, hour
%tmp2=median_rate_over_estrus_samples;%  
full_time_rate=reshape(tmp2',1,size(tmp2,1)*size(tmp2,2));
tmp3=mean_width_over_estrus_samples;%  estrus, hour
%tmp3=median_width_over_estrus_samples;%  estrus, hour
full_time_width=reshape(tmp3',1,size(tmp3,1)*size(tmp3,2));

switch analysis_params.n_windows
    case 15
        % create a group matrix for averaging over time groups, ZT10, ZT11 and ZT12
        light_status_mat=zeros(size(tmp,1),size(tmp,2));
        
        light_status_mat(:,1:3)=1;
        %light_status_mat(:,4)=1;
        light_status_mat(:,7:12)=2;
        light_status_mat(:,13:18)=3;
    case 3
        % create a group matrix for averaging over time groups, ZT10, ZT11 and ZT12
        light_status_mat=zeros(size(tmp,1),size(tmp,2));
        light_status_mat(:,1)=1;
        %light_status_mat(:,4)=1;
        light_status_mat(:,2)=2;
        light_status_mat(:,3)=3;
end

by_state=reshape(by_state_matrix',1,size(by_state_matrix,1)*size(by_state_matrix,2));

clear dF_dark_light rate_dark_light width_dark_light
light_status=reshape(light_status_mat',1,size(light_status_mat,1)*size(light_status_mat,2));

switch mouse_info.analysis_type
    case 'male_female'
        %%% check here if mean will do better !!!
        dF_dark_light=[nanmedian(full_time_dF(light_status==1)) nanmedian(full_time_dF(light_status==2)) nanmedian(full_time_dF(light_status==3))];
        rate_dark_light=[nanmedian(full_time_rate(light_status==1)) nanmedian(full_time_rate(light_status==2)) nanmedian(full_time_rate(light_status==3))];
        width_dark_light=[nanmedian(full_time_width(light_status==1)) nanmedian(full_time_width(light_status==2)) nanmedian(full_time_width(light_status==3))];
        
    case 'estrous_cycle'
        % for estous cycle 
        %ind_end=20*length(full_time_dF)/size(dF1,2);% should give 100 for females with no OVX (5 states), 120 to females with OVX (6 states) 
        %light_switch=sort([light_off, light_on]);
        
       % findgroups(by_state)
        for ti=1:size(by_state_matrix,1)% 
            tmp_df=full_time_dF(strcmp(by_state,by_state_matrix{ti,1}));
            tmp_rate=full_time_rate(strcmp(by_state,by_state_matrix{ti,1}));
            tmp_width=full_time_width(strcmp(by_state,by_state_matrix{ti,1}));
            light_status_array=light_status_mat(strcmp(by_state_matrix,by_state_matrix{ti,1}));
            %             dF_dark_light(ti,:)=[nanmedian(tmp_df(light_status_array==1)) nanmedian(tmp_df(light_status_array==2)) nanmedian(tmp_df(light_status_array==3))];
            %             rate_dark_light(ti,:)=[nanmedian(tmp_rate(light_status_array==1)) nanmedian(tmp_rate(light_status_array==2)) nanmedian(tmp_rate(light_status_array==3))];
            %             width_dark_light(ti,:)=[nanmedian(tmp_width(light_status_array==1)) nanmedian(tmp_width(light_status_array==2)) nanmedian(tmp_width(light_status_array==3))];
            dF_dark_light(ti,:)=[nanmean(tmp_df(light_status_array==1)) nanmean(tmp_df(light_status_array==2)) nanmean(tmp_df(light_status_array==3))];
            rate_dark_light(ti,:)=[nanmean(tmp_rate(light_status_array==1)) nanmean(tmp_rate(light_status_array==2)) nanmean(tmp_rate(light_status_array==3))];
            width_dark_light(ti,:)=[nanmean(tmp_width(light_status_array==1)) nanmean(tmp_width(light_status_array==2)) nanmean(tmp_width(light_status_array==3))];
            
        end
        
%         OVX_dF_dark_light=[nanmedian(full_time_dF(OVX_light_off)) nanmedian(full_time_dF(OVX_light_on))];
%         OVX_rate_dark_light=[nanmedian(full_time_rate(OVX_light_off)) nanmedian(full_time_rate(OVX_light_on))];
%         OVX_width_dark_light=[nanmedian(full_time_width(OVX_light_off)) nanmedian(full_time_width(OVX_light_on))];
%         
%         dF_dark_light=[OVX_dF_dark_light dF_dark_light];
%         rate_dark_light=[OVX_rate_dark_light rate_dark_light];
%         width_dark_light=[OVX_width_dark_light width_dark_light];
end




%% now plot
% 
% figure
% for hi=1:size(mean_dF_over_all_samples,1)
%     plot(t,mean_dF_over_all_samples(hi,:)+5*hi); hold on
% end
% ylim([-20 (hi+3)*5])
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
            set(ph,'Color',ALL_colors(color_ind(sti),:))
        end
        ylim([-20 (hi+3)*step])
        title(['VIPGC' mouse_info.ID ' ' estrus_states_titles{sti}])
        ylabel('dF')
    case 'estrous_cycle'
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
        sti=find(strcmpi(mouse_info.sex,estrus_states_titles));
        subplot(1,3,1)
        %lh=plot(mean_dF_over_hour(estrus_states_ind{sti},:)','o'); hold on
        lh=plot(median_dF_over_hour(estrus_states_ind{sti},:)','o'); hold on
        set(lh,'Color',ALL_colors(color_ind(sti),:))
        %lh2=plot( mean_dF_by_state(:)); hold on
         lh2=plot( median_dF_by_state(:)); hold on
        set(lh2,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        legend (estrus_states_titles{sti})
        ylabel('dF')
        title(['VIPGC' mouse_info.ID ])
        
        % plot rates
        subplot(1,3,2)
        
       % ph=plot(mean_rate_over_estrus_samples(:));hold on
        ph=plot(median_rate_over_estrus_samples(:));hold on
        set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        legend (estrus_states_titles{sti})
        ylabel('rates')
        title(['VIPGC' mouse_info.ID ])
        
        % plot width
        subplot(1,3,3)
       % ph=plot(mean_width_over_estrus_samples(:));hold on
        ph=plot(median_width_over_estrus_samples(:));hold on
        set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
        legend (estrus_states_titles{sti})
        ylabel('width')
        title(['VIPGC' mouse_info.ID ])
        
    case 'estrous_cycle'
        for i=1:length(estrus_states_titles)
            estrus_states_ind2{i}=find(strcmp(estrus_states,estrus_states_titles{i}));
        end
        figure
        subplot(1,3,1)
        for sti=1:size(mean_dF_by_state,1)
            if sum(~isnan(mean_dF_by_state(sti,:)))>0
                lh=plot(mean_dF_over_hour(estrus_states_ind2{sti},:)','o'); hold on
                %lh=plot(median_dF_over_hour(estrus_states_ind{sti},:)','o'); hold on
                set(lh,'Color',ALL_colors(color_ind(sti),:))
                lh2=plot( mean_dF_by_state(sti,:)); hold on
                %lh2=plot( median_dF_by_state(sti,:)); hold on
                set(lh2,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
            end
        end
        legend (estrus_states_titles(find(~cellfun('isempty',estrus_states_ind2))))
        ylabel('dF')
        title(['VIPGC' mouse_info.ID ])
        
        % plot rates
        subplot(1,3,2)
        for sti=1:length(estrus_states_ind2)
            if sum(~isnan(mean_rate_over_estrus_samples(sti,:)))>0
                ph=plot(mean_rate_over_estrus_samples(sti,:));hold on
                %ph=plot(median_rate_over_estrus_samples(sti,:));hold on
                set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
            end
        end
        legend (estrus_states_titles(find(~cellfun('isempty',estrus_states_ind2))))
        ylabel('rates')
        title(['VIPGC' mouse_info.ID ])
        
        % plot width
        subplot(1,3,3)
        for sti=1:length(estrus_states_ind2)
            if sum(~isnan(mean_width_over_estrus_samples(sti,:)))>0
                ph=plot(mean_width_over_estrus_samples(sti,:));hold on
                %ph=plot(median_width_over_estrus_samples(sti,:));hold on
                set(ph,'Color',ALL_colors(color_ind(sti),:),'Linewidth',3)
            end
        end
        legend (estrus_states_titles(find(~cellfun('isempty',estrus_states_ind2))))
        ylabel('width')
        title(['VIPGC' mouse_info.ID ])
end

output.t=t;
output.analysis=analysis;
output.dF1=dF1;
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

output.mean_dF_dark_light=dF_dark_light;
output.mean_rate_dark_light=rate_dark_light;
output.mean_width_dark_light=width_dark_light;
if FFT_cc
    output.FFT_POWER_INT_by_freq=FFT_POWER_INT_by_freq;
end
output.estrus_states=estrus_states;
output.new_f_limits=new_f_limits;

if ~exist(this_filename)
    save([my_path '\LDtransition\output_' mouse_info.analysis_type '_' newF mouse_info.ID '_MT' num2str(analysis_params.minimum_trials)], 'output','-v7.3')
else
    opts.Interpreter = 'tex';
    % Include the desired Default answer
    opts.Default = 'Yes';
    % Use the TeX interpreter to format the question
    quest = 'File exists, to overwrite?';
    answer = questdlg(quest,'Boundary Condition',...
        'Yes','No',opts);
    switch answer
        case  'Yes'
            save([my_path '\LDtransition\output_' mouse_info.analysis_type '_' newF mouse_info.ID '_MT' num2str(analysis_params.minimum_trials)], 'output','-v7.3')
    end
end
% if mouse_info.analysis_type
%  save([my_path '\LDtransition\output_general_' mouse_info.ID], 'output')
% end
disp(['t limits: ' num2str(min(t)) ' ' num2str(max(t))])
disp(['t range: ' num2str(max(t)-min(t)) ' sec'])
disp (['done ' mouse_info.ID])