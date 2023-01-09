function FP_LDtransition_FFT_output_for_classifier_OVX_hormones 
% run the FFT analysis per mouse and look at classificiation with OVX and
% OVX with hormones
% uses 'get_LDtransition_FP_per_mouse'

%% get experimental information: 
my_path='D:\DATA_Glab\fiberphotometry\LDtransition\';

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
ind=ind+1; mouse_info{ind}.ID='115L'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
%ind=ind+1; mouse_info{ind}.ID='A116R';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='119LL';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='122R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='123L';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='128R'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';

% females w/o OVX
ind=ind+1; mouse_info{ind}.ID='161RR'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='166R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='175L';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';%
ind=ind+1; mouse_info{ind}.ID='176RL'; mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';
ind=ind+1; mouse_info{ind}.ID='198L'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
%ind=ind+1; mouse_info{ind}.ID='200RR'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';

n_females=ind;

% ind=ind+1; mouse_info{ind}.ID='25L'; mouse_info{ind}.side='L'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male'; %male
% ind=ind+1; mouse_info{ind}.ID='60N'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
% ind=ind+1; mouse_info{ind}.ID='62L'; mouse_info{ind}.side='R'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
% ind=ind+1; mouse_info{ind}.ID='103R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male not good
% ind=ind+1; mouse_info{ind}.ID='106LL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
% n_males=ind-n_females;
%
%OVX 13-16
% ind=ind+1; mouse_info{ind}.ID='247RRL_OVX';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
% ind=ind+1; mouse_info{ind}.ID='246RL_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
% ind=ind+1;mouse_info{ind}.ID='261RL_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
% ind=ind+1; mouse_info{ind}.ID='259R_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX'; % OVX
% n_OVX=ind-n_males-n_females;

% analysis_type='estrous_cycle'
% %analysis_type='male_female';

% for i=1:length(mouse_info)
%     mouse_info{i}.analysis_type=analysis_type;
% end

estrous_states_classes={'P-1','P+0','P+1'};
%estrous_states_classes={'P-2','P-1','P+0','P+1','P+2'};
%estrous_states_allclasses=[estrous_states_classes 'OVX' 'OVX+Esr' 'OVX+PR' ];
%estrous_states_for_classification={'NR' 'RE' 'Male' 'OVX'};%    
estrous_states_allclasses=[estrous_states_classes 'OVX' ];

[ALL_colors,color_ind]=get_estrus_colors(estrous_states_allclasses);

% get data for each mousetic 
tic
newF='newF'
for idi=1:length(mouse_info)
    mouse_info{idi}.minimum_trials=minimum_trials;
    switch mouse_info{idi}.sex
        case 'Female'
            if ~exist ([my_path 'output_estrous_cycle_' newF mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])
                disp(['get data for ' mouse_info{idi}.ID])
                output= get_LDtransition_FP_per_mouse (mouse_info{idi});
            else
                load([my_path 'output_estrous_cycle_' newF mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])% load 'output'
            end
        case {'male','OVX'} % for now no difference bweteen males and females
            if ~exist ([my_path 'output_male_female_' newF mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])
                disp(['get data for ' mouse_info{idi}.ID])
                output = get_LDtransition_FP_per_mouse (mouse_info{idi});
            else
                load([my_path 'output_male_female_' newF mouse_info{idi}.ID '_MT' num2str(mouse_info{idi}.minimum_trials) '.mat' ])%all_output{idi}=output;
            end 
    end
    output.new_estrus_states=estrus_to_receptive(output.estrus_states);
    all_output{idi}=output;
end
toc
% resize sessions -remove the 30 minutes with no activity 
new_inds=[1:3 7:18];
for idi=1:length(mouse_info)
    output=all_output{idi};
    for si=1:size(output.FFT_POWER_INT_by_freq,2)
        output.FFT_POWER_INT_by_freq{si}=output.FFT_POWER_INT_by_freq{si}(:,new_inds);
    end
    all_output{idi}.FFT_POWER_INT_by_freq=output.FFT_POWER_INT_by_freq;
end


  %new_f_limits=[0 0.03; 0.03 0.1; 0.1 0.35; 0.35 0.65; 0.65 1.0; 1.0 1.35 ; 1.35 1.65; 1.65 2.3; 2.3 4];
new_f_limits=all_output{1}.new_f_limits;        

% plot one female FFT for paper 
ind=5; %mouse_info{ind}.ID. 
this_output=all_output{ind};
% get FFT for states. first find the states:
si_inds=[];index=[];
for i=1:length(estrous_states_allclasses)
    this_ind=min(find(strcmp(this_output.estrus_states,estrous_states_allclasses{i})));
    if ~isempty(this_ind)
        si_inds=[si_inds this_ind];
        index=[index i];
    end
end
for si=1:length(si_inds)
    FFT_POWER_INT{si}=this_output.FFT_POWER_INT_by_freq{si_inds(si)};
end
% now plot                
freq_to_plot=[2:size(new_f_limits,1)-1]; % index
times_to_plot=1:15; % 1-3 is ZT10, 4-9 is ZT11, 10-15 is ZT12
clims=[0 15]; % colorbar values control 
included_states=estrous_states_allclasses(index);
figure
for si=1:length(si_inds)
    subplot(length(si_inds),1,si)
    imagesc(FFT_POWER_INT{si}(freq_to_plot,times_to_plot),clims)
    colorbar
    title([included_states{si} ' ' mouse_info{ind}.ID] )
    yticks([1: length(freq_to_plot)])
    yticklabels(num2str(new_f_limits(freq_to_plot,:)))
    xlabel('ind')
    ylabel('median Int. Power fft')   
end
%print(['LD_FFT_' mouse_info{ind}.ID '_all'],'-depsc')

% plot just these states 
plot_autocorr=0
if plot_autocorr
    for idi=1:n_females
        this_output=all_output{idi};
        FFT_1=[];% look just at the first frequencies
        FFT_2=[];% look just at the second frequencies
        FFT_3=[];% look just at the third frequencies
        FFT_4=[];% look just at the fourth frequencies
        ResponseVarName1=[];
        % go over days and find the days that are relevant
        number_of_states_contains=0;
        for si=1:length(this_output.estrus_states);
            if contains(this_output.estrus_states{si},estrous_states_allclasses)
                this_FFT=this_output.FFT_POWER_INT_by_freq{si}(:,times_to_plot);
                FFT_1=[FFT_1 this_FFT(1,:)];
                FFT_2=[FFT_2 this_FFT(2,:)];
                FFT_3=[FFT_3 this_FFT(3,:)];
                FFT_4=[FFT_4 this_FFT(4,:)]; % size should be 24*number_of_states_contains
                ResponseVarName1=[ResponseVarName1 {this_output.estrus_states{si}}];
                number_of_states_contains=number_of_states_contains+1;
            end
        end
        if ~isempty(FFT_1)
            figure
            subplot(4,1,1)
            autocorr(double(FFT_1),'NumLags',length(FFT_1)-ceil(0.15*length(FFT_1)));hold on;
            xlim([0 100])
            title([mouse_info{idi}.sex  ' ' mouse_info{idi}.ID])
            subplot(4,1,2)
            autocorr(double(FFT_2),'NumLags',length(FFT_2)-ceil(0.15*length(FFT_1)));hold on;
            xlim([0 100])
            
            subplot(4,1,3)
            autocorr(double(FFT_3),'NumLags',length(FFT_3)-ceil(0.15*length(FFT_1)));hold on;
            xlim([0 100])
            subplot(4,1,4)
            autocorr(double(FFT_4),'NumLags',length(FFT_4)-ceil(0.15*length(FFT_1)));hold on;
            xlim([0 100])
        end
    end
end
% gather each state  histogram % results in matrix of #animals x #STATES
 k=0;
for idi=1:length(all_output)
    this_output=all_output{idi};
    ES=this_output.estrus_states;

    % go over days and find the days that are relevant 
     number_of_states_contains=0;
     for si=1:length(estrous_states_allclasses) % + male and OVX
         FFT_by_state{idi,si}=[];
%          FFT_by_ID_state{idi}{1}=[];
%          states_by_ID{idi}{1}=[];
     end
            
     for si=1:length(ES)% over states
         if contains(ES{si},estrous_states_allclasses)
             k=k+1;
             s_ind=find(strcmp(ES{si},estrous_states_allclasses));
             if ~isempty(s_ind)
                 FFT_by_state{idi,s_ind}=[FFT_by_state{idi,s_ind} all_output{idi}.FFT_POWER_INT_by_freq{si}(:,times_to_plot)] ;
                 FFT_by_state_all_ID{k}=this_output.FFT_POWER_INT_by_freq{si}(:,times_to_plot);
                 states{k}=ES{si};
             end
         end
     end
        
end
% save
save_all_matrix=1;
if save_all_matrix
    save('LD_OVX_FFT_matrix.mat','FFT_by_state_all_ID')
    save('states.mat','states')
end
    
% gather information about a specific frequency at a specific hour and look
% at the histogram 

time_frame_start=0;    % start is ZT10, but the function is based 
freq_ind=6;
hour_ind=[1:9]; %1-3 is ZT10
%states_to_compare={{'P-2','P-1'},{'P+0','P+1'}}; states_names={'NR', 'RE'};
states_to_compare={{'P-2','P-1'},{{'P+1'}}}; states_names={'M/D', 'E'};
%states_to_compare={{'P+1','P+2'},{'OVX'}}; states_names={'RE', 'OVX'};
%states_to_compare={'OVX+PR','OVX+Esr'}; states_names=states_to_compare
%states_to_compare={'OVX','Male'}; states_names=states_to_compare
clear state_FFT_byhour_byfreq
for hi=1:length(hour_ind)
    for i=1:length(states_to_compare)
        state_FFT_byhour_byfreq{hi}{i}=[];% first index is hour, second is state
    end
end
for hi=1:length(hour_ind)
    for idi=1:length(all_output)
        this_output=all_output{idi};
        %if strmatch(mouse_info{idi}.sex,'Female')
        for i=1:length(states_to_compare)
            ind1=[];
            for ssi=1:numel(states_to_compare{i})
                ind1=[ind1; find(strcmp(this_output.estrus_states,states_to_compare{i}{ssi}))];
            end
            for ini=1:length(ind1)
                state_FFT_byhour_byfreq{hi}{i}=[state_FFT_byhour_byfreq{hi}{i} this_output.FFT_POWER_INT_by_freq{ind1(ini)}(freq_ind,hour_ind(hi))];
            end
        end
        %end
    end
end
% plot histograms 
edges=[0:1:20];
marker_colors={'b','r'};
figure
for hi=1:length(hour_ind)
    subplot(1,length(hour_ind),hi)
    for i=1:length(state_FFT_byhour_byfreq{hi})
        data=state_FFT_byhour_byfreq{hi}{i};
        h(i)=histogram(data,edges); hold on
        dataMean = nanmean(data);
        dataMedian = nanmedian(data);
        xline(dataMedian, 'Color', marker_colors{i}, 'LineWidth', 2);
        disp(['Mean+-sdt FFT values for time-frame ' num2str(hour_ind(hi)) ' ' states_names{i} ': ' num2str(dataMean) '+-' num2str(std(data)/sqrt(length(data)))])
    end
    legend(states_names)
    title(['Recording hour: time-frame ' num2str(hour_ind(hi))])
    xlim([0 20])
    xlabel('FFT amplitudes')
    axis square
    pd(hi)=pdist2(h(1),h(2),'Euclidean'  ); %  'Euclidean' 
end

% % plot values at 2D 
markers={'sb','or'};

hours_to_compare=[2,3];% the indexs relate to 'state_FFT_byhour_byfreq', not ZT 
figure
for i=1:length(states_to_compare)
    plot(state_FFT_byhour_byfreq{hours_to_compare(1)}{i},state_FFT_byhour_byfreq{hours_to_compare(2)}{i},markers{i}); hold on
end

for i=1:length(states_to_compare) 
    L=length(state_FFT_byhour_byfreq{hours_to_compare(1)}{i});
    med_FFT(i,1)=nanmedian(state_FFT_byhour_byfreq{hours_to_compare(1)}{i});
    med_FFT(i,2)=nanmedian(state_FFT_byhour_byfreq{hours_to_compare(2)}{i});
    sem_FFT(i,1)=1.2533*nanstd(state_FFT_byhour_byfreq{hours_to_compare(1)}{i})/sqrt(L);
    sem_FFT(i,2)=1.2533*nanstd(state_FFT_byhour_byfreq{hours_to_compare(2)}{i})/sqrt(L);
    %SE (median) = 1.2533 × SE()
     plot(med_FFT(i,1),med_FFT(i,2),markers{i},'MarkerSize',10,'MarkerFaceColor',marker_colors{i}); hold on
end
legend(states_names)
xlim([0 20]); ylim([0 20]); axis square
ylabel(['FFT values at time-frame ' num2str(time_frame_start+ hour_ind(hours_to_compare(2)))])
xlabel(['FFT values at time-frame ' num2str(time_frame_start+ hour_ind(hours_to_compare(1)))])

COM=sqrt((med_FFT(1,1)-med_FFT(2,1))^2+(med_FFT(1,2)-med_FFT(2,2))^2);
com_sem=sqrt((sem_FFT(1,1)-sem_FFT(2,1))^2+(sem_FFT(1,2)-sem_FFT(2,2))^2);
disp([states_names{1} ' to ' states_names{2} ' Center of mass distance: ' num2str(COM) '+-' num2str(com_sem)])

% average data by state - females
%full_size=full_size-1;% should be 13- be careful here
full_size=15;
clear FFT_median_by_ID FFT_mean_by_ID FFT_mean_by_ID_sem FFT_median_by_state
%females_FFT_by_state=FFT_by_state(1:n_females,1:numel(estrous_states_classes));
for si=1:size(FFT_by_state,2)
    k=0;
    for idi=1:size(FFT_by_state,1)
        
        this_state_FFT=FFT_by_state{idi,si};
        if ~isempty(this_state_FFT) 
            FFT_temp=reshape(this_state_FFT,size(this_state_FFT,1),full_size,size(this_state_FFT,2)/full_size);
            if size(FFT_temp,3)>1
                k=k+1;
                disp([mouse_info{idi}.ID ' ' estrous_states_allclasses{si}  ' has ' num2str(size(FFT_temp,3)) ' repeats' ]);
                FFT_median_by_ID{si}(k,:,:)=nanmedian(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
                FFT_mean_by_ID{si}(k,:,:)=nanmean(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
                FFT_mean_by_ID_sem{si}(k,:,:)=std(FFT_temp,0,3)./sqrt(size(FFT_temp,3));
            end
        end
    end
    %FFT_median_by_ID{si}=FFT_median_by_ID{si}
    FFT_median_by_state{si}(:,:)=nanmedian(FFT_median_by_ID{si},1);
end


% now plot avergaed FFTs by state
freq_to_plot=[2:size(new_f_limits,1)-1];
clims=[0 8];
figure 
for si=1:length(FFT_median_by_state)
    subplot(length(FFT_median_by_state),1,si)
    imagesc(FFT_median_by_state{si}(freq_to_plot,:),clims)
    colorbar
    title(estrous_states_allclasses{si})
    yticks([1: size(FFT_median_by_state{si},1)])
    yticklabels(num2str(new_f_limits(freq_to_plot,:)))
    xlabel('Time-frames (10 minutes intervals)')
    ylabel('median Int. Power fft')
end


%print('LD_FP_FFT_compare accross_states','-depsc')


%  plot frequencies relationships, per state
k=0;
ref_freq=6;
hours_array=1:13;
figure
for fi=[1 3 4 5]
    k=k+1;
    subplot(1,5,k)
    for si=1:2 % non recptive
        ph1=plot(FFT_median_by_state{si}(ref_freq,hours_array),FFT_median_by_state{si}(fi,hours_array),'*'); hold on
        set(ph1, 'MarkerEdgeColor', [0 1 0], 'Marker', 'o')
    end
    for si=3:4 %  recptive
        ph11=plot(FFT_median_by_state{si}(ref_freq,hours_array),FFT_median_by_state{si}(fi,hours_array),'*'); hold on
        set(ph11, 'MarkerEdgeColor', [1 0 0], 'Marker', 'o')
    end
      title(['Freq' num2str(ref_freq) ' to ' num2str(fi)])
    xlabel(['Median freq ' num2str(ref_freq)])
    ylabel(['Median freq ' num2str(fi)])
end

%% machine learning:  define test and training sets.
% first can run one specific condition 
one_condition=0;
if one_condition
    all_inputs.freq_start_interval=4;
    all_inputs.FX=4; % additional frequencies range taken into account
    all_inputs.rel_hours=[1:3];
    all_inputs.rel_hours=[1:3];
    all_inputs.pca_features=1;% choose if pca is used to reduce dimentionality of features
    all_inputs.selected_states=estrous_states_allclasses([6,7]);
   
   %  all_inputs.method='Discriminant analysis';
    all_inputs.method='Support vector machine'
    all_inputs.mouse_info=mouse_info;
    all_inputs.plot_figure=0;
    end_table=all_machine_learning_for_FFT(all_inputs,all_output);
end

% next, can run multiple conditions, comparing 2 states at the time  
% define input data conditions 
class_by_estrous='full' % 'full' or 'semi_full' 'receptive'
%class_by_estrous='semi_full' % 'full' or 'semi_full' 'receptive'
%class_by_estrous='receptive' % 'full' or 'semi_full' 'receptive'
switch class_by_estrous
    case 'full'; by_estrous=estrous_states_allclasses(1:end) ;
    case 'semi_full'; by_estrous=estrous_states_allclasses([1:4]) ;
    case 'receptive'; by_estrous={'NR' 'RE' 'OVX' 'OVX+Esr' 'OVX+PR'};
end
switch class_by_estrous
    case 'receptive' ; all_inputs.classification_states='receptive'; %/ 'estrus_states'
    case 'semi_full' ; all_inputs.classification_states='estrus_states'; %/ ''
    case 'full' ; all_inputs.classification_states='estrus_states'; %/ ''
end
y

%% check best ZT range to classify
clear LOO_score end_table cr1 range
range=[];
for n=1:13
    a=[1:3]; range=[range; a+n-1];
end
%range=[1:9;10:12];
state_to_compare{1}='P-1';
state_to_compare{2}='P+1';
for zti=1:size(range,1)
    for n_repeats=1:10
        all_inputs.rel_hours=range(zti,:); %ZT10-10:30
        %LOO_score_legend=[];
        sti1=find(strcmp(by_estrous,state_to_compare{1}));
        sti2=find(strcmp(by_estrous,state_to_compare{2}));
        all_inputs.selected_states=by_estrous([sti1,sti2]);
        all_inputs.mouse_info=mouse_info;
        end_table=all_machine_learning_for_FFT(all_inputs,all_output);
        LOO_score{zti}=table2array(end_table(:,2));
        data_to_plot=LOO_score{zti}';
        cr1(zti,n_repeats)=data_to_plot(:,3);
    end
end

% plot score vs time (zt start)
figure
plot(range(:,1),nanmean(cr1,2)); hold on
for i=1:size(range,1)
    cr_mean=nanmean(cr1(i,:));
    cr_sem=nanstd(cr1(i,:))/sqrt(length(cr1(i,:)));
    lh=line([range(i,1) range(i,1)],[cr_mean+cr_sem cr_mean-cr_sem]);
    lh.Color='k';
end
ylabel('LOOCV score')
xlabel('ZT start, 30 minutes intervals')
title ([state_to_compare{1} ' vs. ' state_to_compare{2}])
        % LOO_score_legend=[LOO_score_legend end_table.states];
    

1