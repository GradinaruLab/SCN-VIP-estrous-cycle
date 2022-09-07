function FP_FFT_output_for_classifier 
% run the FFT analysis per mouse and look at classificiation 
% uses 'get_time_series_FP_per_mouse'

%% get experimental information: 
my_path='D:\DATA_Glab\fiberphotometry\mean_outputs\';

% Females. 1-6
ind =1; mouse_info{ind}.ID='198R';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';
ind=2; mouse_info{ind}.ID='200LL';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female';% 
ind =3; mouse_info{ind}.ID='246RL';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Female';
ind=4; mouse_info{ind}.ID='247RRL';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female';%
ind=5; mouse_info{ind}.ID='259R'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1 ;mouse_info{ind}.sex='Female'; 
ind=6; mouse_info{ind}.ID='261RL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1; mouse_info{ind}.sex='Female'; %X=8; % has 3 completed cycles, remove the last 8 sessions to look at complete 3 sets 
n_females=6;
% Males 7-12
ind=ind+1;mouse_info{ind}.ID='262R'; mouse_info{ind}.side='R'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male'; %male
ind=ind+1; mouse_info{ind}.ID='273RL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
ind=ind+1; mouse_info{ind}.ID='286R'; mouse_info{ind}.side='R'; mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
ind=ind+1;mouse_info{ind}.ID='287L';mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male not good
ind=ind+1; mouse_info{ind}.ID='288RL'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
ind=ind+1; mouse_info{ind}.ID='296R'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='Male';%male
n_males=6;
%OVX 13-16
ind=ind+1; mouse_info{ind}.ID='247RRL_OVX';mouse_info{ind}.side='L';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
ind=ind+1; mouse_info{ind}.ID='246RL_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
ind=ind+1;mouse_info{ind}.ID='261RL_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX';% OVX
ind=ind+1; mouse_info{ind}.ID='259R_OVX'; mouse_info{ind}.side='R';mouse_info{ind}.load_new_trials=1;mouse_info{ind}.sex='OVX'; % OVX
n_OVX=4;
% analysis_type='estrous_cycle'
% %analysis_type='male_female';

    


% for i=1:length(mouse_info)
%     mouse_info{i}.analysis_type=analysis_type;
% end

% switch analysis_type
%     case 'estrous_cycle'
%estrus_states_titles={'P-2','P-1','P+0','P+1','P+2','OVX'};
estrous_states_classes={'P-2','P-1','P+0','P+1','P+2'};
estrous_states_allclasses=[estrous_states_classes 'Male' 'OVX'];
estrous_states_for_classification={'NR' 'RE' 'Male' 'OVX'};%    


sex={'Male','Female','OVX'};
[ALL_colors,color_ind]=get_estrus_colors(estrous_states_allclasses);

% get data for each mouse


for idi=1:length(mouse_info)
    switch mouse_info{idi}.sex
        case 'Female'
            if ~exist ([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])
                disp(['get data for ' mouse_info{idi}.ID])
                output= get_time_series_FP_per_mouse (mouse_info{idi});
            else
                load([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])% load 'output'
            end
        case {'Male','OVX'}
            if ~exist ([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])
                disp(['get data for ' mouse_info{idi}.ID])
                output = get_time_series_FP_per_mouse (mouse_info{idi});
            else
                load([my_path 'time_series_output_general_' mouse_info{idi}.ID '.mat' ])%all_output{idi}=output;
            end 
    end
    output.new_estrus_states=estrus_to_receptive(output.estrus_states);
    all_output{idi}=output;
end
        
  %new_f_limits=[0 0.03; 0.03 0.1; 0.1 0.35; 0.35 0.65; 0.65 1.0; 1.0 1.35 ; 1.35 1.65; 1.65 2.3; 2.3 4];
new_f_limits=all_output{1}.new_f_limits;        

% shift data: original data taken at strating at 9am, while at 13 light
% turned off. Data shifted that 1am (Light) will be the start time 

for si=1:length(all_output)
    this_output=all_output{si}; % load the original data to a local variable 
    % correct the first session, based on itself only 
    tmp_FFT=cat(2,this_output.FFT_POWER_INT_by_freq{1}(:,17:24),this_output.FFT_POWER_INT_by_freq{1}(:,1:16));
    % return the new FFT matrix to the global variable: 
    all_output{si}.FFT_POWER_INT_by_freq{1}=tmp_FFT;
    % Do the same, just shift with the 8hours of the previous session,
    % based on estrous cycle logic 
    for idi=2:length(this_output.FFT_POWER_INT_by_freq)
        
            tmp_FFT=cat(2,this_output.FFT_POWER_INT_by_freq{idi-1}(:,17:24),this_output.FFT_POWER_INT_by_freq{idi}(:,1:16));
            all_output{si}.FFT_POWER_INT_by_freq{idi}=tmp_FFT;% also return the new matrix to all_output for machine learning
      
    end
end
% plot one female FFT for paper 
ind=5; %mouse_info{ind}.ID='259R';
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
               
freq_to_plot=[4:9];
clims=[0 55];
included_states=estrous_states_allclasses(index);
figure
for si=1:length(si_inds)
    subplot(length(si_inds),1,si)
    imagesc(FFT_POWER_INT{si}(freq_to_plot,:),clims)
    colorbar
    title([included_states{si} ' ' mouse_info{ind}.ID] )
    yticks([1: size(FFT_POWER_INT{si},1)])
    yticklabels(num2str(new_f_limits(freq_to_plot,:)))
    xlabel('Time (hours)')
    ylabel('median Int. Power fft')   
end
%print(['FFT_' mouse_info{ind}.ID '_all'],'-depsc')

% plot just these states 
% this is for the new_f_limit
XLIMS=[0 100]; YLIMS=[-0.50 0.50];
XLIMS=[0 220]; YLIMS=[-0.150 0.150];
for idi=1:length(all_output)
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
                  this_FFT=this_output.FFT_POWER_INT_by_freq{si};
                  FFT_1=[FFT_1 this_FFT(1,:)];
                  FFT_2=[FFT_2 this_FFT(2,:)];
                  FFT_3=[FFT_3 this_FFT(3,:)];
                  FFT_4=[FFT_4 this_FFT(4,:)]; % size should be 24*number_of_states_contains
                  ResponseVarName1=[ResponseVarName1 {this_output.estrus_states{si}}];
                  number_of_states_contains=number_of_states_contains+1;
              end
    end
    figure
    subplot(4,1,1)
    autocorr(double(FFT_1),'NumLags',length(FFT_1)-ceil(0.15*length(FFT_1)));hold on;
    xlim(XLIMS);     ylim(YLIMS)
    title([mouse_info{idi}.sex  ' ' mouse_info{idi}.ID])
    subplot(4,1,2)
    autocorr(double(FFT_2),'NumLags',length(FFT_2)-ceil(0.15*length(FFT_1)));hold on;
    xlim(XLIMS);    ylim(YLIMS)
    subplot(4,1,3)
    autocorr(double(FFT_3),'NumLags',length(FFT_3)-ceil(0.15*length(FFT_1)));hold on;
    xlim(XLIMS);     ylim(YLIMS)
    subplot(4,1,4)
    autocorr(double(FFT_4),'NumLags',length(FFT_4)-ceil(0.15*length(FFT_1)));hold on; 
    xlim(XLIMS); ylim(YLIMS)
end

% gather each state  histogram % results in matrix of #animals x #STATES

 


 k=0;
for idi=1:length(all_output)
    this_output=all_output{idi};
    % go over days and find the days that are relevant 
     number_of_states_contains=0;
     for si=1:length(estrous_states_allclasses) % + male and OVX
         FFT_by_state{idi,si}=[];
%          FFT_by_ID_state{idi}{1}=[];
%          states_by_ID{idi}{1}=[];
     end
     switch mouse_info{idi}.sex
         case 'Female'
            
             for si=1:length(this_output.estrus_states)% over states
                 if contains(this_output.estrus_states{si},estrous_states_allclasses)
                     k=k+1;
                     s_ind=find(strcmp(this_output.estrus_states{si},estrous_states_allclasses));
                     % shift data: original data taken at strating at 9am, while at 13 light
                     % turned off. Data shifted that 1am (Light) will be the start time
%                      tmp_FFT=cat(2,this_output.FFT_POWER_INT_by_freq{si-1}(:,17:24),this_output.FFT_POWER_INT_by_freq{si}(:,1:16));
%                      all_output{idi}.FFT_POWER_INT_by_freq{si}=tmp_FFT;% also return the new matrix to all_output for machine learning  
                      FFT_by_state{idi,s_ind}=[FFT_by_state{idi,s_ind} all_output{idi}.FFT_POWER_INT_by_freq{si}] ;
                      FFT_by_state_all_ID{k}=this_output.FFT_POWER_INT_by_freq{si};
                     states{k}=this_output.estrus_states{si};
                 end
             end
         case {'Male','OVX'}
             for si=1:length(this_output.estrus_states)
                     s_ind=find(strcmp(this_output.estrus_states{si},estrous_states_allclasses));
                      % shift data: original data taken at strating at 9am, while at 13 light
                     % turned off. Data shifted that 1am (Light) will be
                     % the start time- here si-1 is not needed because state is the same 
                    % tmp_FFT=cat(2,this_output.FFT_POWER_INT_by_freq{si}(:,17:24),this_output.FFT_POWER_INT_by_freq{si}(:,1:16));
                     %all_output{idi}.FFT_POWER_INT_by_freq{si}=tmp_FFT; % also return the new matrix to all_output for machine learning  
                     FFT_by_state{idi,s_ind}=[FFT_by_state{idi,s_ind} all_output{idi}.FFT_POWER_INT_by_freq{si}];
                     %FFT_by_state_all_ID{k}=this_output.FFT_POWER_INT_by_freq{s_ind};
                     %states{k}=this_output.estrus_states{si};
             end 
     end
end
% save
save_all_matrix=1;
if save_all_matrix
    save('FFT_matrix.mat','FFT_by_state_all_ID')
    save('states.mat','states')
end
    
% gather information about a specific frequency at a specific hour and look
% at the histogram 

ZT_start=0;    % data was shifted so FFT matrix start at ZT0     
freq_ind=9;
hour_ind=[9:12];
%states_to_compare={'P-1','P+1'}
states_to_compare={'OVX','Male'}
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
            ind1=find(strcmp(this_output.estrus_states,states_to_compare{i}));
            for ini=1:length(ind1)
                state_FFT_byhour_byfreq{hi}{i}=[state_FFT_byhour_byfreq{hi}{i} this_output.FFT_POWER_INT_by_freq{ind1(ini)}(freq_ind,hour_ind(hi))];
            end
        end
        %end
    end
end
% plot histograms 
XLIM=100;
edges=[0:4:200];
marker_colors={'b','r'};
figure
for hi=1:length(hour_ind)
    subplot(1,length(hour_ind),hi)
    for i=1:length(states_to_compare)
        data=state_FFT_byhour_byfreq{hi}{i};
        h(i)=histogram(data,edges); hold on
        dataMean = nanmean(data);
        dataMedian = nanmedian(data);
        xline(dataMedian, 'Color', marker_colors{i}, 'LineWidth', 2);
        disp(['Mean+-sdt FFT amplitudes for ZT' num2str(hour_ind(hi)) ' ' states_to_compare{i} ': ' num2str(dataMean) '+-' num2str(std(data)/sqrt(length(data)))])
    end
    legend(states_to_compare)
    title(['Recording hour: ZT ' num2str(hour_ind(hi))])
    xlim([0 XLIM])
    xlabel('FFT amplitudes')
    axis square
    pd(hi)=pdist2(h(1),h(2),'Euclidean'  ); %  'Euclidean' 
end

% % plot amplitudes at 2D 
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
legend(states_to_compare)
xlim([0 XLIM]); ylim([0 XLIM]); axis square
ylabel(['FFT amplitudes at ZT ' num2str(ZT_start+ hour_ind(hours_to_compare(2)))])
xlabel(['FFT amplitudes at ZT ' num2str(ZT_start+ hour_ind(hours_to_compare(1)))])

COM=sqrt((med_FFT(1,1)-med_FFT(2,1))^2+(med_FFT(1,2)-med_FFT(2,2))^2);
com_sem=sqrt((sem_FFT(1,1)-sem_FFT(2,1))^2+(sem_FFT(1,2)-sem_FFT(2,2))^2);
disp([states_to_compare{1} ' to ' states_to_compare{2} ' Center of mass distance: ' num2str(COM) '+-' num2str(com_sem)])

% average data by state - females
females_FFT_by_state=FFT_by_state(1:n_females,1:numel(estrous_states_classes));
for si=1:size(females_FFT_by_state,2)
    for idi=1:size(females_FFT_by_state,1)
        this_state_FFT=females_FFT_by_state{idi,si};
        FFT_temp=reshape(this_state_FFT,size(this_state_FFT,1),24,size(this_state_FFT,2)/24);
        FFT_median_by_ID{si}(idi,:,:)=nanmedian(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
        FFT_mean_by_ID{si}(idi,:,:)=nanmean(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
        FFT_mean_by_ID_sem{si}(idi,:,:)=std(FFT_temp,0,3)./sqrt(size(FFT_temp,3)); 
    end
    FFT_median_by_state{si}(:,:)=nanmedian(FFT_median_by_ID{si},1);
end


% males
males_FFT_by_state=FFT_by_state(n_females+1:n_females+n_males,6);
for si=1:size(males_FFT_by_state,2)
    for idi=1:size(males_FFT_by_state,1)
        this_state_FFT=males_FFT_by_state{idi,si};
        FFT_temp=reshape(this_state_FFT,size(this_state_FFT,1),24,size(this_state_FFT,2)/24);
        FFT_median_by_ID{si}(idi,:,:)=nanmedian(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
        FFT_mean_by_ID{si}(idi,:,:)=nanmean(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
        FFT_mean_by_ID_sem{si}(idi,:,:)=std(FFT_temp,0,3)./sqrt(size(FFT_temp,3)); 
    end
    FFT_median_male{si}(:,:)=nanmedian(FFT_median_by_ID{si},1);
end
% OVX
OVX_FFT_by_state=FFT_by_state(n_females+n_males+1:n_females+n_males+n_OVX,7);
for si=1:size(OVX_FFT_by_state,2)
    for idi=1:size(OVX_FFT_by_state,1)
        this_state_FFT=OVX_FFT_by_state{idi,si};
        FFT_temp=reshape(this_state_FFT,size(this_state_FFT,1),24,size(this_state_FFT,2)/24);
        FFT_median_by_ID{si}(idi,:,:)=nanmedian(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
        FFT_mean_by_ID{si}(idi,:,:)=nanmean(FFT_temp,3); % indexes: si- estrus state, idi- animal ID, 9 frequencies bins, 24 hours
        FFT_mean_by_ID_sem{si}(idi,:,:)=std(FFT_temp,0,3)./sqrt(size(FFT_temp,3)); 
    end
    FFT_median_OVX{si}(:,:)=nanmedian(FFT_median_by_ID{si},1);
end

% now plot avergaed FFTs by state
freq_to_plot=[4:9];
clims=[0 55];
figure 
for si=1:length(FFT_median_by_state)
    subplot(length(FFT_median_by_state)+2,1,si)
    imagesc(FFT_median_by_state{si}(freq_to_plot,:),clims)
    colorbar
    title(estrous_states_allclasses{si})
    yticks([1: size(FFT_median_by_state{si},1)])
    yticklabels(num2str(new_f_limits(freq_to_plot,:)))
    xlabel('Time (hours)')
    ylabel('median Int. Power fft')
end
% plot male and OVX
data_to_plot={'FFT_median_male', 'FFT_median_OVX'};
for nsi=1:numel(data_to_plot)
    subplot(length(FFT_median_by_state)+2,1,si+nsi)
    eval(['data=' data_to_plot{nsi} ';']);
    imagesc(data{1}(freq_to_plot,:),clims)
    colorbar
    title(estrous_states_allclasses{si+nsi})
    yticks([1: size(data{1},1)])
    yticklabels(num2str(new_f_limits(freq_to_plot,:)))
    xlabel('Time (hours)')
    ylabel('median Int. Power fft')
end

%print('FP_FFT_compare accross_states','-depsc')


%  plot frequencies relationships, per state
k=0;
ref_freq=9;
hours_array=1:4;
figure
for fi=[1 3 4 6 8]
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
    ph2=plot(FFT_median_male{1}(ref_freq,hours_array),FFT_median_male{1}(fi,hours_array),'*')   ; hold on
    set(ph2, 'MarkerEdgeColor', [0.7 0.7 0.7], 'Marker', '*')
    ph3=plot(FFT_median_OVX{1}(ref_freq,hours_array),FFT_median_OVX{1}(fi,hours_array),'*')   ; hold on
    set(ph3, 'MarkerEdgeColor', [0.3 0.3 0.3], 'Marker', '+')
    title(['Freq' num2str(ref_freq) ' to ' num2str(fi)])
    xlabel(['Median freq ' num2str(ref_freq)])
    ylabel(['Median freq ' num2str(fi)])
end
% now plot frequecies relationships per animal, normalized 
figure
N_mice=n_females;
%N_mice=length(all_output)
for idi=1:N_mice %
    subplot(2,N_mice/2,idi)
    for si=1:length(FFT_median_by_state)
        tmp1(:)=FFT_mean_by_ID{si}(idi,1,:);
        tmp1=tmp1-min(tmp1); M1=max(tmp1); tmp1=tmp1/M1;% normalization 
        tmp2(:)=FFT_mean_by_ID{si}(idi,2,:);
        tmp2=tmp2-min(tmp2); M2=max(tmp2); tmp2=tmp2/M2;% normalization 
        yneg(:)=FFT_mean_by_ID_sem{si}(idi,3,:)/M2;
        ypos(:)=FFT_mean_by_ID_sem{si}(idi,3,:)/M2;
        xneg(:)=FFT_mean_by_ID_sem{si}(idi,1,:)/M1;
        xpos(:)=FFT_mean_by_ID_sem{si}(idi,1,:)/M1;
        %plot(tmp1,tmp2,'-*'); hold on;
        errorbar(tmp1,tmp2,yneg,ypos,xneg,xpos,'-o')
    end
    xlim([0 1.2])
    ylim([0 1.2])
    title(mouse_info{idi}.ID)
end
%% machine learning:  define test and training sets.
% first can run one specific condition 
one_condition=0;
if one_condition
    all_inputs.freq_start_interval=4;
    all_inputs.FX=4; % additional frequencies range taken into account
    all_inputs.rel_hours=[1:24];
    all_inputs.pca_features=1;% choose if pca is used to reduce dimentionality of features
    all_inputs.selected_states=estrous_states_allclasses([6,7]);
   
     all_inputs.method='Discriminant analysis';
   %all_inputs.method='Support vector machine'
    all_inputs.mouse_info=mouse_info;
    all_inputs.plot_figure=0;
    end_table=all_machine_learning_for_FFT(all_inputs,all_output);
end

% next, can run multiple conditions, comparing 2 states at the time  
% define input data conditions 
%class_by_estrous='full' % 'full' or 'receptive'
class_by_estrous='receptive' % 'full' or 'receptive'
switch class_by_estrous
    case 'full'; by_estrous=estrous_states_allclasses ;
    case 'receptive'; by_estrous=estrous_states_for_classification;
end
switch class_by_estrous
    case 'receptive' ; all_inputs.classification_states='receptive'; %/ 'estrus_states'
    case 'full' ; all_inputs.classification_states='estrus_states'; %/ ''
end
all_inputs.freq_start_interval=4;
all_inputs.FX=9-all_inputs.freq_start_interval; % additional frequencies range taken into account
all_inputs.rel_hours=[7:12];
all_inputs.pca_features=0;% choose if pca is used to reduce dimentionality of features
%all_inputs.method='Discriminant analysis';
all_inputs.method='Support vector machine'
all_inputs.plot_figure=0;
n_iteration=5;
cd('Z:\Anat\fiberphotometry\FFT_Analysis')
% check if file exist 
if exist (['FP_FFT_classification_corr_'  num2str(n_iteration) '_iterations_' all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', hours ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ' ' all_inputs.classification_states '.mat'])
    load(['FP_FFT_classification_corr_'  num2str(n_iteration) '_iterations_' all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', hours ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ' ' all_inputs.classification_states '.mat'])
else
    clear LOO_score cr2 cr1 cr2_std
    %LOO_score_legend=[];
    for ni=1:n_iteration
        for sti1=1:length(by_estrous)
            for sti2=1:length(by_estrous)
                all_inputs.selected_states=by_estrous([sti1,sti2]);
                all_inputs.mouse_info=mouse_info;
                end_table=all_machine_learning_for_FFT(all_inputs,all_output);
                LOO_score{sti1}(:,sti2)=table2array(end_table(:,2));
                % LOO_score_legend=[LOO_score_legend end_table.states];
            end
        end
        
        for sti1=1:length(LOO_score)
            data_to_plot=LOO_score{sti1}';
            cr1(ni,sti1,:)=mean(data_to_plot,2);
        end
    end
    
    cr2(:,:)=nanmean(cr1,1);
    cr2_std(:,:)=std(cr1,1)/sqrt(n_iteration);
    disp(['Mean std = ' num2str(mean(nanmean(cr2_std)))])
    save (['FP_FFT_classification_corr_' num2str(n_iteration) '_iterations_' all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', hours ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ' ' all_inputs.classification_states '.mat'],'cr2')
end
% plot classification map 
map='gray';
corr_plot_FFT(cr2/100,by_estrous,map)
print(['FP_FFT_classification_corr_' all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', hours ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ' ' all_inputs.classification_states],'-depsc')

title([all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', hours ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ])
disp(['Minimal validation score = ' num2str(min(min(cr2)))])
disp(['Average validation score = ' num2str(mean(mean(cr2)))])

disp(['Average above chance = ' num2str(mean(cr2(find((cr2>55).*(cr2<99))))) ' +- ' num2str(std(cr2(find((cr2>55).*(cr2<99))))/sqrt(length(cr2(find((cr2>55).*(cr2<99))))))])
disp(['Average bellow chance = ' num2str(nanmean(cr2(find(cr2<50)))) ' +- ' num2str(std(cr2(find(cr2<55)))/sqrt(length(cr2(find(cr2<55)))))])

% square plot
my_matvisual(cr2(end:-1:1,:),'annotation',by_estrous)
title([all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', time-frames ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ])
print(['FP_FFT_classification_corr__' all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', hours ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ' ' all_inputs.classification_states],'-depsc')

print(['FP_FFT_classification_corr2_' all_inputs.method ', PCA = ' num2str(all_inputs.pca_features) ', hours ' num2str(all_inputs.rel_hours(1)) ' to ' num2str(all_inputs.rel_hours(end)) ', Freq ' num2str(all_inputs.freq_start_interval) ' to ' num2str(all_inputs.freq_start_interval+all_inputs.FX) ' ' all_inputs.classification_states],'-depsc')
1