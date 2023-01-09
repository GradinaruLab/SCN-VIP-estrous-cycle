function end_table= all_machine_learning_for_FFT(all_inputs, all_output)
% used in FP_FFT_output_for classifier
% for 24 h Fiberphotometry analysis
% uses classifier_fft_FP
% Anat Kahan April 2022
freq_start_interval=all_inputs.freq_start_interval;
FX=all_inputs.FX; % additional frequencies range taken into account
rel_hours=all_inputs.rel_hours;
selected_states=all_inputs.selected_states;
pca_features=all_inputs.pca_features;% choose if pca is used to reduce dimentionality of features
%method='Discriminant analysis';
method=all_inputs.method;
mouse_info=all_inputs.mouse_info;
plot_figure=all_inputs.plot_figure;
classification_states=all_inputs.classification_states;

end_table=table();
selected_states_title=[];
for esi=1:numel(selected_states)
    selected_states_title=[selected_states_title  ' ' selected_states{esi}];
end
 
for ffi=freq_start_interval % loop is actually not needed
    clear training_set sex estrous_state ID score true_positive_per
    rel_freq=ffi:ffi+FX;% choose just the first FX+1 frequencies integrals ranges
    disp(['Freq index from '  num2str(ffi) ' to ' num2str(ffi+FX)])
    t_table=table();
    m=0;
    full_data_set_size=numel(all_output);
    shuffledArray = randperm(full_data_set_size);
    for ki=1:numel(all_output)%-1 % remove one animal each time to check results stability
        %ti=shuffledArray(ki);
        ti=ki;
        output=all_output{ti};
        %     train_set_inds=[1: numel(all_output)];
        %     train_set_inds=train_set_inds(train_set_inds~=test_ind);
        for si=1:numel(output.FFT_POWER_INT_by_freq)
            switch classification_states
                case {'estrus_states','just_estrus_states'}
                    this_estrous_state=output.estrus_states{si};
                case 'receptive'
                    this_estrous_state=output.new_estrus_states{si};
            end
            if sum(strcmp(this_estrous_state,selected_states))>0
                m=m+1;
                sex{m}=mouse_info{ti}.sex;
                ID{m}=mouse_info{ti}.ID;
                estrous_state{m}=this_estrous_state;
                training_set{m}=output.FFT_POWER_INT_by_freq{si}(rel_freq,rel_hours);
            end
        end
    end
    t_table.ID=ID';
    t_table.sex=sex';
    t_table.estrous_state=estrous_state';
    t_table.training_set=training_set';
    % clean the t_table from training_sets that are all NaN
    good_ind=[];
    for ti=1:numel(training_set)
        if sum(sum(isnan(training_set{ti})))<numel(training_set{ti})
            good_ind=[good_ind ti];
        else
            disp('all-NaN set was removed')
        end
    end
    t_table=t_table(good_ind,:);
    if numel(t_table)<8
        disp('array is too small')
    end
    %%% goal- to pickup one element from the training set to be tested
    % define 'output' which is for the training set
    clear params mylabels
    for m=1:size(t_table,1)
        % test data:
        test_set=t_table.training_set(m);
         
        % arrange data for classification - training data:
        temp=1:size(t_table,1);
        training_set=t_table.training_set(temp(temp~=m));% excludes the test index
        estrous_state=t_table.estrous_state(temp(temp~=m));
               
        % training set should be evenly contributed from each state,
        % therefore randomaly remove extra sessions
        [training_set, estrous_state]=remove_states_for_classification(training_set,estrous_state);
 
        
        % check that arrays size are correct 
      %  disp([num2str(sum(strcmp(estrous_state,all_inputs.selected_states{1}))) ' ' all_inputs.selected_states{1} ', ' ... 
       %  num2str(sum(strcmp(estrous_state,all_inputs.selected_states{2}))) ' ' all_inputs.selected_states{2}])
        if sum(strcmp(estrous_state,all_inputs.selected_states{1}))-sum(strcmp(estrous_state,all_inputs.selected_states{2}))~=0; disp ('Check classifier array size in ''all_machine_learning_for_FFT'''); end
        
        params.pca_features=pca_features;% choose if pca is used to reduce dimentionality of features
        params.method=method;
       %params.states=t_table.estrous_state(temp(temp~=m));
        params.states=estrous_state;
        params.test_state=t_table.estrous_state(m);
        params.rel_freq=rel_freq;
        params.rel_hours=rel_hours;
           %params.estrous_states_allclasses=estrous_states_allclasses;
        
        % get training set and test set as input. export the success rate
        [mylabels(m)]=classifier_fft_FP(training_set, test_set, params);
    end
    
    t_table.score=mylabels'; 
    
    end_table.states=[selected_states'; 'all'];
    
    % a check point- making sure the matrix is not nan due to missing data 
    nan_ind=find(strcmp(mylabels,'nan')); 
    disp(['Total '  num2str(length(nan_ind)) ' all nans'])
    non_nan_ind=find(~strcmp(mylabels,'nan'));
    original_size=size(t_table,1);
    t_table=t_table(non_nan_ind,:);
    new_size=size(t_table,1);
    disp(['Original table size is  ' num2str(original_size) ' , new size is ' num2str(new_size)])
    
    for si=1:numel(end_table.states)    
        true_positive(si,:)=(strcmp(t_table.estrous_state,end_table.states{si})).*(strcmp(t_table.score,end_table.states{si}));
        true_positive_per(si)=100*sum(true_positive(si,:))/sum(strcmp(t_table.score,end_table.states{si}));
        % a check point- check the classifier was it results nan 
        if sum(strcmp(t_table.score,end_table.states{si}))==0
            disp(['For states ' end_table.states{1} ' and ' end_table.states{2} ' no matching was found- did very poor!!'])
        end
    end
    % true positive out of both states
    true_positive_per(numel(end_table.states))=100*sum(sum(true_positive))/size(true_positive,2);
    % put all in one table 
    eval(['end_table.true_positive_per' num2str(ffi) '=true_positive_per'' ;'])
    eval(['end_table.true_positive_per' num2str(ffi) '=true_positive_per'' ;'])
    
end
disp(end_table)

if plot_figure
    figure
    for ffi=freq_start_interval
        eval(['plot(ffi*ones(1,size(end_table,1)),end_table.true_positive_per' num2str(ffi) ',''*'')'])
        hold on
    end
    for ffi=freq_start_interval
        My_xticklabels=[freq_start_interval; freq_start_interval+4];
    end
    xlim([freq_start_interval(1)-0.5 freq_start_interval(end)+0.5])
    xlabel('Frequencies intervals')
    ylabel('State classification test - true positive scores')
    set(gca,'XTick',freq_start_interval)
    set(gca,'XTickLabel',num2str(My_xticklabels'))
    ylim([15,102])
    
    title([params.method ', PCA = ' num2str(pca_features) ', ' selected_states_title ',hours ' num2str(rel_hours(1)) ' to ' num2str(rel_hours(end)) ])
end
