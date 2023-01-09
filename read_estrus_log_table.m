function [y]=read_estrus_log_table
%this function reads the estrus log file and show the estrus cycle for each
%animal and show few other parameters
%   Detailed explanation goes here
% exp_num is the experiment number
% a
% AFTER running this function, use 'estrous_analysis(y)' for farther analysis 
% PAY ATTANTION that the animal ID should be txt!!

% read the data from the file. remember to save the lateset Drive Sheet to
% the computer!
%exp='Esr1OK'
%exp='GnRHcas9_KD'
%exp='VIP ablation'
%exp='DD to DL' % VIP compare a few conditions 
exp='DD to DL2'%without  432RL
%exp='VIPChR2'
%exp='VIPChR2_no30_35'
%exp='VIP_DREADD';
%exp='DD to DL3' % VIP and DREADD- compare baseline to rescue
%exp='C57'
%exp='WT jet lag'
switch exp
    case 'VIP ablation'; file_name='Estrous cycle cytology ablation AK2';  exp_num=120;
    case 'Esr1OK'        ;file_name='Estrous cycle cytology Esr1KO AK';    exp_num=121;
    case 'GnRHcas9_KD' ; file_name='GnRHcas9';                              exp_num=126; n_Groups=3;
    case 'VIPChR2';      file_name='VIPChR2';                               exp_num=23;
    case 'VIPChR2_no30_35'; file_name='VIPChR2_no30_35';                   exp_num=23
    case 'C57';         file_name='C57 DL to DD with L';                    exp_num=24
       % DREADD - 13 females total. 6 ctrl. 7 exp. 4 excluded due to not showing any response to light
    case 'VIP_DREADD';  file_name='VIP_DREADD';                            exp_num=25; n_Groups=2;
    case 'DD to DL';    file_name='VIP DL to DD';                          exp_num=122
    case 'DD to DL2'  ;  file_name='VIP DL to DD3';                         exp_num=122; n_Groups=1;
    case 'DD to DL3' ;  file_name='VIPCHR2_DREADD DL to DD';  exp_num=125% estrus_analysis_Table- doesn't run well yet
    case 'WT jet lag';   file_name=exp;                        exp_num=4;
end
%path='C:\Users\anatk\Documents\Data_Glab_home_work\Behavioral\data\'
path='D:\DATA_Glab\Behavioral\estrous_cycle_data\';
full_path=[path file_name '.xlsx'];
%[NUM,TXT,RAW]=xlsread(full_path,'VS AK');
[NUM,TXT,RAW]=xlsread(full_path,'VS');
% TXT is the one in which all the data is in 
T=readtable(full_path);

full_EVENTS=TXT(:,1); % creats event array 
% find the strat and end points of a specific experiment, for example, 'Exp1',
si=0;
START_array=strfind(full_EVENTS,['Exp' num2str(exp_num)]);
STOP_array=strfind(full_EVENTS,['Exp' num2str(exp_num+1)]);
for i=1:length(START_array)
    if ~isempty(START_array{i})
        START=i;
    end
    if ~isempty(STOP_array{i})
        STOP=i;
        si=si+1;
    end
   
end
if si==0
    STOP=size(TXT,1);
end

% find the dates
clear DATES
%DATES=TXT(START+3:STOP-1,2);
DATES=TXT(START+2:STOP-0,2);
% find the events
%EVENTS=full_EVENTS(START+3:STOP-1); 
EVENTS=full_EVENTS(START+2:STOP-0); 
% find the states of each individual, and makes an array of strains and ID 
n=0;
for k=4:size(TXT,2)
    if ~isempty(TXT{START,k}) % only if there is ID
    n=n+1;
    STATES{n}=TXT(START+2:STOP,k);
    %STATES{n}=TXT(START+3:STOP-1,k);
    IDS{n}=TXT(START+1,k);
    STRAINS{n}=TXT(START,k);
    end
end

% change the state array to 0-1-2 array, to easily visualize it 
for n=1:length(IDS) % n is the animal 
    temp_states=STATES{n};
    temp_state_array=est_state2num(temp_states);
    STATE_array(n,:)=temp_state_array;
    STRAIN_ID{n} =[STRAINS{n}{1} ' ' IDS{n}{1}]; % combine the strain and the ID for the figure
end

% create an event start and stop identifier
exp_type=0; % initial value
for ei=1:length(EVENTS)
    this_event=EVENTS{ei};
    switch this_event
        case {[]};             event_array(ei)=0;
        case {'habituation'};  event_array(ei)=1;
        case{'LASER on'};      event_array(ei)=2;     exp_type=1;
        case {'9pm-9am'} ;     event_array(ei)=4; ;   exp_type=2;% Jet lag control
        case {'JL 9pm-9am', 'JL 3pm-3am','JL 9pm-9am ','JL 9am-9pm', ' JL 9am-9pm', 'JL 3am-3pm'}; event_array(ei)=5;   exp_type=2;% Jet lag exp
        case 'before_injection';  event_array(ei)=1;
        case 'after_injection';  event_array(ei)=2;
        case{ 'preablation '};  event_array(ei)=4;   exp_type=3;
        case {'postablation'};  event_array(ei)=5;   exp_type=3;
        case ('DL12_12');       event_array(ei)=4;   exp_type=4;
        case ('DD');           event_array(ei)=5;    exp_type=4;
        case('DD+30minL');      event_array(ei)=6;   exp_type=4;
        case('DD+30min_light8pm_light6pm');  event_array(ei)=7;     exp_type=4;
        case('DD+30min_light8pm_light12am')  ;event_array(ei)=8;     exp_type=4;          
        case('DD+30min_lightZT12_lightZT22')  ;event_array(ei)=14;     exp_type=5;
        case('DD+30min_lightZT12_laserZT22');   event_array(ei)=9;  exp_type=5;
        case('DD+30min_lightZT12_laserZT16');   event_array(ei)=10; exp_type=5;
        case('manipulation');            event_array(ei)=11;      exp_type=6;
        case('manipulation_later');      event_array(ei)=12;      exp_type=6;
        case('manipulation_later2');     event_array(ei)=13;      exp_type=6;
    end
end



% finds in the array the start and stop indexes of events
hab_start=min(find (event_array==1));
hab_stop=max(find (event_array==1));
laser_start=min(find (event_array==2));
laser_stop=max(find (event_array==2));

%%% put everything to y
y.exp_type=exp_type;
y.event_array=event_array;
y.STATES=STATES;
y.STATE_array=STATE_array;
y.STRAIN_ID=STRAIN_ID;
y.STRAINS=STRAINS;
y.exp_num=exp_num;
y.full_path=full_path;
y.T=T;


% now plot it
switch exp
    case 'WT jet lag'
        STATE_array=STATE_array(:,2:end);
        XLIM=size(STATE_array,2);
        g_limit=[1:floor(XLIM/2);floor(XLIM/2)+1:floor(XLIM)];
        %XLIM=46;
        yticks_label_array={'M/D', 'P', 'E', ' ' };
        for gi=1:size(g_limit,1)
            figure('Name',['Exp' num2str(exp_num) ' vaginal smears ' num2str(gi) ],'NumberTitle','off');
            % plot all the females
            for ni=1:length(IDS)
                subplot (length(IDS)+1,1,ni)
                plot(STATE_array(ni,g_limit(gi,:)),'-*')
                set(gca, 'YTickLabel' ,yticks_label_array, 'ylim' , [0,3],'xlim',[0,XLIM/2])
                title (STRAIN_ID{ni})
            end
            % plot the event
            % subplot(length(IDS)+1,1,ni+1)
        end
    otherwise
        XLIM=size(STATE_array,2);
        %XLIM=46;
        yticks_label_array={'M/D', 'P', 'E', ' ' };
        figure('Name',['Exp' num2str(exp_num) ' vaginal smears' ],'NumberTitle','off');
        % plot all the females
        for ni=1:length(IDS)
            subplot (length(IDS)+1,1,ni)
            plot(STATE_array(ni,:),'-*')
            set(gca, 'YTickLabel' ,yticks_label_array, 'ylim' , [0,3],'xlim',[0,XLIM])
            title (STRAIN_ID{ni})
        end
        % plot the event
        subplot(length(IDS)+2,1,ni+1)
end


switch exp_type
    case 1 % laser 
        ph=line([hab_start,hab_stop+1],[1,1]);% habituation
        set(ph, 'linewidth',14, 'color',[0.82 0.82 0.82])
        hold on
        ph2=line([laser_start,laser_stop+1],[1,1]);% laser on
        set(ph2, 'linewidth',14, 'color',[0 0 1])
        set(gca,'xlim',[0,XLIM], 'ylim', [.5,1.5],'YTick',[], 'Color', [0.95 0.95 0.95])
%     case 2 % L/D manipulation
%         for ei=1:length(event_array)
%             if event_array(ei)==5
%             ph2=line([ei-0.2,ei+0.2],[1,1]);
%             set(ph2, 'linewidth',14, 'color',[0.5 0.5 0.5])
%             hold on
%             end
%         end
   % case {[]}
        
end


xlabel('Days')


colors=[0.8500 0.3250 0.0980; 0.8 0.8 0.8; 0.6 0.6 0.6; 0.4 0.4 0.4];
colors2=[0.8500 0.3250 0.0980; 0.8500 0.3250 0.0980; 0.8 0.8 0.8;0.8 0.8 0.8; 0.6 0.6 0.6; 0.6 0.6 0.6; 0.4 0.4 0.4; 0.4 0.4 0.4];
ablT=y.T;
switch exp_type
    case 2 % jet-lag. 
        labels_str={'Ctrl','Exp'};
        JLT_exp=ablT(~strcmp(ablT.Exp4,'9pm-9am'),:); % jet-lag
        JLT_ctrl=ablT(strcmp(ablT.Exp4,'9pm-9am'),:);% LD
        X=repmat({'JL'},numel( JLT_exp.Exp4),1);
        JLT_exp.Exp4=X;
        % create a new table ; change names of states
        new_T=[JLT_ctrl(:,[1,4:end]);JLT_exp(:,[1,4:end])];
        
        for i=2:numel(new_T.Properties.VariableNames) % change Pro to P ect.
            name=new_T.Properties.VariableNames{i};
            eval(['new_T.' name '(strcmp(new_T.' name ',''Pro'')) = {''P''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Est'')) = {''E''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Met'')) = {''M''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Di'')) = {''D''};']);
        end
        
        newT_exp=new_T(~strcmp(new_T.Exp4,'9pm-9am'),:);
        newT_ctrl=new_T(strcmp(new_T.Exp4,'9pm-9am'),:);
        
        if size(newT_ctrl,1)>size(newT_exp,1)
            newT_ctrl=newT_ctrl(1:size(newT_exp,1),:);
            new_T=[newT_ctrl; newT_exp];
        end
        
        clear summary_T
        % ctrl
        ctrl_times_P=[];ctrl_times_E=[];ctrl_times_M=[];ctrl_times_D=[];
        for i=2:numel(new_T.Properties.VariableNames) % change Pro to P etc.
            name=new_T.Properties.VariableNames{i};
            eval(['ctrl_times_P=[ctrl_times_P sum(strcmp(newT_ctrl.' name ',''P''))];']);
            eval(['ctrl_times_E=[ctrl_times_E sum(strcmp(newT_ctrl.' name ',''E''))];']);
            eval(['ctrl_times_M=[ctrl_times_M sum(strcmp(newT_ctrl.' name ',''M''))];']);
            eval(['ctrl_times_D=[ctrl_times_D sum(strcmp(newT_ctrl.' name ',''D''))];']);
        end
        
        % exp
        exp_times_P=[];exp_times_E=[];exp_times_M=[];exp_times_D=[];
        for i=2:numel(new_T.Properties.VariableNames) % change Pro to P ect.
            name=new_T.Properties.VariableNames{i};
            eval(['exp_times_P=[exp_times_P sum(strcmp(newT_exp.' name ',''P''))];']);
            eval(['exp_times_E=[exp_times_E sum(strcmp(newT_exp.' name ',''E''))];']);
            eval(['exp_times_M=[exp_times_M sum(strcmp(newT_exp.' name ',''M''))];']);
            eval(['exp_times_D =[ exp_times_D sum(strcmp(newT_exp.' name ',''D''))];']);
        end
        x=[ctrl_times_P exp_times_P ctrl_times_E exp_times_E ctrl_times_M exp_times_M ctrl_times_D exp_times_D];
        g=[ones(1,length(ctrl_times_P)) 2*ones(1,length(exp_times_P)) 3*ones(1,length(ctrl_times_E)) 4*ones(1,length(exp_times_E)) 5*ones(1,length(ctrl_times_M)) 6*ones(1,length(exp_times_M)) 7*ones(1,length(ctrl_times_D)) 8*ones(1,length(exp_times_D))];
        
        figure
        subplot(1,6,1)
        h=pie([sum(ctrl_times_P) sum(ctrl_times_E) sum(ctrl_times_M) sum(ctrl_times_D)]);
        for k=1:2:length(h)
            set(h(k), 'FaceColor', colors2(k,:))
        end
        title(labels_str{1})
        subplot(1,6,2)
        h= pie([sum(exp_times_P) sum(exp_times_E) sum(exp_times_M) sum(exp_times_D)]);
        title(labels_str{2})
        for k=1:2:length(h)
            set(h(k), 'FaceColor', colors2(k,:))
        end
        set(gca,'YTickLabel',{'P'})
        
        
        subplot(1,6,[3 4])
        state_dist=[sum(ctrl_times_P) sum(ctrl_times_E) sum(ctrl_times_M) sum(ctrl_times_D);sum(exp_times_P) sum(exp_times_E) sum(exp_times_M) sum(exp_times_D)];
        state_dist_per=(state_dist'./sum(state_dist')*100)';
        bh2=bar(state_dist_per,'stacked','DisplayName','state_dist');
        set(gca,'XTick',1:size(state_dist,1));
        set(gca,'XTickLabel',labels_str);
        for k=1:length(bh2)
            set(bh2(k), 'FaceColor', colors(k,:))
        end
        ylabel('States distribution')
        legend({'P' 'E' 'M' 'D'})
        ylim([0 105])
        
        subplot(1,6,[5 6])
        boxplot(x,g, ...
            'Labels', {'P ctrl','P exp','E ctrl','E exp','M ctrl','M exp','D ctrl','D exp'}, ...
            'Colors',colors2,'PlotStyle','compact'); hold on
        
        [pP,hP] = kstest2(ctrl_times_P,exp_times_P);% two-sample Kolmogorov-Smirnov test
        [pE,hE] = kstest2(ctrl_times_E,exp_times_E);
        [pM,hM] = kstest2(ctrl_times_M,exp_times_M);
        [pD,hD] = kstest2(ctrl_times_D,exp_times_D);
        
        disp(['P changed from ' num2str(mean(ctrl_times_P)) '+-' num2str(std(ctrl_times_P)/sqrt(length(ctrl_times_P)))...
            ' to ' num2str(mean(exp_times_P)) '+-' num2str(std(exp_times_P)/sqrt(length(exp_times_P)))]);
        
        disp(['E changed from ' num2str(mean(ctrl_times_E)) '+-' num2str(std(ctrl_times_E)/sqrt(length(ctrl_times_E)))...
            ' to ' num2str(mean(exp_times_E)) '+-' num2str(std(exp_times_E)/sqrt(length(exp_times_E)))]);
        
        
        cd('Z:\Anat\behavioral\data')
        
        print('Jet Lag summary','-depsc')
        
    case 3 % vip ablation. or any experiment with 2 groups, before and after manipulation 
        n_conditions=2;% two conditions. such as before injection  
        n_groups=2; % two group, experimental and control 
        conditions_names={'preablation','postablation'};
        groups_names={'Control','Casp3'};
        labels_str={'Ctrl C0','Ctrl C1','Exp C0','Exp C1'};
        
        for ni=1:n_conditions
            eval(['C' num2str(ni-1) '=ablT(strcmp(ablT.Exp120,conditions_names{ni}),:);']); %
        end
        
        % create a new table ;
        new_T=[C1(:,[1,4:end]);C0(:,[1,4:end])];
        % change names of states
        for i=2:numel(new_T.Properties.VariableNames) % change Pro to P ect.
            name=new_T.Properties.VariableNames{i};
            eval(['new_T.' name '(strcmp(new_T.' name ',''Pro'')) = {''P''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Est'')) = {''E''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''E-M'')) = {''E''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Met'')) = {''M''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Di'')) = {''D''};']);
        end
        % find the indexes for experimental and control 
        for gi=1:n_groups
            eval(['G' num2str(gi-1) '_inds=find(~cellfun(@isempty,strfind(new_T.Properties.VariableNames,groups_names{gi})));']);
        end

        for ni=1:n_conditions
            eval(['C' num2str(ni-1) '=new_T(strcmp(new_T.Exp120,conditions_names{ni}),:);'])
            eval(['G1_C' num2str(ni-1) '=C' num2str(ni-1) '(:,G1_inds);']);
            eval(['G0_C' num2str(ni-1) '=C' num2str(ni-1) '(:,G0_inds);']);
        end
        
        % ctrl  and experimental % get number of occurance of each state in the control 
        % group, for each condition. C - condition. G - group 
       
        for gi=1:n_groups
            for ni=1:n_conditions
                eval(['nei=numel(G'  num2str(gi-1) '_C' num2str(ni-1) '.Properties.VariableNames);']); %
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_P=[];']);
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_E=[];']);
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_M=[];']);
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_D=[];']);
                for i=1:nei
                    eval(['name=G'  num2str(gi-1) '_C'  num2str(ni-1) '.Properties.VariableNames{i};']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_P=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_P sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''P''))];']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_E=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_E sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''E''))];']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_M=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_M sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''M''))];']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_D=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_D sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''D''))];']);
                end
            end
        end
        
        for ni=1:n_conditions
            eval(['x' num2str(ni-1) '=[G0_C' num2str(ni-1) '_times_P G1_C' num2str(ni-1) '_times_P G0_C' num2str(ni-1) '_times_E G1_C' num2str(ni-1) '_times_E G0_C' num2str(ni-1) '_times_M G1_C' num2str(ni-1) '_times_M G0_C' num2str(ni-1) '_times_D G1_C' num2str(ni-1) '_times_D];']);
            eval(['g' num2str(ni-1) '=[ones(1,length(G0_C' num2str(ni-1) '_times_P)) 2*ones(1,length(G1_C' num2str(ni-1) '_times_P)) 3*ones(1,length(G0_C' num2str(ni-1) '_times_E)) 4*ones(1,length(G1_C' num2str(ni-1) '_times_E)) 5*ones(1,length(G0_C' num2str(ni-1) '_times_M)) 6*ones(1,length(G1_C' num2str(ni-1) '_times_M)) 7*ones(1,length(G0_C' num2str(ni-1) '_times_D)) 8*ones(1,length(G1_C' num2str(ni-1) '_times_D))];']);
        end
        
        figure
        p=0;
        for gi=1:n_groups
            for ni=1:n_conditions
                p=p+1;
                subplot(3,n_conditions*n_groups,p)
                eval(['data_to_plot=[sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_P) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_E) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_M) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_D)];']);
                h=pie(data_to_plot);
                for k=1:2:length(h)
                    set(h(k), 'FaceColor', colors2(k,:))
                end
                title(labels_str{p})
            end
        end
        
       figure      
       %subplot(3,4,[5:8])
        state_dist=[];
        for gi=n_groups:-1:1 % have to do reversed to get the order which fits the upper panel 
            for ni=n_conditions:-1:1
                %state_dist=[sum(G0_C0_times_P) sum(G0_C0_times_E) sum(G0_C0_times_M) sum(G0_C0_times_D);sum(G1_C0_times_P) sum(G1_C0_times_E) sum(G1_C0_times_M) sum(G1_C0_times_D)];
                eval(['state_dist=[sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_P) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_E) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_M) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_D);state_dist ];']);
            end
        end
        state_dist_per=(state_dist'./sum(state_dist')*100)';
        
        bh2=bar(state_dist_per,'stacked','DisplayName','state_dist');
        set(gca,'XTick',1:size(state_dist,1));
        set(gca,'XTickLabel',labels_str);
        for k=1:length(bh2)
            set(bh2(k), 'FaceColor', colors(k,:))
        end
        ylabel('States distribution')
        legend({'P' 'E' 'M' 'D'})
        ylim([0 105])
        
        figure
        %subplot(3,4,[9:10])
        subplot(2,1,1)
        boxplot(x0,g0, ...
            'Labels', {'P C0 ctrl','P C0 exp','E C0 ctrl','E C0 exp','M C0 ctrl','M C0 exp','D C0 ctrl','D C0 exp'}, ...
            'Colors',colors2,'PlotStyle','compact'); hold on
        ylim([0 15])
        
        %subplot(3,4,[11 12])
        subplot(2,1,2)
        boxplot(x1,g1, ...
            'Labels', {'P C1 ctrl','P C1 exp','E C1 ctrl','E C1 exp','M C1 ctrl','M C1 exp','D C1 ctrl','D C1 exp'}, ...
            'Colors',colors2,'PlotStyle','compact'); hold on
        ylim([0 15])
        
        % check normality of the data to decide on statistical test 
         H_normalityP=[];H_normalityE=[];H_normalityM=[];H_normalityD=[];
          for gi=1:n_groups
            for ni=1:n_conditions
                eval(['H_normalityP=[H_normalityP lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_P)];'])%% if zero - normality can't be rejected 
                eval(['H_normalityE=[H_normalityE lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_E)];'])%% if zero - normality can't be rejected 
                eval(['H_normalityM=[H_normalityM lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_M)];'])%% if zero - normality can't be rejected 
                eval(['H_normalityD=[H_normalityD lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_D)];'])%% if zero - normality can't be rejected 
            end
          end
       % test shows normality - so ttest2 can be used 
        [pP,hP] = ttest2(G0_C1_times_P,G1_C1_times_P);% two-sample Kolmogorov-Smirnov test
        [pE,hE] = ttest2(G0_C1_times_E,G1_C1_times_E);
        [pM,hM] = ttest2(G0_C1_times_M,G1_C1_times_M);
        [pD,hD] = ttest2(G0_C1_times_D,G1_C1_times_D);

        [pPg,hPg] = ttest2(G1_C0_times_P,G1_C1_times_P);% two-sample Kolmogorov-Smirnov test
        [pDg,hDg] = ttest2(G1_C0_times_D,G1_C1_times_D);
        
        disp(['P changed from ' num2str(mean(G1_C0_times_P)) '+-' num2str(std(G1_C0_times_P)/sqrt(length(G1_C0_times_P)))...
            ' to ' num2str(mean(G1_C1_times_P)) '+-' num2str(std(G1_C1_times_P)/sqrt(length(G1_C1_times_P)))]);
        
        disp(['E changed from ' num2str(mean(G1_C0_times_E)) '+-' num2str(std(G1_C0_times_E)/sqrt(length(G1_C0_times_E)))...
            ' to ' num2str(mean(G1_C1_times_E)) '+-' num2str(std(G1_C1_times_E)/sqrt(length(G1_C1_times_E)))]);
        
        
        cd('Z:\Anat\behavioral\data')
        
        print('VIP ablation','-depsc')
    case 4 % vip DREADD. or any experiment with a few groups, a few conditions
        EVENTS=EVENTS(find(~cellfun(@isempty,EVENTS)));
        n_conditions=length(unique(EVENTS));% two conditions. such as before injection
        n_groups=n_Groups; % three group, experimental and control and AR
        conditions_names=unique(EVENTS,'stable');
        switch exp
            case 'VIP_DREADD'
                %                 groups_names={'ctrl','DREADD', 'AR'};% should have a string that can be identify in the xls sheet
                %                 labels_str={'Ctrl C0','Ctrl C1','Ctrl C2','Ctrl C3','Exp C0','Exp C1','Exp C2','Exp C3','AR C0','AR C1','AR C2','AR C3'};
                %
                groups_names={'ctrl','DREADD'};% should have a string that can be identify in the xls sheet
                labels_str={'Ctrl C0','Ctrl C1','Ctrl C2','Ctrl C3','Exp C0','Exp C1','Exp C2','Exp C3'};
            case 'DD to DL2'
                groups_names={'VIP_cre'};% should have a string that can be identify in the xls sheet
                labels_str={'C0','C1','C2','C3','C4'};
        end
        
        for ni=1:n_conditions
            eval(['C' num2str(ni-1) '=ablT(strcmp(ablT.Exp' num2str(exp_num) ',conditions_names{ni}),:);']); %
        end
        
        % create a new table ;
        new_T=[];
         for ni=1:n_conditions
            eval(['new_T=[new_T;C' num2str(ni-1) '(:,[1,4:end])];']);
         end
        % change names of states
        for i=2:numel(new_T.Properties.VariableNames) % change Pro to P ect.
            name=new_T.Properties.VariableNames{i};
            eval(['new_T.' name '(strcmp(new_T.' name ',''Pro'')) = {''P''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Est'')) = {''E''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''E-M'')) = {''E''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Met'')) = {''M''};']);
            eval(['new_T.' name '(strcmp(new_T.' name ',''Di'')) = {''D''};']);
        end
        % find the indexes for experimental and control
        for gi=1:n_groups
            eval(['G' num2str(gi-1) '_inds=find(~cellfun(@isempty,strfind(new_T.Properties.VariableNames,groups_names{gi})));']);
        end
        
        % creates new tables, G0_C0 etc...
        for ni=1:n_conditions
            eval(['C' num2str(ni-1) '=new_T(strcmp(new_T.Exp' num2str(exp_num) ',conditions_names{ni}),:);'])
            for gi=1:n_groups
                eval(['G' num2str(gi-1) '_C' num2str(ni-1) '=C' num2str(ni-1) '(:,G' num2str(gi-1) '_inds);']);
            end
        end
        
        % ctrl  and experimental % get number of occurance of each state in the control
        % group, for each condition. C - condition. G - group
        
        for gi=1:n_groups
            for ni=1:n_conditions
                eval(['nei=numel(G'  num2str(gi-1) '_C' num2str(ni-1) '.Properties.VariableNames);']); %
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_P=[];']);
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_E=[];']);
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_M=[];']);
                eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_D=[];']);
                for i=1:nei
                    eval(['name=G'  num2str(gi-1) '_C'  num2str(ni-1) '.Properties.VariableNames{i};']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_P=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_P sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''P''))];']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_E=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_E sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''E''))];']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_M=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_M sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''M''))];']);
                    eval(['G'  num2str(gi-1) '_C' num2str(ni-1) '_times_D=[G'  num2str(gi-1) '_C' num2str(ni-1) '_times_D sum(strcmp(G'  num2str(gi-1) '_C'  num2str(ni-1) '.' name ',''D''))];']);
                end
            end
        end
        
        switch n_groups
            case 3
                for ni=1:n_conditions
                    eval(['x' num2str(ni-1) '=[G0_C' num2str(ni-1) '_times_P G1_C' num2str(ni-1) '_times_P G2_C' num2str(ni-1) '_times_P G0_C' num2str(ni-1) '_times_E G1_C' num2str(ni-1) '_times_E G2_C' num2str(ni-1) '_times_E G0_C' num2str(ni-1) '_times_M G1_C' num2str(ni-1) '_times_M G2_C' num2str(ni-1) '_times_M G0_C' num2str(ni-1) '_times_D G1_C' num2str(ni-1) '_times_D G2_C' num2str(ni-1) '_times_D];']);
                    eval(['g' num2str(ni-1) '=[ones(1,length(G0_C' num2str(ni-1) '_times_P)) 2*ones(1,length(G1_C' num2str(ni-1) '_times_P)) 3*ones(1,length(G2_C' num2str(ni-1) '_times_P)) 4*ones(1,length(G0_C' num2str(ni-1) '_times_E)) 5*ones(1,length(G1_C' num2str(ni-1) '_times_E)) 6*ones(1,length(G2_C' num2str(ni-1) '_times_E)) 7*ones(1,length(G0_C' num2str(ni-1) '_times_M)) 8*ones(1,length(G1_C' num2str(ni-1) '_times_M)) 9*ones(1,length(G2_C' num2str(ni-1) '_times_M)) 10*ones(1,length(G0_C' num2str(ni-1) '_times_D)) 11*ones(1,length(G1_C' num2str(ni-1) '_times_D)) 12*ones(1,length(G2_C' num2str(ni-1) '_times_D))];']);
                end
            case 2
                for ni=1:n_conditions
                    eval(['x' num2str(ni-1) '=[G0_C' num2str(ni-1) '_times_P G1_C' num2str(ni-1) '_times_P G0_C' num2str(ni-1) '_times_E G1_C' num2str(ni-1) '_times_E G0_C' num2str(ni-1) '_times_M G1_C' num2str(ni-1) '_times_M G0_C' num2str(ni-1) '_times_D G1_C' num2str(ni-1) '_times_D];']);
                    eval(['g' num2str(ni-1) '=[ones(1,length(G0_C' num2str(ni-1) '_times_P)) 2*ones(1,length(G1_C' num2str(ni-1) '_times_P)) 3*ones(1,length(G0_C' num2str(ni-1) '_times_E)) 4*ones(1,length(G1_C' num2str(ni-1) '_times_E)) 5*ones(1,length(G0_C' num2str(ni-1) '_times_M)) 6*ones(1,length(G1_C' num2str(ni-1) '_times_M)) 7*ones(1,length(G0_C' num2str(ni-1) '_times_D)) 8*ones(1,length(G1_C' num2str(ni-1) '_times_D))];']);
                end
            case 1
                for ni=1:n_conditions
                    eval(['x' num2str(ni-1) '=[G0_C' num2str(ni-1) '_times_P  G0_C' num2str(ni-1) '_times_E G0_C' num2str(ni-1) '_times_M G0_C' num2str(ni-1) '_times_D];']);
                    eval(['g' num2str(ni-1) '=[ones(1,length(G0_C' num2str(ni-1) '_times_P)) 2*ones(1,length(G0_C' num2str(ni-1) '_times_E))  3*ones(1,length(G0_C' num2str(ni-1) '_times_M))  4*ones(1,length(G0_C' num2str(ni-1) '_times_D)) ];']);
                end
        end
        
                
        cd('Z:\Anat\behavioral\data')
        
        
        figure
        p=0;
        for gi=1:n_groups
            for ni=1:n_conditions
                p=p+1;
                subplot(3,n_conditions*n_groups,p)
                eval(['data_to_plot=[sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_P) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_E) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_M) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_D)];']);
                h=pie(data_to_plot);
                for k=1:2:length(h)
                    set(h(k), 'FaceColor', colors2(k,:))
                end
                title(labels_str{p})
            end
        end
        switch exp
            case 'VIP_DREADD'
                print('VIP DREADD_pie','-depsc')
        end
                
                
        figure
        %subplot(3,4,[5:8])
        state_dist=[];
        for gi=n_groups:-1:1 % have to do reversed to get the order which fits the upper panel
            for ni=n_conditions:-1:1
                %state_dist=[sum(G0_C0_times_P) sum(G0_C0_times_E) sum(G0_C0_times_M) sum(G0_C0_times_D);sum(G1_C0_times_P) sum(G1_C0_times_E) sum(G1_C0_times_M) sum(G1_C0_times_D)];
                eval(['state_dist=[sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_P) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_E) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_M) sum(G' num2str(gi-1) '_C' num2str(ni-1) '_times_D);state_dist ];']);
            end
        end
        state_dist_per=(state_dist'./sum(state_dist')*100)';
        
        bh2=bar(state_dist_per,'stacked','DisplayName','state_dist');
        set(gca,'XTick',1:size(state_dist,1));
        set(gca,'XTickLabel',labels_str);
        for k=1:length(bh2)
            set(bh2(k), 'FaceColor', colors(k,:))
        end
        ylabel('States distribution')
        legend({'P' 'E' 'M' 'D'})
        ylim([0 105])
        switch exp
            case 'VIP_DREADD'
                print('VIP DREADD_dist','-depsc')
        end
        
        if n_groups==3
            switch n_conditions
                case 2
                    Label_str_C0={'P C0 G0','P C0 G1','P C0 G2','E C0 G0','E C0 G1','E C0 G2','M C0 G0','M C0 G1','M C0 G2','D C0 G0','D C0 G1','D C0 G2'};
                    Label_str_C1={'P C1 G0','P C1 G1','P C1 G2','E C1 G0','E C1 G1','E C1 G2','M C1 G0','M C1 G1','M C1 G2','D C1 G0','D C1 G1','D C1 G2'};
                case 3
                    Label_str_C0={'P C0 G0','P C0 G1','P C0 G2','E C0 G0','E C0 G1','E C0 G2','M C0 G0','M C0 G1','M C0 G2','D C0 G0','D C0 G1','D C0 G2'};
                    Label_str_C1={'P C1 G0','P C1 G1','P C1 G2','E C1 G0','E C1 G1','E C1 G2','M C1 G0','M C1 G1','M C1 G2','D C1 G0','D C1 G1','D C1 G2'};
                    Label_str_C2={'P C2 G0','P C2 G1','P C2 G2','E C2 G0','E C2 G1','E C2 G2','M C2 G0','M C2 G1','M C2 G2','D C2 G0','D C2 G1','D C2 G2'};
                case 4
                    Label_str_C0={'P C0 G0','P C0 G1','P C0 G2','E C0 G0','E C0 G1','E C0 G2','M C0 G0','M C0 G1','M C0 G2','D C0 G0','D C0 G1','D C0 G2'};
                    Label_str_C1={'P C1 G0','P C1 G1','P C1 G2','E C1 G0','E C1 G1','E C1 G2','M C1 G0','M C1 G1','M C1 G2','D C1 G0','D C1 G1','D C1 G2'};
                    Label_str_C2={'P C2 G0','P C2 G1','P C2 G2','E C2 G0','E C2 G1','E C2 G2','M C2 G0','M C2 G1','M C2 G2','D C2 G0','D C2 G1','D C2 G2'};
                    Label_str_C3={'P C3 G0','P C3 G1','P C3 G2','E C3 G0','E C3 G1','E C3 G2','M C3 G0','M C3 G1','M C3 G2','D C3 G0','D C3 G1','D C3 G2'};
            end
        elseif n_groups==2
            switch n_conditions
                case 2
                    Label_str_C0={'P C0 G0','P C0 G1','E C0 G0','E C0 G1','M C0 G0','M C0 G1','D C0 G0','D C0 G1'};
                    Label_str_C1={'P C1 G0','P C1 G1','E C1 G0','E C1 G1','M C1 G0','M C1 G1','D C1 G0','D C1 G1'};
                case 3
                    Label_str_C0={'P C0 G0','P C0 G1','E C0 G0','E C0 G1','M C0 G0','M C0 G1','D C0 G0','D C0 G1'};
                    Label_str_C1={'P C1 G0','P C1 G1','E C1 G0','E C1 G1','M C1 G0','M C1 G1','D C1 G0','D C1 G1'};
                    Label_str_C2={'P C2 G0','P C2 G1','E C2 G0','E C2 G1','M C2 G0','M C2 G1','D C2 G0','D C2 G1'};
                case 4
                    Label_str_C0={'P C0 G0','P C0 G1','E C0 G0','E C0 G1','M C0 G0','M C0 G1','D C0 G0','D C0 G1'};
                    Label_str_C1={'P C1 G0','P C1 G1','E C1 G0','E C1 G1','M C1 G0','M C1 G1','D C1 G0','D C1 G1'};
                    Label_str_C2={'P C2 G0','P C2 G1','E C2 G0','E C2 G1','M C2 G0','M C2 G1','D C2 G0','D C2 G1'};
                    Label_str_C3={'P C3 G0','P C3 G1','E C3 G0','E C3 G1','M C3 G0','M C3 G1','D C3 G0','D C3 G1'};
            end
        elseif n_groups==1
             switch n_conditions
                   case 5
                    Label_str_C0={'P C0 G0','E C0 G0','M C0 G0','D C0 G0'};
                    Label_str_C1={'P C1 G0','E C1 G0','M C1 G0','D C1 G0'};
                    Label_str_C2={'P C2 G0','E C2 G0','M C2 G0','D C2 G0'};
                    Label_str_C3={'P C3 G0','E C3 G0','M C3 G0','D C3 G0'};
                    Label_str_C4={'P C4 G0','E C4 G0','M C4 G0','D C3 G0'};
             end

        end
        switch n_groups
            case 3
                colors2=[    0.8500    0.3250    0.0980;...
                    0.8500    0.3250    0.0980;...
                    0.8500    0.3250    0.0980;...
                    0.8000    0.8000    0.8000;...
                    0.8000    0.8000    0.8000;...
                    0.8000    0.8000    0.8000;...
                    0.6000    0.6000    0.6000;...
                    0.6000    0.6000    0.6000;...
                    0.6000    0.6000    0.6000;...
                    0.4000    0.4000    0.4000;...
                    0.4000    0.4000    0.4000;...
                    0.4000    0.4000    0.4000];
        end
        
        figure
        %subplot(3,4,[9:10])
        subplot(n_conditions,1,1)
        boxplot(x0,g0, ...
            'Labels', {Label_str_C0}, ...
            'Colors',colors2,'PlotStyle','compact'); hold on
        ylim([0 15])
        
        %subplot(3,4,[11 12])
        subplot(n_conditions,1,2)
        boxplot(x1,g1, ...
            'Labels', {Label_str_C1}, ...
            'Colors',colors2,'PlotStyle','compact'); hold on
        ylim([0 15])
        
        if n_conditions>2
            subplot(n_conditions,1,3)
            boxplot(x2,g2, ...
                'Labels', {Label_str_C2}, ...
                'Colors',colors2,'PlotStyle','compact'); hold on
            ylim([0 15])
        end
        
         if n_conditions>3
            subplot(n_conditions,1,4)
            boxplot(x3,g3, ...
                'Labels', {Label_str_C3}, ...
                'Colors',colors2,'PlotStyle','compact'); hold on
            ylim([0 15])
        end
        print('VIP DREADD_3','-depsc')
        
        % check normality of the data to decide on statistical test
        % The result 1 means that it reject the hypothesis that X has a
        % normal distribution (ttest2 cannot be used)
        H_normalityP=[];H_normalityE=[];H_normalityM=[];H_normalityD=[];
        for gi=1:n_groups
            for ni=1:n_conditions
                eval(['H_normalityP=[H_normalityP lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_P)];'])%% if zero - normality can't be rejected
                eval(['H_normalityE=[H_normalityE lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_E)];'])%% if zero - normality can't be rejected
                eval(['H_normalityM=[H_normalityM lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_M)];'])%% if zero - normality can't be rejected
                eval(['H_normalityD=[H_normalityD lillietest(G' num2str(gi-1) '_C' num2str(ni-1) '_times_D)];'])%% if zero - normality can't be rejected
            end
        end
        
        % if lillietest shows normality - so ttest2 can be used
       if n_groups==1
            

                [pP03,hP03] = ttest2(G0_C0_times_P,G0_C3_times_P);%
                [pP34,hP34] = ttest2(G0_C3_times_P,G0_C4_times_P);%
                [pP23,hP23] = ttest2(G0_C2_times_P,G0_C3_times_P);%

                  L=length(G0_C0_times_P);
                  [p,tbl,stats] = kruskalwallis([G0_C0_times_P G0_C1_times_P G0_C2_times_P G0_C3_times_P G0_C4_times_P],[ones(1,L) 2*ones(1,L) 3*ones(1,L) 4*ones(1,L) 5*ones(1,L)]);
                  c_G1 = multcompare(stats,'CType','hsd')
                  cn = multcompare(stats)
                
                [pE,hE] = ttest2(G0_C0_times_E,G0_C4_times_E);
                [pM,hM] = ttest2(G0_C3_times_M,G0_C4_times_M);
                [pD,hD] = ttest2(G0_C3_times_D,G0_C4_times_D);
       else
                [pP,hP] = ttest2(G0_C3_times_P,G1_C3_times_P);%
                [pE,hE] = ttest2(G0_C3_times_E,G1_C3_times_E);
                [pM,hM] = ttest2(G0_C3_times_M,G1_C3_times_M);
                [pD,hD] = ttest2(G0_C3_times_D,G1_C3_times_D);
                
                [pP13,hP13] = ttest2(G1_C1_times_P,G1_C3_times_P);%
                [pP12,hP12] = ttest2(G1_C1_times_P,G1_C2_times_P);%
                [pP03,hP03] = ttest2(G1_C0_times_P,G1_C3_times_P);%
                [pP23,hP23] = ttest2(G1_C2_times_P,G1_C3_times_P);%
                
                
                
                
                %  non normality based statistics
                
                [p,tbl,stats] = kruskalwallis([G0_C3_times_P G1_C3_times_P],[ones(1,length(G0_C3_times_P)) 2*ones(1,length(G1_C3_times_P))]);
                [p,tbl,stats_G1] = kruskalwallis([G1_C0_times_P G1_C1_times_P G1_C2_times_P G1_C3_times_P],[ones(1,length(G1_C0_times_P)) 2*ones(1,length(G1_C0_times_P)) 3*ones(1,length(G1_C0_times_P)) 4*ones(1,length(G1_C0_times_P))]);
                
                
                %c = multcompare(stats,'CType','bonferroni');
                %c = multcompare(stats,'CType','dunn-sidak');
                c = multcompare(stats,'CType','hsd')
                c_G1 = multcompare(stats_G1,'CType','hsd')
                cn = multcompare(stats)
                
                
                disp(['P changed from C0 ' num2str(mean(G1_C0_times_P)) '+-' num2str(std(G1_C0_times_P)/sqrt(length(G1_C0_times_P)))...
                    ' to C1 ' num2str(mean(G1_C1_times_P)) '+-' num2str(std(G1_C1_times_P)/sqrt(length(G1_C1_times_P)))]);
                if n_conditions>2;        disp([' to C2 ' num2str(mean(G1_C2_times_P)) '+-' num2str(std(G1_C2_times_P)/sqrt(length(G1_C2_times_P)))]); end
                if n_conditions>3;        disp([' to C3 ' num2str(mean(G1_C3_times_P)) '+-' num2str(std(G1_C3_times_P)/sqrt(length(G1_C3_times_P)))]); end
                
                disp(['E changed from ' num2str(mean(G1_C0_times_E)) '+-' num2str(std(G1_C0_times_E)/sqrt(length(G1_C0_times_E)))...
                    ' to ' num2str(mean(G1_C1_times_E)) '+-' num2str(std(G1_C1_times_E)/sqrt(length(G1_C1_times_E)))]);
                if n_conditions>2;        disp([' to C2 ' num2str(mean(G1_C2_times_E)) '+-' num2str(std(G1_C2_times_E)/sqrt(length(G1_C2_times_E)))]); end
                if n_conditions>3;        disp([' to C3 ' num2str(mean(G1_C3_times_E)) '+-' num2str(std(G1_C3_times_E)/sqrt(length(G1_C3_times_E)))]); end
        end

end
% 
