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
%exp='DD to DL2'%without  432RL
%exp='VIPChR2'
%exp='VIPChR2_no30_35'
exp='VIP_DREADD';
%exp='DD to DL3' % VIP and DREADD- compare baseline to rescue
%exp='C57'
%exp='WT jet lag'
switch exp
    case 'VIP ablation'; file_name='Estrous cycle cytology ablation AK2';  exp_num=120;
    case 'Esr1OK'        ;file_name='Estrous cycle cytology Esr1KO AK';    exp_num=121;
    case 'GnRHcas9_KD' ; file_name='GnRHcas9';                              exp_num=126;
    case 'VIPChR2';      file_name='VIPChR2';                               exp_num=23;
    case 'VIPChR2_no30_35'; file_name='VIPChR2_no30_35';                   exp_num=23
    case 'C57';         file_name='C57 DL to DD with L';                    exp_num=24
    case 'VIP_DREADD';  file_name='VIP_DREADD';                            exp_num=25
    case 'DD to DL';    file_name='VIP DL to DD';                          exp_num=122
    case 'DD to DL2'  ;  file_name='VIP DL to DD2';                         exp_num=122
    case 'DD to DL3' ;  file_name='VIPCHR2_DREADD DL to DD';  exp_num=125% estrus_analysis_Table- doesn't run well yet
    case 'WT jet lag';   file_name=exp;                        exp_num=4;
end
path='C:\Users\anatk\Documents\Data_Glab_home_work\Behavioral\data\'
%path='D:\DATA_Glab\Behavioral\data\';
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
DATES=TXT(START+3:STOP-1,2);
% find the events
EVENTS=full_EVENTS(START+3:STOP-1); 
% find the states of each individual, and makes an array of strains and ID 
n=0;
for k=4:size(TXT,2)
    if ~isempty(TXT{START,k}) % only if there is ID
    n=n+1;
    STATES{n}=TXT(START+3:STOP,k);
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
subplot(length(IDS)+1,1,ni+1)

switch exp_type
    case 1 % laser 
        ph=line([hab_start,hab_stop+1],[1,1]);% habituation
        set(ph, 'linewidth',14, 'color',[0.82 0.82 0.82])
        hold on
        ph2=line([laser_start,laser_stop+1],[1,1]);% laser on
        set(ph2, 'linewidth',14, 'color',[0 0 1])
        
    case 2 % L/D manipulation
        for ei=1:length(event_array)
            if event_array(ei)==5
            ph2=line([ei-0.2,ei+0.2],[1,1]);
            set(ph2, 'linewidth',14, 'color',[0.5 0.5 0.5])
            hold on
            end
        end
   % case {[]}
        
end

set(gca,'xlim',[0,XLIM], 'ylim', [.5,1.5],'YTick',[], 'Color', [0.95 0.95 0.95])
xlabel('Days')


switch exp_type
    case 2 % jet-lag
        labels_str={'Ctrl','Exp'};
        colors=[0.8500 0.3250 0.0980; 0.8 0.8 0.8; 0.6 0.6 0.6; 0.4 0.4 0.4];
        colors2=[0.8500 0.3250 0.0980; 0.8500 0.3250 0.0980; 0.8 0.8 0.8;0.8 0.8 0.8; 0.6 0.6 0.6; 0.6 0.6 0.6; 0.4 0.4 0.4; 0.4 0.4 0.4];
        ablT=y.T;
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

end
% 
