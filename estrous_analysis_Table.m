function estrous_analysis_Table(y)
% first run 'y=read_estrus_log_table' and choose the right exp number
% numbers to state based on 'est_state2num'
%exp= 'VIP_ablation';
% 9 animals, 41 days

clear mi 
select_sessions_to_plot=0;


%new_T(:,1)=y.T(2:end,1);
M=y.T(2:end,4:end);
%GnRHcas9:
new_T(:,1)=y.T(1:end,1);
M=y.T(1:end,4:end);

exp=['Exp' num2str(y.exp_num)];
% switch y.full_path
%     case 'C:\Users\anatk\Documents\Data_Glab_home_work\Behavioral\data\VIPChR2.xlsx'
%         MyVariableNames={'VIP-ChR2_exp' 'VIP-ChR2_exp'  'VIP-ChR2_exp' 'VIP-ChR2_ctrl' 'VIP-ChR2_ctrl' };
%     otherwise
%         MyVariableNames=y.T.Properties.VariableNames(4:end);
% end

% get mouse IDs
MyVariableNames=y.T.Properties.VariableNames(4:end);
%P_table=array2table(P ,'VariableNames',MyVariableNames);

new_T2=[new_T M];
phases=unique(new_T2(:,1),'stable');% finds how many different conditions were done
% remove the first 3 weeks of manipulation
switch exp
    case 'Exp4'% Jetlag
        for si=1:length(new_T2.Exp4)
            if contains(new_T2.Exp4{si},'JL')
                new_T2.Exp4{si}='JL';
            end
        end
        phases=unique(new_T2(:,1));% finds how many different conditions were done
    case 'Exp24'% C57 light manipulation
        new_T2=new_T2([1:21 43:end],:);
        phases=phases([1 3 4],:);
%     case 'Exp120'% vip ablation 
%         new_T2=new_T2([1:21 43:end],:);
%         phases=phases([1 3 4],:);
    case 'Exp122'
        
        new_T2=new_T2(2:end,1:length(y.STRAIN_ID)+1);
        tmp_T=new_T2(strcmp(new_T2{:,1},'DL12_12'),:);
        inc_ind=(sum(strcmp(tmp_T{:,:},'P'))>2);
        new_STRAIN_ID=y.STRAIN_ID(inc_ind(2:end));
        new_T2=[new_T2(:,1) new_T2(:,inc_ind)]; % except only baseline with more then 2 proestrus events
        % excluded one female due to rare response 
        
        new_T2=new_T2(:,find([1 ~strcmp(new_STRAIN_ID,'VIP-cre 441L')]));
        new_STRAIN_ID=new_STRAIN_ID(~strcmp(new_STRAIN_ID,'VIP-cre 441L'));
end
% create the new table

switch exp
    case 'Exp122'
        T=array2table(new_T2.Properties.VariableNames(2:end)');
        T.Var1=new_STRAIN_ID';
    otherwise
        T=array2table(new_T2.Properties.VariableNames');
end

switch exp
    case {'Exp25'; 'Exp23';'Exp126';'Exp4';'Exp120'}
        n_conditions=2; % before and after - Check!!! - good for 126
        G=eval(['findgroups(new_T2.' exp ')']);% get identification of the different conditions as different groups
    case 'Exp122'
        n_conditions=5;
      %  G=eval(['findgroups(new_T2.Var1)']);
          G=eval(['findgroups(new_T2.' exp ')']);% get identification of the different conditions as different groups

    otherwise % should be checked here- which exp needs the upper version and which the lower
        n_conditions=2; % before and after - CHECK!!!!
        G=eval(['findgroups(new_T2.Var1)']);% get identification of the different conditions as different groups
end
%switch exp
% case 'Exp126'
%    G_order=unique(G);;
%otherwise
G_order=unique(G,'stable');;
%end

V_names=[];
clear T_array
for n=1:length(G_order)% go over the different conditions
    T_temp=new_T2(G==G_order(n),:);
    L=4*n-3; %defines how many columbs each condition gets. the last is to add 'time between proestrous'/ 4 parameters are being tested
    for idi=1:length(T_temp.Properties.VariableNames)-1 % go over IDs
        k=idi+1;
        
        var_temp=T_temp.Properties.VariableNames{k};
        % S=sortrows(T_temp,var_temp);
        states1=eval(['T_temp.' var_temp]);
        if ~isempty(states1)
            G_temp=est_state2num_table(eval(['T_temp.' var_temp]));% sorted by ABC order, so D=1,E=2,M=3,P=4
            G_temp=G_temp';
            T_array(idi,L)=length([find(G_temp==1) ;find(G_temp==3)]);%D&M
            T_array(idi,L+1)=length(find(G_temp==4));%P
            T_array(idi,L+2)=length(find(G_temp==2));%E
            if length(diff(find(G_temp==4)))>1;
                T_array(idi,L+3)=median(diff(find(G_temp==4)));% median days between P
            elseif isempty(find(G_temp==4))
                T_array(idi,L+3)=nan;
            elseif length(diff(find(G_temp==4)))==1
                T_array(idi,L+3)=diff(find(G_temp==4));% median days between P
                if find(G_temp==4)==0
                    T_array(idi,L+3)=max(21-find(G_temp==4), find(G_temp==4));
                end
            end
        end
    end
    % gives titles per condtion n
    V_names=[V_names {['nDM_C' num2str(n)] ['nP_C' num2str(n)] ['nE_C' num2str(n)] ['cycle_length_C' num2str(n)]}];
    
end

T=table(T_array,'RowNames',new_T2.Properties.VariableNames(2:end)');
%T=table(T_array,'RowNames',y.STRAIN_ID);
T1=splitvars(T);
T1.Properties.VariableNames=V_names;


% define relation to individual group
 labels_str={};
switch exp
    case 'Exp23'% VIP ChR2
        % before 07282022 
        T1=T1([1:5,7:end],:); % remove 30R ,  locomotor activity shifted a lot and low cfos
        Group={'exp' 'exp' 'exp' 'ctrl' 'ctrl' 'ctrl' 'exp' 'ctrl' 'exp' 'ctrl'  'exp' 'exp' 'GFP' 'GFP' 'GFP'};
        
        % 30cand 35 removed due to LMA and c-fos
        %Group={'exp' 'exp' 'exp' 'ctrl' 'ctrl' 'ctrl' 'exp' 'ctrl' 'ctrl'  'exp' 'exp' 'GFP' 'GFP' 'GFP'};
        %T1=addvars(T1,Group','Before','nDM_C1');
    case 'Exp24'% C57 light manipulation
        Group={'ctrl' 'ctrl' 'ctrl' 'ctrl' 'ctrl' 'exp_L3' 'exp_L3' 'exp_L3' 'exp_L3' 'exp_L3' 'exp_L1' 'exp_L1' 'exp_L1' 'exp_L1' 'exp_L1'};
        %         T1=T1(1:15,:);% remove non relevant data
        %         T1=addvars(T1,Group','Before','nDM_C1');
        %         T1=T1([1:13 15],:);
        
        T1=T1(1:15,:);% remove non relevant females- which had low accurance of estrous cycle at the first 3 weeks
        % add the Group titles to T1
        T1=addvars(T1,Group','Before','nDM_C1');
        %T1=T1([1:13 15],:);
        
    case 'Exp25'% VIP DREADD
        % 7 exp,  6 ctrl. 4 were not responsive to light effect
        Group={'ctrl' 'ctrl' 'exp' 'exp' 'exp' 'ctrl' 'ctrl' 'exp' 'exp' 'ctrl' 'exp' 'exp' 'ctrl' };
        T1=addvars(T1,Group','Before','nDM_C1');
    case 'Exp125'% VIP DREADD and ChR2 just baseline and rescue
        Group={'DD+L1+L2' 'DD+L1+L2' 'DD+L1+L2' 'DD+L1+L2' 'DD+L1+L2' 'DD+L1+L2' 'DD+L1+L2' 'DD+L1' 'DD+L1' 'DD+L1' 'DD+L1' 'DD+L1' 'DD+L1'};
        
        T1=addvars(T1,Group','Before','nDM_C1');
        T1=T1([1:2, 4:end],:);% #3 is 450, which was extreamlt high weight (46gr).
    case 'Exp126'% GnRH cas9 vipr2 or Kiss1R KD with CRISPRVIPGC
        Group={'ctrl' 'ctrl' 'ctrl' 'ctrl' 'ctrl' 'VIPR2 KD' 'VIPR2 KD' 'VIPR2 KD'  'VIPR2 KD' 'KISS1R KD' 'KISS1R KD' 'KISS1R KD' 'KISS1R KD' 'KISS1R KD' 'KISS1R KD' };
        T1=addvars(T1,Group','Before','nDM_C1');
    case 'Exp122' % VIP-cre LD manipulation
        Group=repmat({'VIP-cre'},1,length(new_STRAIN_ID));
        T1=addvars(T1,Group','Before','nDM_C1');
        labels_str={'LD','DD','DD+L0','DD+L0+L10','DD+L0+L4'};
    case 'Exp120' % VIP-cre ablation
        Group={'exp' 'exp' 'exp' 'exp' 'exp'  'exp'  'ctrl'  'ctrl' 'ctrl' 'ctrl' 'ctrl' 'ctrl' };
        T1=addvars(T1,Group','Before','nDM_C1');
        labels_str={'Ctrl','Exp'};
end


disp(T1)

% gather the data to one matrix
all_np=[];
all_ne=[];
all_nMD=[];
all_cycle_length=[];
for n=1:length(G_order)% go over the different conditions
    nP_C=eval(['T1.nP_C' num2str(n)]); % look at nP: number of Proestrus events
    all_np=[all_np nP_C];
    nE_C=eval(['T1.nE_C' num2str(n)]); % look at nE: number of estrus events
    all_ne=[all_ne nE_C];
    nMD_C=eval(['T1.nDM_C' num2str(n)]); % look at nM nD: number of M/D events
    all_nMD=[all_nMD nMD_C];
    cycle_length_C=eval(['T1.cycle_length_C' num2str(n)]); % look at nM nD: number of M/D events
    all_cycle_length=[all_cycle_length cycle_length_C];
end


% now plot
% distributions % figure 2 in SCN_VIP rep paper
switch exp
    case 'Exp126'% GnRH cas9 vipr2 or Kiss1R KD with CRISPRVIPGC
        colors=[0.8500 0.3250 0.0980; 0.8 0.8 0.8; 0.6 0.6 0.6; 0.4 0.4 0.4];
        state_dist=[];
        figure
        for i=1:size(all_nMD,2)
            state_dist=[state_dist ; sum(all_np(:,i)) sum(all_ne(:,i)) sum(all_nMD(:,i))];
        end
        state_dist_per=(state_dist'./sum(state_dist')*100)';
        bh2=bar(state_dist_per,'stacked','DisplayName','state_dist');
        set(gca,'XTick',1:size(state_dist,1));
        set(gca,'XTickLabel',labels_str);
        for k=1:length(bh2)
            set(bh2(k), 'FaceColor', colors(k,:))
        end
        ylabel('States distribution')
        legend({'P' 'E' 'M/D' })
        ylim([0 105])
end
clear X

% first the proestrus
figure
for n=1:length(G_order)% go over the different conditions
    nP_C=eval(['T1.nP_C' num2str(n)]); % look at nP: number of Proestrus events
    switch exp
        case 'Exp4'
            G_exp=ones(size(T1,1),1);
        otherwise
            G_exp=findgroups(T1.Var1);
    end
    
    this_mean=splitapply(@nanmean,nP_C,G_exp);
    this_med=splitapply(@nanmedian,nP_C,G_exp);
    
    switch exp
        case 'Exp4'; X=[n n+3];%Jet lag
        case 'Exp125';  X=[n n+4];% VIP ChR2 and DREADD - just compare baseline with rescure
        case 'Exp23';  X=[n n+4 n+8];% VIP ChR2  
        case 'Exp24' ; X=[n n+4 n+8] ;% C57 Light manipulation     
        case 'Exp25';  X=[n n+5]; % DREADD % 4 conditions: DL, DD+L1, DD+L1+CNOZT16, DD+L1+CNOZT22
        case 'Exp126' ;  X=[n n+3 n+6];% GnRH Cas9. % 3 conditions: before and after injection
        case 'Exp122';  X=n; % VIP-cre light manipulation
        case 'Exp120';  X=[n n+3];
    end
    
    bh=bar(X,this_mean,0.6); hold on
    bh.FaceColor=[0.75 0.75 0.75];
    ylim([0 6])
end


clear X
% now plot individual
switch exp
    case 'Exp23'% VIP ChR2
        X(1,:)=[1:3];  X(2,:)=[5:7]; X(3,:)=[9:11];
        gca.XTick=[X(1,:) X(2,:)];
    case 'Exp24' % C57 Light manipulation
        X(1,:)=[1:3];  X(2,:)=[5:7];   X(3,:)=[9:11]; X(4,:)=[12:14];
        gca.XTick=[X(1,:) X(2,:) X(3,:) X(4,:)];
        gca.XTickLabel=['C1'];
    case 'Exp25'% VIP DREADD
        X(1,:)=[1:4];  X(2,:)=[6:9]; %X(3,:)=[11:14];
        gca.XTick=[X(1,:) X(2,:) ];
    case 'Exp125'% VIP DREADD ChR2-  just compare baseline with rescure
        X(1,:)=[1:3];  X(2,:)=[5:7];
        gca.XTick=[X(1,:) X(2,:)];
    case 'Exp126'% GnRH cas9
        X(1,:)=[1:2];  X(2,:)=[4:5];  X(3,:)=[7:8];
        gca.XTick=[X(1,:) X(2,:) ];
    case 'Exp122' ;  X(1,:)=[1:5];% VIP-cre light manipulation
     case 'Exp120' ;  X(1,:)=[1:2];  X(2,:)=[4:5]; % VIP-cre light manipulation   
      %  gca.XTick=[X(1,:)];
end
k1=0;k2=0;k3=0;
for gi=1:length(G_exp)
    switch exp
        case 'Exp25'% VIP DREADD
            these_val=[T1.nP_C1 T1.nP_C2 T1.nP_C3 T1.nP_C4];
        case 'Exp125'% VIP DREADD and ChR@- comparing baselines
            these_val=[T1.nP_C1 T1.nP_C2 T1.nP_C3];
        case {'Exp126' 'Exp120'}% GnRH cas9 and VIP ablation - 2 conditions
            these_val=[T1.nP_C1 T1.nP_C2 ];
        case 'Exp122' % VIP-cre light manipulation
            these_val=[T1.nP_C1 T1.nP_C2 T1.nP_C3 T1.nP_C4 T1.nP_C5];
        otherwise
            these_val=[T1.nP_C1 T1.nP_C2 T1.nP_C3];
    end
    shift=(1+(gi-3)*0.001);
    if G_exp(gi)==1
        plot(X(1,:)*shift,these_val(gi,:),'o-'); hold on
        k1=k1+1;
    end
    if G_exp(gi)==2
        plot(X(2,:)*shift,these_val(gi,:),'o-'); hold on
        k2=k2+1;
    end
    if G_exp(gi)==3
        plot(X(3,:)*shift,these_val(gi,:),'o-'); hold on
        k3=k3+1;
    end
end
ylim([0 7])
ylabel('# proestrus in 3 weeks')



switch exp
    case 'Exp24' % C57 Light manipulation
        % plotbox figure
        clear all_np_plot
        all_np_plot(1,:,:)=all_np(1:5,:)';
        all_np_plot(3,:,:)=all_np(6:10,:)';
        all_np_plot(2,:,:)=all_np(11:15,:)';
        figure;
        x=1:3;
        h=boxplot2(all_np_plot,x);
        set([h.lwhis h.uwhis], 'linestyle', '-');
        set(h.out, 'marker', '.');
        
end
ordG=unique(G_exp,'stable');
%switch exp;  case 'Exp126'  ordG=unique(G_exp);end% GnRHcas9;
lgd_title=[];
for i=1:length(ordG)
    lgd_ind=min(find(G_exp==(ordG(i))));
    lgd_title=[lgd_title T1.Var1(lgd_ind)];
end
xticklabels(lgd_title(ordG))
% switch exp
%     case 'Exp122' % VIP-cre Light manipulation
%      set(gca,'XTick',[1:5]);
%      set(gca,'XTickLabel',phases.Var1);
% end
ylim([0 6])
% now plot the distributions
clear these_val

figure
switch exp
    case 'Exp23'; order=[1 4 2 5 3 6];% VIP ChR2
    case 'Exp24'; order=[1:9]; % C57 Light manipulation
     
    case {'Exp120' 'Exp126'}; order=[1:13]; % GnRHcas9/ablation 
    case 'Exp122' ; order=[1:5];% VIP-cre light manipulation
end
l=0;
for gi=1:length(ordG)
    k=ordG(gi);
    % order the different conditions
    
    these_val{1}=[T1.nP_C1 T1.nE_C1 T1.("nDM_C1")] ;
    these_val{2}=[T1.nP_C2 T1.nE_C2 T1.("nDM_C2")] ;
    if n_conditions>2
        these_val{3}=[T1.nP_C3 T1.nE_C3 T1.("nDM_C3")] ;
    end
    if n_conditions>4
        these_val{4}=[T1.nP_C4 T1.nE_C4 T1.("nDM_C4")] ;
        these_val{5}=[T1.nP_C5 T1.nE_C5 T1.("nDM_C5")] ;
    end
    L1=length(ordG)*length(these_val);
    for i=1:n_conditions
        l=l+1;
        %subplot(length(ordG),length(these_val),order(l))
        subplot(length(ordG),length(these_val),i)
        h=pie(sum(these_val{i}(G_exp==k,:),1));hold on
    end
end
% create legend
labels={'P' ,'E' ,'M/D'};
lgd = legend (labels);

% now do some statistics
if n_conditions>2
    [p,tbl,stats] = kruskalwallis([all_np(:,1) ;all_np(:,2);all_np(:,3) ;all_np(:,4)],[G_exp;G_exp+max(G_exp);G_exp+2*max(G_exp);G_exp+3*max(G_exp)]);
    [p,tbl,stats] = kruskalwallis([all_np(:,1) ;all_np(:,2);all_np(:,3) ],[G_exp;G_exp+max(G_exp);G_exp+2*max(G_exp)]);
elseif n_conditions>4   
    [p,tbl,stats] = kruskalwallis([all_np(:,1) ;all_np(:,2);all_np(:,3) ;all_np(:,4);all_np(:,5)],[G_exp;G_exp+max(G_exp);G_exp+2*max(G_exp);G_exp+3*max(G_exp);G_exp+4*max(G_exp)]);
else
    [p,tbl,stats] = kruskalwallis([all_np(:,1) ;all_np(:,2)],[G_exp;G_exp+max(G_exp)]);
    %[p,tbl,stats] = kruskalwallis([all_np(:,1) ;all_np(:,2) ],[G_exp;G_exp+max(G_exp);G_exp+2*max(G_exp)]);
end

%c = multcompare(stats,'CType','bonferroni');
%c = multcompare(stats,'CType','dunn-sidak');
c = multcompare(stats,'CType','hsd')
cn = multcompare(stats)

for ci=1:size(all_np,2)
    disp (['# proestrus, condition ' num2str(ci) ' : ' num2str(mean(all_np(:,ci))) '+-' num2str(std(all_np(:,ci))/sqrt(size(all_np,1)))])
end

for ci=1:size(all_np,2)
    for gi=1:length(unique(G_exp))
        disp (['# proestrus, condition ' num2str(ci) ' group ' num2str(gi) ': ' num2str(mean(all_np(find(G_exp==gi),ci))) '+-' num2str(std(all_np(find(G_exp==gi),ci))/sqrt(length(find(G_exp==gi))))])
    end
end

for ci=1:size(all_np,2)
    for gi=1:length(unique(G_exp))
        disp (['cycle length, condition ' num2str(ci) ' group ' num2str(gi) ': ' num2str(mean(all_cycle_length(find(G_exp==gi),ci))) '+-' num2str(std(all_np(find(G_exp==gi),ci))/sqrt(length(find(G_exp==gi))))])
    end
end

1

