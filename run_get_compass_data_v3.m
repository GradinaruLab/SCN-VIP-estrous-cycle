function run_get_compass_data_v3
% run comparison between compass data
% Anat Kahan R2020b

% compare motion data between DL, DD with red light, and complete DD
%all_exp={'VIP_cre_DL','VIP_cre_DD_red_light', 'VIP_cre_DD_completeDD'};

% compare motion data between DL, DD with blue light, and complete DD
%all_exp={'VIP_cre_LD 102021','VIP_cre_DD_30min_blue','VIP_cre_DD_completeDD_Blue','VIP_cre_DD_30min@8pm'}%, 'VIP_cre_DD_completeDD','VIP_cre_DD_blue_light'};

% just DD+L1
%all_exp={'VIP_cre_DD_30min@8pm'}%, 'VIP_cre_DD_completeDD','VIP_cre_DD_blue_light'};

% compare light manipulation data with VIP-cre mice (estrus cycle)
all_exp={'VIP_cre_DL', 'VIP_cre_DD','VIP_cre_DD_30min@8pm','VIP_cre_DD_30min@8pm_60min@530am','VIP_cre_DD_30min@8pm_60min@12am'};

One_day_COMP=24*60*6; % recording every 10 seconds

for ei=1:length(all_exp)
    [output{ei}]=get_compass_data_v3(all_exp{ei});
end

% get all the mean activity values
clear hourly_mean_activity
L=20 ;% number of days
slope_string=[];
for ei=1:length(all_exp)
    % get mean values and set as input for linear regression 
    y_mean=output{ei}.hour_mean_activity;   
    y_mean=y_mean(~isnan(y_mean)); % re;moves nan but also put in 1D array
    
    y_onset=output{ei}.activity_onsets;    
    y_onset=y_onset(~isnan(y_onset)); % re;moves nan but also put in 1D array
 
    y_offset=output{ei}.activity_offsets;    
    y_offset=y_offset(~isnan(y_offset)); % re;moves nan but also put in 1D array

    L=floor(size(~isnan(y_mean),1)/2);
    if length(y_mean)>2*L; y_mean=y_mean(1:2*L);end
    
    L=floor(size(~isnan(y_onset),1)/2);
    if length(y_onset)>2*L; y_onset=y_onset(1:2*L);end
 
    L=floor(size(~isnan(y_offset),1)/2);
    if length(y_offset)>2*L; y_offset=y_offset(1:2*L);end
    % set y
    x=[1:L; 1:L];
    x=x(1:2*L); % transforms to 1D 
    X = [ones(length(x),1) x'];
    % calculate the slope and the intersect for the mean 
    b1{ei} = X\y_mean;    
    yCalc1 = X*b1{ei};
% calculate the slope and the intersect for the onset  
    b2{ei}=X\y_onset;
    yCalc2 = X*b2{ei};
    
    % calculate the slope and the intersect for the offset   
    b3{ei}=X\y_offset;
    yCalc3 = X*b3{ei};
%     tmp=Y/[X; ones(1,N)];%find least squares line fit
%     b_lse=tmp(1);
%     a_lse=tmp(2);

%now plot

    figure; 
    %scatter(y_mean,x); hold on ;scatter(y_onset,x); hold on ;scatter(y_offset,x); hold on; plot(yCalc1,x,'b--'); hold on; plot(yCalc2,x,'r--'); plot(yCalc3,x,'g--'); xlim([0 24]);ylim([0 20]); title(output{ei}.exp_title)
    scatter(y_mean,x); hold on ;scatter(y_onset,x); hold on; plot(yCalc1,x,'b--'); hold on; plot(yCalc2,x,'r--'); xlim([0 24]);ylim([0 20]); title(output{ei}.exp_title)
    xlabel('Time (hours)'); ylabel ('Time (days)');
    set(gca, 'YDir','reverse')
    % calculate R^2
    Rsq1(ei) = sum((y_mean - yCalc1).^2)/sum((y_mean - mean(y_mean)).^2);
    Rsq2(ei) = sum((y_onset - yCalc2).^2)/sum((y_onset - mean(y_onset)).^2);
    Rsq3(ei) = sum((y_offset - yCalc3).^2)/sum((y_offset - mean(y_offset)).^2);
    
    fit{ei}.x=x' ;
    fit{ei}.yCalc1=yCalc1;
    fit{ei}.yCalc2=yCalc2;
    fit{ei}.yCalc3=yCalc3;
    disp([all_exp{ei} ': total phase shift of mean: ' num2str(yCalc1(end)-yCalc1(1))])
    disp([all_exp{ei} ': mean activity: ' num2str(mean(yCalc1)+12) '+-' num2str(std(yCalc1)/sqrt(length(yCalc1)))])
end
%disp (['slopes are ' num2str(b{1}(2)) ' ' num2str(b{2}(2)) ' ' num2str(b{3}(2))  ' ' num2str(b{4}(2)) ' ' num2str(b{5}(2))])
disp (['mean slopes are ' num2str(b1{1}(2)) ' ' num2str(b1{2}(2)) ' ' num2str(b1{3}(2))])
disp (['onset slopes are ' num2str(b2{1}(2)) ' ' num2str(b2{2}(2)) ' ' num2str(b2{3}(2))])
disp (['offset slopes are ' num2str(b3{1}(2)) ' ' num2str(b3{2}(2))])

% check how it looks as a bar plot 
data_for_plots={'mean activity','activity onset'};
figure
for i=1:length(data_for_plots)
    subplot(1,length(data_for_plots),i)
    for ei=1:length(all_exp)
        switch data_for_plots{i}
            case 'mean activity'
                data_to_plot=output{ei}.hour_mean_activity';
            case 'activity onset'
                data_to_plot=output{ei}.activity_onsets';
            case 'activity offset'
                data_to_plot=output{ei}.activity_offsets';
        end
        diff_mean=diff(data_to_plot);
        sem=std(mean(diff_mean))/sqrt(size(diff_mean,2));
        bar(ei, mean(mean(diff_mean'))); hold on
        %plot(ei*ones(size(diff_mean,1),2),diff_mean,'*'); hold on 
        plot([ei*ones(size(diff_mean,1),1) ei*ones(size(diff_mean,1),1)],[mean(mean(diff_mean))+sem mean(mean(diff_mean))-sem ])
    end
    ylabel(['Phase shift in ' data_for_plots{i} ' (min)'])
    xlim([0.5 length(all_exp)+0.5])
    ylim([-0.15 0.75])
    set(gca,'XTick',1:length(all_exp));
    set(gca,'XTickLabel',all_exp);
end

% statistics: 
for i=1:length(data_for_plots)   
    y=[];g=[];
    for ei=1:length(all_exp)
        switch data_for_plots{i}
            case 'mean activity'
                data_to_plot=output{ei}.hour_mean_activity';
            case 'activity onset'
                data_to_plot=output{ei}.activity_onsets';
            case 'activity offset'
                data_to_plot=output{ei}.activity_offsets';
        end
        diff_mean=diff(data_to_plot);
        y=[y mean(diff_mean)];
        g=[g ei*ones(1,length(mean(diff_mean)))];
    end
    figure
    [p,tbl,stats] = kruskalwallis(y,g);
    c{i} = multcompare(stats,'CType','bonferroni');
end


% subplot(1,3,2)
% for ei=1:length(all_exp)
%     diff_onset=diff(output{ei}.activity_onsets');
%     bar(ei, mean(mean(diff_onset'))); hold on
%     plot(ei*ones(size(diff_onset,1),2),diff_onset,'*'); hold on
% end
% ylabel('Phase shift in onset (min)')
% xlim([0.5 length(all_exp)+0.5])
% 
% set(gca,'XTick',1:length(all_exp));
% set(gca,'XTickLabel',all_exp);


% check
%  x=[1:L+1];
%  figure
% plot(y(1,:),x,'*')

LINESTYLES={'-','--','-.',':','-'};
colors=[0 0 0;  0.15 0.15 0.15; 0.3 0.3 0.3;  0.45 0.45 0.45; 0.7 0.7 0.7];
colors2=[0.1 0.2 0]+[0 0 0;  0.15 0.15 0.15; 0.3 0.3 0.3;  0.45 0.45 0.45; 0.6 0.6 0.6];
figure
for ei=1:length(all_exp)
    ph=plot(fit{ei}.yCalc2,fit{ei}.x,'Color',colors(ei,:)) ; hold on    
     ph.LineStyle = LINESTYLES{ei};
    legend_string{ei}=output{ei}.exp_title;
end
for ei=1:length(all_exp)
    ph=plot(fit{ei}.yCalc1,fit{ei}.x,'Color',1.1*colors2(ei,:)) ; hold on  
    ph.LineStyle = LINESTYLES{ei};
    legend_string{ei}=output{ei}.exp_title;
end
set(gca, 'YDir','reverse')
xlabel('Time (hours)'); ylabel ('Time (days)');
xlim([0 24]);ylim([1 20]);    
legend(legend_string)
1