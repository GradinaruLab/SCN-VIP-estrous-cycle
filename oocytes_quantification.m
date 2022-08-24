function oocytes_quantification 
% compare light manipulated females with regular day cycle females
% June 2021 Anat Kahan 

all_T=readtable('C:\Users\anatk\Documents\Data_Glab_home_work\Behavioral\data\Oocytes in light manipulated females.xlsx');

T=all_T(find(all_T.include_final==1),:);
conditions={'ctrl','exp1','rescue1','rescue2'};
conditions_names={'LD','DD+L1','DD+L1+L3','back to LD'};
for i=1:length(conditions)
    N{i}=T.numberOfOocytes(find(strcmp(T.exp_type,conditions{i})));
end
% N{2}=[T.numberOfOocytes(find(strcmp(T.exp_type,'exp1'))) T.numberOfOocytes(find(strcmp(T.exp_type,'exp2')))];% DD+L1 or DD+L1+L2
% N{3}=T.numberOfOocytes(find(strcmp(T.exp_type,'rescue1')));% DD+L1+L3
% N{4}=T.numberOfOocytes(find(strcmp(T.exp_type,'rescue2')));% back to DL


x=[];
g=[];
num_el_each=[];
for i=1:length(N)
    x=[x N{i}'];
    g=[g i*ones(1,length(N{i}))];
    num_el_each=[num_el_each length(N{i})];
end
F = repelem(conditions_names, num_el_each);

% check if data is normally distributed
for i=1:length(N)
    h(i) = kstest(N{i});
    if length(N{i})>3
        h_a(i) = adtest(N{i});
    end
end
% startistical test for non- normally distribution 
[p,tbl,stats] = kruskalwallis(x,g); % also returns the ANOVA table as the cell array tbl and the structure stats containing information about the test statistics.
c = multcompare(stats,'CType','bonferroni');
%c = multcompare(stats,'CType','dunn-sidak');
%c = multcompare(stats,'CType','hsd');

disp(c)
figure
boxplot(x,g, ...
    'Labels', conditions_names, ...
    'Colors',[0 0 0],'PlotStyle','compact'); hold on
plotSpread(N,'categoryIdx',g,'categoryColors',{'r','r','r','r'}); hold on
ylabel('number of oocytes')
set(gca, 'XTickLabel',conditions_names)

disp(conditions)
disp(N)
1
% data = [randn(50,1);randn(50,1)+3.5]*[1 1];
%            catIdx = [ones(50,1);zeros(50,1);randi([0,1],[100,1])];
%            figure
%            plotSpread(data,'categoryIdx',catIdx,...
%                 'categoryMarkers',{'o','+'},'categoryColors',{'r','b'})

cd('Z:\Anat\behavioral\data')
print('Oocytes_quantification','-depsc')

