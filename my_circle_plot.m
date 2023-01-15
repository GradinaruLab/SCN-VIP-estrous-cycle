function [circ_params] = my_circle_plot(data,exp)
% plot 24 h data in a circle
% requires https://github.com/circstat/circstat-matlab
% returns the mean time of activity in hours
CIRC_PLOT_FIG=0;

X=[1:ceil(length(data)/360):length(data)];
Y=[];
for i=1:length(X)-1
    %Y=[Y nanmean(data(X(i):X(i+1),:),2)];
    Y=[Y nanmean(data(X(i):X(i+1)))];
end

% finds all the inds that are above the threshold
Yn=Y(~isnan(Y));
wf=2.25;
wf=2.3;
threshold=median(Yn)+wf*std(Yn);

% calculates angle and w (weights) which are above the threshold
angle=find(Y>threshold); 
w=Y(find(Y>threshold));
% allow to reduce the threshold if the array is empty 
while isempty(w);
    wf=wf-1;
    n_threshold=median(Yn)+wf*std(Yn);
    angle=find(Y>n_threshold);
    w=Y(find(Y>n_threshold));   
end

%converts to radians to calculate std, var, onset and offset, and convents
%back to angle
w(isnan(w))=1;
rad_data=circ_ang2rad(angle);
%Cmean=circ_mean(rad_data,w,2);% mean in rads
[s,s0]=circ_std(rad_data,w,[],2) ; % std in rads
Cvar = circ_var(rad_data, w, [], 2);% varience in rads
Avar=circ_rad2ang(Cvar);% Var in angles
onset=circ_rad2ang(rad_data(1)); 
offset=circ_rad2ang(rad_data(end)); 

n=0;
% for 'VIP_cre_DD_30min@8pm' should be 8 
% blue stim is very long- 7 doesn't fit. 4 
switch exp
    case 'VIP_cre_DD_30min@8pm'; offset_limit=9;
    case {'VIP_cre_DD_30min_blue','VIP_cre_DD_completeDD_Blue'}; offset_limit=4;
    otherwise offset_limit=7;
end
        
while 24-(offset/360)*24<offset_limit %is that general enough? - yes - data was pushed to ZT0 . was 7 
    n=n+1;
    offset=circ_rad2ang(rad_data(end-n)); 
end
n=2;
while (onset/360)*24<0.3% is that general enough? 
    n=n+1;
    onset=circ_rad2ang(rad_data(n)); 
end

% calculates angle and w (weights) which are above the threshold2 to
% calculate averaged activity 


% threshold2=median(Yn)+std(Yn);
% angle=find(Y>threshold2); 
% w=Y(find(Y>threshold2));
% w(isnan(w))=1;
%rad_data=circ_ang2rad(angle);
Cmean=circ_mean(rad_data,w,2);% mean in rads
Amean=circ_rad2ang(Cmean);% mean in angle
if Amean<0; Amean=360-abs(Amean); end
Hour_mean=(Amean/360)*24;

if CIRC_PLOT_FIG
    figure
   circ_plot(rad_data','hist',[],20,true,true,'linewidth',2,'color','r'); hold on 
  
end

circ_params.Hour_mean=Hour_mean;
circ_params.onset=(onset/360)*24;
circ_params.offset=(offset/360)*24;
