function  check_fft 
fs = 1.0173e+03;            % Sampling frequency   
fs = 1.0173e+03;            % Sampling frequency  
%fs=5000;
T = 1/fs;             % Sampling period       
L = 610361;             % Length of signal
t = (0:L-1)*T;        % Time vector

%Form a signal containing a 0.016 Hz sinusoid of amplitude 0.7 and a 1.5Hz sinusoid of amplitude 1.
for hi=[1:12]
   % dF(hi,:)= 10*sin(2*pi*0.016*t) + sin(2*pi*1.5*t);
     dF(hi,:)= 10*sin(2*pi*0.016*t) + sin(2*pi*1.5*t); 
     % white noise:
     %dF(hi,:) = wgn(1,L,0);
end
for hi=[13:24]
   % dF(hi,:)= 10*sin(2*pi*0.016*t) + sin(2*pi*1.5*t);
    dF(hi,:)= sin(2*pi*0.02*t) + 0.2*sin(2*pi*1.0*t);  
    % white noise
   %  dF(hi,:) = wgn(1,L,0);
end

t2 = downsample( t , 10 ) ;
figure
for hi=1:size(dF,1)
    dF2 = downsample( dF(hi,:) , 10 ) ;
    plot(t2,dF2+25*hi,'k'); hold on
end
ylim([-20 (hi+3)*25])
xlabel('Time (sec)')
ylabel('dF/F /hours')
title ('artificial data')
print('Artificial data','-depsc')    
%% fft
MdF=dF-mean(dF,2);% to identify better low frequencies 

Y = fft(MdF');
%Compute the two-sided spectrum P2. Then compute the single-sided spectrum P1 based on P2 and the even-valued signal length L.
L1=size(Y,1);
P2 = abs(Y/L1);
P1 = P2(1:L1/2+1,:);
P1(2:end-1,:) = 2*P1(2:end-1,:);
%Define the frequency domain f and plot the single-sided amplitude spectrum P1. The amplitudes are not exactly at 0.7 and 1, as expected, because of the added noise. On average, longer signals produce better frequency approximations.

% 1/11/22- find the max 
f = fs*(0:(L1/2))/L1;
figure;
subplot(2,1,1)
GW=0;
% B1 = smoothdata(P1,1,'gaussian',GW(wi));
B1=P1;
TF_all = islocalmax(B1);
%for di=1:size(B1,2)
for di=1:12:25
    % [pks,locs] = findpeaks(B1(:,di)) ;
    plot(f,B1(:,di)) ; hold on; xlim([0 2])
    plot(f(TF_all(:,di)),B1(TF_all(:,di),di)','r*')
end
xlabel('freq (Hz)')
ylabel('|P1(f)|')
title(['GW= ' num2str(GW)])

figure
subplot(2,1,2)
%for di=1:size(B1,2)
for di=1:12:24
    % [pks,locs] = findpeaks(B1(:,di)) ;
    plot(log10(f),log10(B1(:,di))) ; hold on; xlim([-2.2 3])
    plot(log10(f(TF_all(:,di))),log10(B1(TF_all(:,di),di))','r*')
end
xlabel('Log10 (freq (Hz))')
ylabel('Log10 (|P1(f)|)')
title(['GW= ' num2str(GW)])


new_f_limits=[0 0.03; 0.03 0.1; 0.1 0.35; 0.35 0.65; 0.65 1.0; 1.0 1.35 ; 1.35 1.65; 1.65 2.3; 2.3 4];
new_f_limits=flip(new_f_limits,1);
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

clims=[0 20];
   
figure
%image( B1_int,'CDataMapping','scaled')
imagesc( B1_int,clims)
colorbar
yticks([1: size(B1_int,1)])
yticklabels(num2str(new_f_limits))
xlabel('Time (hours)')
ylabel('Int. Power fft')
print('Artificial data FFT matrix','-depsc')  
% figure
% subplot(2,1,1) ; plot(f(1:end-1),diff(P1(:,[1:4,16:24])),'k'); hold on; xlim([1 5])
% subplot(2,1,2) ; plot(f(1:end-1),diff(P1(:,[5:16])),'k'); hold on; xlim([1 5])
%title(['VIPGC' mouse_info.ID ' ' trial_info.date ' sess ' num2str(trial_info.sess_num) ' ' trial_info.estrus]) 
