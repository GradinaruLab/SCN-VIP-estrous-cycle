function light_array=find_light_status_by_figure(all_dF,full_Sname,files1)

if ~isfield(all_dF,'TTL')
    TTL=[];
else
    TTL=all_dF.TTL;
end

Sname=full_Sname(1:end-1);
if str2num(Sname(end))>0
  Sname=full_Sname(1:end-2);  
end    
if contains(full_Sname,'test6Rblue') 
    Sname='test6Rblue';
end
if contains(full_Sname,'test6RredXXhigh') || contains(full_Sname,'test6RredXmed') || contains(full_Sname,'test6Rredmed')|| contains(full_Sname,'test6RredXXlow')|| contains(full_Sname,'test6RredXlow')||contains(full_Sname,'_438')||contains(full_Sname,'_650')
    Sname='test6RwithTTL_on_off';
end

%TTL=df.TTL;
Ex=exist(['D:\DATA_Glab\fiberphotometry\TDT_FP\light_array\' files1 '_light.mat']);
if Ex>0
    load(['D:\DATA_Glab\fiberphotometry\TDT_FP\light_array\' files1 '_light'],'light_array')
else
   
    if contains(full_Sname,'test6Rred') || isempty(TTL)
        if ~contains(Sname,'test6RwithTTL_on_off')
            figure
            plot(all_dF.t,all_dF.data(1,:)); hold on
            title('Use the cursor to choose the first light event')
            [x,y] = ginput(1); % choose 1 cursors which is the begining of the first light on
            
            disp([num2str(x) ' sec: start of red light'])
            phase=0;% might need adjustment for specific mice
            light_array.light_off=[x+15:45:x+45*5+15]'-phase;
            light_array.light_on= [x:45:x+45*5]'-phase;
            light_array.exp='test6Rred';
            save(['D:\DATA_Glab\fiberphotometry\TDT_FP\light_array\' files1 '_light'],'light_array')
            return
        end
    end
    switch Sname
        case 'test';           N=2;   XMAX=320;
        case {'test6R';'test6Rred';'test6Rblue'}; N=12;   XMAX=460;
        case  {'Sess';'LDold'};  N=1;   XMAX=8000;
        case {'DL';'DLold'};     N=1;   XMAX=3600;
        case 'SessOnset';        N=0;   XMAX=1800;
            
    end
    
    switch Sname
        case  {'DL';'Sess';'test6R';'test6Rred';'LDold';'DLold';'test'}
            dTTL=diff(TTL);
            k=find(dTTL>mean(dTTL));
            h=figure;
            plot(TTL,ones(1,length(TTL)),'X'); hold on
            plot(TTL,zeros(1,length(TTL))+0.9,'X'); hold on 
            plot(TTL(k),zeros(1,length(k))+0.8,'Xk'); hold on
            plot(all_dF.t,(all_dF.data(1,:))/max(all_dF.data(1,:))); hold on
            ylim([0.7,1.05])
            xlim([0 XMAX])
            opts.Interpreter = 'tex';
            opts.Default= 'identified onset';
            answer = questdlg('Use identified onset or sign manually?','Boundary Condition',...
                  'identified onset','Manually',opts);
              switch answer
                  case  'Manually'
                      title('test 6R choose DARK start time first, i.e. TTL is on, and then light start. others- choose change')
                      disp('test 6R choose DARK start time first, i.e. TTL is on, and then light start. others- choose change')
                      [x,y] = ginput(N); % choose 2 cursors
                  case 'identified onset'
                      
                      x=[TTL(k)+15; TTL(k)];
                     % x=TTL(k(2:7));
                     %x=[TTL(k(2:7))+15; x];
              end
    end
    
    %close (h)
    
    switch Sname
        case 'test'
            light_array.light_off=x(1);
            light_array.light_on=x(2);
        case {'test6R';'test6Rred'}
            light_array.exp=Sname;
            light_array.light_off=x(1:6);
            light_array.light_on=x(7:12);
          
        case {'test6Rblue','test6RwithTTL_on_off'}
                       
            light_array.exp=Sname;
            light_array.light_on=TTL-1;
            light_array.light_off=TTL-1+15;
            
        case  {'DL';'DLold'}
            light_array.light_off=1;
            light_array.light_on=x;
            light_array.exp=Sname;
        case {'Sess';'LDold'}
            light_array.light_off=x;
            light_array.light_on=1;
            light_array.exp=Sname;
        case 'SessOnset'
            light_array.light_off=nan;
            light_array.light_on=1;
            light_array.exp=Sname;
    end
    
    save(['D:\DATA_Glab\fiberphotometry\TDT_FP\light_array\' files1 '_light'],'light_array')
end





end
