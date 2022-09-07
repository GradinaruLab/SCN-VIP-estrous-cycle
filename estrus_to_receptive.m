function new_estrus_states=estrus_to_receptive(estrus_states);
% this function is used to create a new array that tranform estrus stages
% to receptive or non-receptive
clear new_estrus_states
for sti=1:length(estrus_states)
    switch estrus_states{sti}
        case {'E','P','P+0','P+1','P+2'}
            % case {'E','P','P+0'}
            new_estrus_states{sti}='RE';% receptive
        case {'M','D','P-2','P-1'}
            new_estrus_states{sti}='NR';     % non-receptive
        case 'OVX'
            new_estrus_states{sti}='OVX';
        case 'OVX+PR'
            new_estrus_states{sti}='OVX+PR';
        case 'OVX+Esr'
            new_estrus_states{sti}='OVX+Esr';
        case {'Male','male'}
            new_estrus_states{sti}='Male';
        otherwise
            new_estrus_states{sti}='';
    end
end

new_estrus_states=new_estrus_states';
end