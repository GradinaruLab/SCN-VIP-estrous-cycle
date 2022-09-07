function [ALL_colors,color_ind]=get_estrus_colors(estrus_states_titles);
% set colors for figures that involves different estrus states

ALL_estrus_names={'M-D','P-E','Metestrus','M','Proestrus','P','Estrus','E','RE','Female Estrus','Diestrus','D','NR','Female Diestrus','P-2','P-1','P+0','P+1','P+2','OVX-D','OVX','OVX+E+P','OVX+P+E','OVX+Esr','OVX+PR','Female','Male','male','Male Progesterone','NA'};
ALL_colors= [0.8500, 0.3250, 0.0980;
    0.95, 0.15, 0;
    0.8500, 0.3250, 0.0980;
    0.8500, 0.3250, 0.0980;
    0.95, 0.15, 0;
    0.95, 0.15, 0;
    0.9290, 0.6940, 0.1250;
    0.9290, 0.6940, 0.1250;
    0.9290, 0.6940, 0.1550;%Female RE: receptive
    0.9290, 0.6940, 0.1550;%Female Estrus
    0.7850, 0.1780, 0.1;
     0.7850, 0.1780, 0.1;
     0.7450, 0.1180, 0.25;% female NR non-receptive
     0.7450, 0.1180, 0.25;% female diestrus
    0.2500, 0.50, 0.4;% P-2
    0.65, 0.45, 0.3;%P-1
    0.8500, 0.3250, 0.0980;% P+0
    0.9290, 0.6940, 0.1550;% P+1
    0.7450, 0.1180, 0.25;%P+2
    0.3940, 0.2340, 0.650;
    0.3940, 0.2340, 0.650;
    0.62 0.12 0.62;
    0.85 0 0.8;
    0.85 0.2 0.8;
    0.85 0.4 0.8;
    1, 0, 0;
    0, 0.4470, 0.7410;
    0, 0.4470, 0.7410;
    0.1, 0.4470, 0.8410;
    0.4, 0.4 0.4];
clear color_ind
for k=1:length(estrus_states_titles)
    color_ind(k)=strmatch(estrus_states_titles{k},ALL_estrus_names,'exact');
end
