# SCN-VIP-estrous-cycle
Anat Kahan, Aug 2022 

Estrous cycle regulation experiments: (Figure 1, 2, 6, S6) 

[y]=read_estrus_log_table
read .xlsx file with estrous states and creates output y

estrous_analysis_Table (y)
read y and analyze estrous-cycle parameters and plot it 



Virus transduction and efficiency (Figure 2)

GnRHcas9_KO.m
read .xlsx file with data collected with Imaris 'spot' function 


FP 24/7 10 minutes per hour (Figure 3)
get_time_series_FP_per_mouse
analyse dataset of one mouse 

get_time_series_FP
read the whole dataset 

FP 24/7 FFT cross-validation classification (Figure 4, S4) 
FP_FFT_output_for_classifier  
run classification after 

FP ZT10-13 (Figure 5, S5)
get_LDtransition_FP_per_mouse
read data set for one mouse

get_LDtransition_FP_all
Compare the whole dataset, estrous-cycle dependent 

get_LDtransition_FP_male_female 
Compare males-females only 

FP_LDtransition_FFT_output_for_classifier 
used to run classification after "get_LDtransition_FP_per_mouse" is done for all the mice in the dataset


Oocyte release (figure 6)
oocytes_quantification 
reads .xlsx file with the quantified data

