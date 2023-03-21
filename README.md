# SCN-VIP-estrous-cycle
Anat Kahan, Jan 2023

Estrous cycle regulation experiments: 

[y]=read_estrus_log_table
read .xlsx file with estrous states and creates output y

estrous_analysis_Table (y)
read y and analyze estrous-cycle parameters and plot it 



Virus transduction and efficiency for GnRH-Cas9 experiment 

GnRHcas9_KO.m
read .xlsx file with data collected with Imaris 'spot' function 


FP 24/7 10 minutes per hour
SynFP_get_dF_time_series;
get_time_series_FP_single_trial;
get data of one trail and calculates dF/F

get_time_series_FP_per_mouse;
analyse dataset of one mouse

get_time_series_FP;
read the whole dataset 

FP 24/7 FFT cross-validation classification
FP_FFT_output_for_classifier ; 
run classification after 

FP ZT10-13
SynFP_get_dF
get_LDtransition_FP_single_trial_v2;
read data per trial
get_LDtransition_FP_per_mouse;
read data set for one mouse

get_LDtransition_FP_all; 
Compare the whole dataset, estrous-cycle dependent 

get_LDtransition_FP_male_female 
Compare males-females only 

FP_LDtransition_FFT_output_for_classifier ;
used to run classification after "get_LDtransition_FP_per_mouse" is done for all the mice in the dataset


Oocyte release
oocytes_quantification ;
reads .xlsx file with the quantified data

Locomotor activity: 
run_get_compass_data_v3
uses:
get_compass_data_v3
my_circle_plot
