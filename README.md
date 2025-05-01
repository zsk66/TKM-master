# TKM-master
Python code for paper "Highly-Efficient Large-Scale k-means with Individual Fairness"
## Introduction

This repo holds the source code and scripts for reproducing the key experiments of our paper.

## Datasets

Download the following datasets, and run our `data_process.py`, you can get the data format that can be used in our codes. 

|Datasets|  Description  | Source |
|-------- |------- |-------- |
|Athlete |    Including 271,117 data points and 15 features, we select 3 numerical features: Age, Height, Weight.          |       https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results  |
|Bank    | Including 4,521 data points and 16 features, we select 3 numerical features: age, balance, and duration.  |  https://archive.ics.uci.edu/ml/datasets/bank+marketing |
|Census  |  Including 32,561 data points and 15 features, we select 5 numerical features: age, final-weight, education-num, capital-gain, hours-per-week.                |   https://archive.ics.uci.edu/ml/datasets/Adult|
|Diabetes |  Including 101,766 data points and 50 features, we select 2 numerical and 2 text features: admission_source_id, time_in_hospital, num_medications.|https://archive.ics.uci.edu/ml/datasets/diabetes|
|Recruitment| Including 4,001 data points and 50 features, we select 3 numerical features: age,ind-university_grade,ind-languages.   |https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring/|
|Spanish | Including 4,747 data points and 15 features, we select 6 numerical features: NPcreated,ns_talk,ns_userTalk,C_man,E_NEds,E_Bpag.          |https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring/|
|Student |Including 32,594 data points and 21 features, we select 3 numerical features: age_band,studied_credits,num_of_prev_attempts,gender.      |https://analyse.kmi.open.ac.uk/open_dataset|
|3D-Spatial|Including 434,874 data points and 4 features, we select 3 numerical features: longitude, latitude, altitude.|https://archive.ics.uci.edu/dataset/246/3d+road+network+north+jutland+denmark    |
|Census1990|Including 2,458,285  data points and 69 features, we select 11 numerical features: dAncstry1, dAncstry2, iAvail, iCitizen, iClass, dDepart, iDisabl1, iDisabl2, iEnglish, iFeb55, iFertil. | https://proceedings.neurips.cc/paper/2019/file/fc192b0c0d270dbf41870a63a8c76c2f-Paper|
|HMDA |Including 5,986,660 data points and 53 features, we select 8 numerical features: agency_code, loan_type, loan_purpose, loan_amount_000s, preapproval, state_code, county_code, applicant_ethnicity. | https://ffiec.cfpb.gov/data-browser/|
|Argoverse |Including more than 12,000,000 data points and 2 features: Latitude and Longitude. | https://openaccess.thecvf.com/content_CVPR_2019/papers/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.pdf |

## Start

### Visualization

Run `plot_toy_k=2` and `plot_toy_k=3` directly, then you can get the visualization figures in our experiment.
<img width="880" alt="toy_example" src="./visualization.pdf">


### Comparison among Various Methods and Parameters

After downloading and processing the raw dataset, then directly run `main.py`, then you get the results of TKM.

Run `Comparison_main` in individually-fair-k-clustering-main, which we forked from https://github.com/moonin12/individually-fair-k-clustering, and you get the results of our comparison methods.

Then, running the codes in plot_figures, you get all of the figures in our experiments.
