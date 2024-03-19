# TKM-master
Python code for paper "Individually fair k-means via tilting"
## Introduction

This repo holds the source code and scripts for reproducing the key experiments of our paper: Individually fair k-means via tilting.

## Datasets

Download the following datasets, and run our `data_process.py`, you can get the data format that can be used in our codes. 

|Datasets|  Description  | Source |
|-------- |------- |-------- |
|Athlete |                |       https://www.kaggle.com/heesoo37/120-years-of-olympic-history-athletes-and-results  |
|Bank    | Including 4,521 data points and 16 features, we select 3 numerical features: age, balance, and duration.  |  https://archive.ics.uci.edu/ml/datasets/bank+marketing |
|Census  |  Including 32,561 data points and 15 features, we select 5 numerical features: age, final-weight, education-num, capital-gain, hours-per-week.                |   https://archive.ics.uci.edu/ml/datasets/Adult|
|Diabetes |  Including 101,766 data points and 50 features, we select 3 numerical features: admission_source_id, time_in_hospital, num_medications.|https://archive.ics.uci.edu/ml/datasets/diabetes|
|Recruitment|    |https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring/|
|Spanish |         |https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring/|
|Student |      | https://analyse.kmi.open.ac.uk/open_dataset|
|3D-Spatial|Including 434,874 data points and 4 features, we select 3 numerical features: longitude, latitude, altitude.|https://archive.ics.uci.edu/dataset/246/3d+road+network+north+jutland+denmark    |
|Census1990|Including 2,458,285  data points and 69 features, we select 11 numerical features: dAncstry1, dAncstry2, iAvail, iCitizen, iClass, dDepart, iDisabl1, iDisabl2, iEnglish, iFeb55, iFertil. |     https://proceedings.neurips.cc/paper/2019/file/fc192b0c0d270dbf41870a63a8c76c2f-Paper|
|HMDA |         | https://ffiec.cfpb.gov/data-browser/|

Bank Including 4,521 data points and 16 features, we select 3 numerical features: age, balance, and duration.

Census  Including 32,561 data points and 15 features, we select 5 numerical features: age, final-weight, education-num, capital-gain, hours-per-week.

Diabetes  Including 101,766 data points and 50 features, we select 3 numerical features: admission_source_id, time_in_hospital, num_medications.

3D-spatial Including 434,874 data points and 4 features, we select 3 numerical features: longitude, latitude, altitude.

Census1990  Including 2,458,285  data points and 69 features, we select 11 numerical features: dAncstry1, dAncstry2, iAvail, iCitizen, iClass, dDepart, iDisabl1, iDisabl2, iEnglish, iFeb55, iFertil. 

HMDA  Including 5,986,660 data points and 53 features, we select 8 numerical features: agency_code, loan_type, loan_purpose, loan_amount_000s, preapproval, state_code, county_code, applicant_ethnicity.

Synthetic. We use the \texttt{make\_blobs} function in Scikit-learn \cite{Fabian2011scikit} to generate synthetic data, with 200 data points, 2 and 3 clusters respectively, and 2 dimensions.
