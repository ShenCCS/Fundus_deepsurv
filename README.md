# Fundus_deepsurv
## Fundus Image for Regression model then deepsurv model
>
## Enviroment
* python 3.7.5 
* conda env : retfound
* Model execution environment : Nvidia 3080Ti

## Data Preprocessing
> Making regression data route: /john/network/RETFound/RETFound_MAE-main/Fundus_deepsurv/Regression
* code: regressiondata.ipynb

## Regression model
> Using XGBoost model
* code: regressionTest.ipynb

## deepsurv dataset preprocessing
> Using XGBoost prediction then add other labels making deepsurv dataset
> Route: /john/network/RETFound/RETFound_MAE-main/Fundus_deepsurv/deepsurv
* code: deepsurvdata.ipynb

## deepsurv model
> Route: /john/network/RETFound/RETFound_MAE-main/Fundus_deepsurv/deepsurv
* code: DeepsurvModel.ipynb
