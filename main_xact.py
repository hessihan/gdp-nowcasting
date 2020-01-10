from xact import *
from advizX import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings
import pickle

warnings.simplefilter('ignore')
###############################################################################################
# 15 days before nowcasting
#########################################
#### Pseudo setting
## pseudo dataset
pseudo_dataset15 = Dataset(setting='pseudo', predict_period=30, valid_size=12, days_before=15)
pseudo_dataset15.build()
with open('pseudo_dataset15.pkl', 'wb') as f:
    pickle.dump(pseudo_dataset15, f)

## ar1
pseudo_ar115 = Model(method='ar1', dataset=pseudo_dataset15)
pseudo_ar115.execute()
with open('pseudo_ar115.pkl', 'wb') as f:
    pickle.dump(pseudo_ar115, f)
    
## lin reg
pseudo_lin_reg15 = Model(method='lin_reg', dataset=pseudo_dataset15)
pseudo_lin_reg15.execute()
with open('pseudo_lin_reg15.pkl', 'wb') as f:
    pickle.dump(pseudo_lin_reg15, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_lasso15 = Model(method='lasso', dataset=pseudo_dataset15)
pseudo_lasso15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_lasso15)
with open('pseudo_lasso15.pkl', 'wb') as f:
    pickle.dump(pseudo_lasso15, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_ridge15 = Model(method='ridge', dataset=pseudo_dataset15)
pseudo_ridge15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_ridge15)
with open('pseudo_ridge15.pkl', 'wb') as f:
    pickle.dump(pseudo_ridge15, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
pseudo_elastic15 = Model(method='elastic', dataset=pseudo_dataset15)
pseudo_elastic15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_elastic15.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(pseudo_elastic15, i)
with open('pseudo_elastic15.pkl', 'wb') as f:
    pickle.dump(pseudo_elastic15, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
pseudo_rf15 = Model(method='rf', dataset=pseudo_dataset15)
pseudo_rf15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_rf15.predict_period):
    adviz_X_Y_Zrmse(pseudo_rf15, i)
with open('pseudo_rf15.pkl', 'wb') as f:
    pickle.dump(pseudo_rf15, f)

#########################################
#### fully setting pseudo validation

## full pseudo validation dataset
full_pseudo_dataset15 = Dataset(setting='full', validation='pseudo', predict_period=30, valid_size=12, days_before=15)
full_pseudo_dataset15.build()
with open('full_pseudo_dataset15.pkl', 'wb') as f:
    pickle.dump(full_pseudo_dataset15, f)

## ar1
full_pseudo_ar115 = Model(method='ar1', dataset=full_pseudo_dataset15)
full_pseudo_ar115.execute()
with open('full_pseudo_ar115.pkl', 'wb') as f:
    pickle.dump(full_pseudo_ar115, f)

## lin reg
full_pseudo_lin_reg15 = Model(method='lin_reg', dataset=full_pseudo_dataset15)
full_pseudo_lin_reg15.execute()
with open('full_pseudo_lin_reg15.pkl', 'wb') as f:
    pickle.dump(full_pseudo_lin_reg15, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_lasso15 = Model(method='lasso', dataset=full_pseudo_dataset15)
full_pseudo_lasso15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_lasso15)
with open('full_pseudo_lasso15.pkl', 'wb') as f:
    pickle.dump(full_pseudo_lasso15, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_ridge15 = Model(method='ridge', dataset=full_pseudo_dataset15)
full_pseudo_ridge15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_ridge15)
with open('full_pseudo_ridge15.pkl', 'wb') as f:
    pickle.dump(full_pseudo_ridge15, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_pseudo_elastic15 = Model(method='elastic', dataset=full_pseudo_dataset15)
full_pseudo_elastic15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_elastic15.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_pseudo_elastic15, i)
with open('full_pseudo_elastic15.pkl', 'wb') as f:
    pickle.dump(full_pseudo_elastic15, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_pseudo_rf15 = Model(method='rf', dataset=full_pseudo_dataset15)
full_pseudo_rf15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_rf15.predict_period):
    adviz_X_Y_Zrmse(full_pseudo_rf15, i)
with open('full_pseudo_rf15.pkl', 'wb') as f:
    pickle.dump(full_pseudo_rf15, f)
    
## full full validation dataset
full_full_dataset15 = Dataset(setting='full', validation='full', predict_period=30, valid_size=12, days_before=15)
full_full_dataset15.build()
with open('full_full_dataset15.pkl', 'wb') as f:
    pickle.dump(full_full_dataset15, f)

## ar1
full_full_ar115 = Model(method='ar1', dataset=full_full_dataset15)
full_full_ar115.execute()
with open('full_full_ar115.pkl', 'wb') as f:
    pickle.dump(full_full_ar115, f)

## lin reg
full_full_lin_reg15 = Model(method='lin_reg', dataset=full_full_dataset15)
full_full_lin_reg15.execute()
with open('full_full_lin_reg15.pkl', 'wb') as f:
    pickle.dump(full_full_lin_reg15, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_full_lasso15 = Model(method='lasso', dataset=full_full_dataset15)
full_full_lasso15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_lasso15)
with open('full_full_lasso15.pkl', 'wb') as f:
    pickle.dump(full_full_lasso15, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_full_ridge15 = Model(method='ridge', dataset=full_full_dataset15)
full_full_ridge15.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_ridge15)
with open('full_full_ridge15.pkl', 'wb') as f:
    pickle.dump(full_full_ridge15, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_full_elastic15 = Model(method='elastic', dataset=full_full_dataset15)
full_full_elastic15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_elastic15.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_full_elastic15, i)
with open('full_full_elastic15.pkl', 'wb') as f:
    pickle.dump(full_full_elastic15, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_full_rf15 = Model(method='rf', dataset=full_full_dataset15)
full_full_rf15.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_rf15.predict_period):
    adviz_X_Y_Zrmse(full_full_rf15, i)
with open('full_full_rf15.pkl', 'wb') as f:
    pickle.dump(full_full_rf15, f)
    


###############################################################################################
# 45 days before nowcasting
#########################################
#### Pseudo setting
## pseudo dataset
pseudo_dataset45 = Dataset(setting='pseudo', predict_period=30, valid_size=12, days_before=45)
pseudo_dataset45.build()
with open('pseudo_dataset45.pkl', 'wb') as f:
    pickle.dump(pseudo_dataset45, f)

## ar1
pseudo_ar145 = Model(method='ar1', dataset=pseudo_dataset45)
pseudo_ar145.execute()
with open('pseudo_ar145.pkl', 'wb') as f:
    pickle.dump(pseudo_ar145, f)

## lin reg
pseudo_lin_reg45 = Model(method='lin_reg', dataset=pseudo_dataset45)
pseudo_lin_reg45.execute()
with open('pseudo_lin_reg45.pkl', 'wb') as f:
    pickle.dump(pseudo_lin_reg45, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_lasso45 = Model(method='lasso', dataset=pseudo_dataset45)
pseudo_lasso45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_lasso45)
with open('pseudo_lasso45.pkl', 'wb') as f:
    pickle.dump(pseudo_lasso45, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_ridge45 = Model(method='ridge', dataset=pseudo_dataset45)
pseudo_ridge45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_ridge45)
with open('pseudo_ridge45.pkl', 'wb') as f:
    pickle.dump(pseudo_ridge45, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
pseudo_elastic45 = Model(method='elastic', dataset=pseudo_dataset45)
pseudo_elastic45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_elastic45.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(pseudo_elastic45, i)
with open('pseudo_elastic45.pkl', 'wb') as f:
    pickle.dump(pseudo_elastic45, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
pseudo_rf45 = Model(method='rf', dataset=pseudo_dataset45)
pseudo_rf45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_rf45.predict_period):
    adviz_X_Y_Zrmse(pseudo_rf45, i)
with open('pseudo_rf45.pkl', 'wb') as f:
    pickle.dump(pseudo_rf45, f)

#########################################
#### fully setting pseudo validation

## full pseudo validation dataset
full_pseudo_dataset45 = Dataset(setting='full', validation='pseudo', predict_period=30, valid_size=12, days_before=45)
full_pseudo_dataset45.build()
with open('full_pseudo_dataset45.pkl', 'wb') as f:
    pickle.dump(full_pseudo_dataset45, f)
    
## ar1
full_pseudo_ar145 = Model(method='ar1', dataset=full_pseudo_dataset45)
full_pseudo_ar145.execute()
with open('full_pseudo_ar145.pkl', 'wb') as f:
    pickle.dump(full_pseudo_ar145, f)

## lin reg
full_pseudo_lin_reg45 = Model(method='lin_reg', dataset=full_pseudo_dataset45)
full_pseudo_lin_reg45.execute()
with open('full_pseudo_lin_reg45.pkl', 'wb') as f:
    pickle.dump(full_pseudo_lin_reg45, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_lasso45 = Model(method='lasso', dataset=full_pseudo_dataset45)
full_pseudo_lasso45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_lasso45)
with open('full_pseudo_lasso45.pkl', 'wb') as f:
    pickle.dump(full_pseudo_lasso45, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_ridge45 = Model(method='ridge', dataset=full_pseudo_dataset45)
full_pseudo_ridge45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_ridge45)
with open('full_pseudo_ridge45.pkl', 'wb') as f:
    pickle.dump(full_pseudo_ridge45, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_pseudo_elastic45 = Model(method='elastic', dataset=full_pseudo_dataset45)
full_pseudo_elastic45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_elastic45.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_pseudo_elastic45, i)
with open('full_pseudo_elastic45.pkl', 'wb') as f:
    pickle.dump(full_pseudo_elastic45, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_pseudo_rf45 = Model(method='rf', dataset=full_pseudo_dataset45)
full_pseudo_rf45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_rf45.predict_period):
    adviz_X_Y_Zrmse(full_pseudo_rf45, i)
with open('full_pseudo_rf45.pkl', 'wb') as f:
    pickle.dump(full_pseudo_rf45, f)

#########################################
#### fully setting full validation

## full full validation dataset
full_full_dataset45 = Dataset(setting='full', validation='full', predict_period=30, valid_size=12, days_before=45)
full_full_dataset45.build()
with open('full_full_dataset45.pkl', 'wb') as f:
    pickle.dump(full_full_dataset45, f)

## ar1
full_full_ar145 = Model(method='ar1', dataset=full_full_dataset45)
full_full_ar145.execute()
with open('full_full_ar145.pkl', 'wb') as f:
    pickle.dump(full_full_ar145, f)

## lin reg
full_full_lin_reg45 = Model(method='lin_reg', dataset=full_full_dataset45)
full_full_lin_reg45.execute()
with open('full_full_lin_reg45.pkl', 'wb') as f:
    pickle.dump(full_full_lin_reg45, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_full_lasso45 = Model(method='lasso', dataset=full_full_dataset45)
full_full_lasso45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_lasso45)
with open('full_full_lasso45.pkl', 'wb') as f:
    pickle.dump(full_full_lasso45, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_full_ridge45 = Model(method='ridge', dataset=full_full_dataset45)
full_full_ridge45.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_ridge45)
with open('full_full_ridge45.pkl', 'wb') as f:
    pickle.dump(full_full_ridge45, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_full_elastic45 = Model(method='elastic', dataset=full_full_dataset45)
full_full_elastic45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_elastic45.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_full_elastic45, i)
with open('full_full_elastic45.pkl', 'wb') as f:
    pickle.dump(full_full_elastic45, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_full_rf45 = Model(method='rf', dataset=full_full_dataset45)
full_full_rf45.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_rf45.predict_period):
    adviz_X_Y_Zrmse(full_full_rf45, i)
with open('full_full_rf45.pkl', 'wb') as f:
    pickle.dump(full_full_rf45, f)



###############################################################################################
# 75 days before nowcasting
#########################################
#### Pseudo setting
## pseudo dataset
pseudo_dataset75 = Dataset(setting='pseudo', predict_period=30, valid_size=12, days_before=75)
pseudo_dataset75.build()
with open('pseudo_dataset75.pkl', 'wb') as f:
    pickle.dump(pseudo_dataset75, f)

## ar1
pseudo_ar175 = Model(method='ar1', dataset=pseudo_dataset75)
pseudo_ar175.execute()
with open('pseudo_ar175.pkl', 'wb') as f:
    pickle.dump(pseudo_ar175, f)

## lin reg
pseudo_lin_reg75 = Model(method='lin_reg', dataset=pseudo_dataset75)
pseudo_lin_reg75.execute()
with open('pseudo_lin_reg75.pkl', 'wb') as f:
    pickle.dump(pseudo_lin_reg75, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_lasso75 = Model(method='lasso', dataset=pseudo_dataset75)
pseudo_lasso75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_lasso75)
with open('pseudo_lasso75.pkl', 'wb') as f:
    pickle.dump(pseudo_lasso75, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
pseudo_ridge75 = Model(method='ridge', dataset=pseudo_dataset75)
pseudo_ridge75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(pseudo_ridge75)
with open('pseudo_ridge75.pkl', 'wb') as f:
    pickle.dump(pseudo_ridge75, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
pseudo_elastic75 = Model(method='elastic', dataset=pseudo_dataset75)
pseudo_elastic75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_elastic75.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(pseudo_elastic75, i)
with open('pseudo_elastic75.pkl', 'wb') as f:
    pickle.dump(pseudo_elastic75, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
pseudo_rf75 = Model(method='rf', dataset=pseudo_dataset75)
pseudo_rf75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(pseudo_rf75.predict_period):
    adviz_X_Y_Zrmse(pseudo_rf75, i)
with open('pseudo_rf75.pkl', 'wb') as f:
    pickle.dump(pseudo_rf75, f)

#########################################
#### fully setting pseudo validation

## full pseudo validation dataset
full_pseudo_dataset75 = Dataset(setting='full', validation='pseudo', predict_period=30, valid_size=12, days_before=75)
full_pseudo_dataset75.build()
with open('full_pseudo_dataset75.pkl', 'wb') as f:
    pickle.dump(full_pseudo_dataset75, f)

## ar1
full_pseudo_ar175 = Model(method='ar1', dataset=full_pseudo_dataset75)
full_pseudo_ar175.execute()
with open('full_pseudo_ar175.pkl', 'wb') as f:
    pickle.dump(full_pseudo_ar175, f)

## lin reg
full_pseudo_lin_reg75 = Model(method='lin_reg', dataset=full_pseudo_dataset75)
full_pseudo_lin_reg75.execute()
with open('full_pseudo_lin_reg75.pkl', 'wb') as f:
    pickle.dump(full_pseudo_lin_reg75, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_lasso75 = Model(method='lasso', dataset=full_pseudo_dataset75)
full_pseudo_lasso75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_lasso75)
with open('full_pseudo_lasso75.pkl', 'wb') as f:
    pickle.dump(full_pseudo_lasso75, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_pseudo_ridge75 = Model(method='ridge', dataset=full_pseudo_dataset75)
full_pseudo_ridge75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_pseudo_ridge75)
with open('full_pseudo_ridge75.pkl', 'wb') as f:
    pickle.dump(full_pseudo_ridge75, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_pseudo_elastic75 = Model(method='elastic', dataset=full_pseudo_dataset75)
full_pseudo_elastic75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_elastic75.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_pseudo_elastic75, i)
with open('full_pseudo_elastic75.pkl', 'wb') as f:
    pickle.dump(full_pseudo_elastic75, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_pseudo_rf75 = Model(method='rf', dataset=full_pseudo_dataset75)
full_pseudo_rf75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_pseudo_rf75.predict_period):
    adviz_X_Y_Zrmse(full_pseudo_rf75, i)
with open('full_pseudo_rf75.pkl', 'wb') as f:
    pickle.dump(full_pseudo_rf75, f)

#########################################
#### fully setting full validation

## full full validation dataset
full_full_dataset75 = Dataset(setting='full', validation='full', predict_period=30, valid_size=12, days_before=75)
full_full_dataset75.build()
with open('full_full_dataset75.pkl', 'wb') as f:
    pickle.dump(full_full_dataset75, f)

## ar1
full_full_ar175 = Model(method='ar1', dataset=full_full_dataset75)
full_full_ar175.execute()
with open('full_full_ar175.pkl', 'wb') as f:
    pickle.dump(full_full_ar175, f)

## lin reg
full_full_lin_reg75 = Model(method='lin_reg', dataset=full_full_dataset75)
full_full_lin_reg75.execute()
with open('full_full_lin_reg75.pkl', 'wb') as f:
    pickle.dump(full_full_lin_reg75, f)

## lasso
hyparam_list1 = np.linspace(-10, 10, 21) # 10^-10 10^-9 ... 10^10
hyparam_list1 = list(10 ** hyparam_list1)
full_full_lasso75 = Model(method='lasso', dataset=full_full_dataset75)
full_full_lasso75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_lasso75)
with open('full_full_lasso75.pkl', 'wb') as f:
    pickle.dump(full_full_lasso75, f)

## ridge
hyparam_list1 = np.linspace(-10, 10, 21)
hyparam_list1 = list(10 ** hyparam_list1)
full_full_ridge75 = Model(method='ridge', dataset=full_full_dataset75)
full_full_ridge75.execute(hpl1=hyparam_list1)
adviz_Xlog10_Yperiod_Zrmse(full_full_ridge75)
with open('full_full_ridge75.pkl', 'wb') as f:
    pickle.dump(full_full_ridge75, f)

## elastic
hyparam_list1 = np.linspace(-10, 10, 21) 
hyparam_list1 = list(10 ** hyparam_list1)
hyparam_list2 = np.linspace(-10, 0, 11)  # 10^-10, 10^-9, ... 10^-1, 10^0
hyparam_list2 = list(10 ** hyparam_list2)
full_full_elastic75 = Model(method='elastic', dataset=full_full_dataset75)
full_full_elastic75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_elastic75.predict_period):
    adviz_Xlog10_Ylog10_Zrmse(full_full_elastic75, i)
with open('full_full_elastic75.pkl', 'wb') as f:
    pickle.dump(full_full_elastic75, f)

## random forest
hyparam_list1 = [50, 100, 150] # 50 100 150
hyparam_list2 = [1, 2, 3, 4, 5] # 1, 2, 3, 4, 5
full_full_rf75 = Model(method='rf', dataset=full_full_dataset75)
full_full_rf75.execute(hpl1=hyparam_list1, hpl2=hyparam_list2)
for i in range(full_full_rf75.predict_period):
    adviz_X_Y_Zrmse(full_full_rf75, i)
with open('full_full_rf75.pkl', 'wb') as f:
    pickle.dump(full_full_rf75, f)
