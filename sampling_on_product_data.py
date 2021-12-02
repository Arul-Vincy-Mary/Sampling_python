

import os
os.chdir("E:/python")
import numpy as np
import pandas as pd

#READ THE DATAFRAME
samp = pd.read_csv('prod.csv',index_col=0)
#SAVE THE MEAN IN NEW VARIABLE
real_mean = round(samp['Measure'].mean(),3)
c = real_mean

#SIMPLE RANDOM SAMPLING
srs = samp.sample(n=4).sort_values(by='Product_id')
#SAVE THE MEAN IN NEW VARIABLE
a = round(srs['Measure'].mean(),3)

#SYSTEMATIC SAMPLING
def systematic_sampling (samp,step):
    sys = np.arange(0,len(samp),step)
    sys_sample = samp.iloc[sys]
    return sys_sample
#OBTAIN AND SAVE IT IN A NEW VARIABLE
sys_sample = systematic_sampling(samp,3)
#SAVE THE MEAN IN A NEW VARIABLE
b = round(sys_sample['Measure'].mean(),3)

#STRATIFIED RANDOM SAMPLING
strata_sample = pd.read_csv('prod2.csv')
#IMPORT THE StratifiedShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=5)
for x,y in split.split(strata_sample,strata_sample['Strata']):
    stratified_random_sample = strata_sample.iloc[y].sort_values(by='Product_id')
    stratified_random_sample
#OBTAIN THE SAMPLE MEAN FOR EACH GROUP
stratified_random_sample.groupby('Strata').mean().drop(['Product_id'],axis=1)

#comparing the mean of systematic and startified sampling
#ABS ERROR(create a dict to find the abs_error)
outcomes = {'sample_mean':[a,b],'real_mean':c}
#TRANSFORM THE DICT INTO A DATAFRAME
outcomes = pd.DataFrame(outcomes,index=['simple random sampling','systematic sampling'])
outcomes['abs_error'] = abs(outcomes['real_mean'] - outcomes['sample_mean'])
outcomes.sort_values(by='abs_error')
