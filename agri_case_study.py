# Data description:

#The data refers to district wise, crop wise, season wise and year wise data on crop covered area (Hectare) and production (Tonnes) 
#The data is being used to study and analyse crop production, production contribution to district/state/country, agro-climatic zone wise performance, and high yield production order for crops, crop growing pattern and diversification
#The system is also a vital source for formulating crop related schemes and assessing their impacts


import os
os.chdir("D:/python")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
#read data
case = pd.read_csv("agri.csv")
case.info()

#unique vaules
np.unique(case['State_Name'],return_counts=True)
np.unique(case['Season'],return_counts=True)
np.unique(case['Crop'],return_counts=True)
np.unique(case['District_Name'],return_counts=True)
np.unique(case['Crop_Year'],return_counts=True)
np.unique(case['Area'],return_counts=True)
np.unique(case['Production'],return_counts=True)

#replace nan values as zero
case.dropna(axis=0,inplace=True)
#dtype for production
case.Production = case.Production.astype('int64')
#(6.51e+07 = 6.51*10^7 = 6.51*10000000 = 65100000) 
np.array_equal(case.Production, case.Production.astype('int64'))
#dtype for area
case.Area = case.Area.astype('int64')
np.array_equal(case.Area,case.Area.astype(float))
case.dtypes
#drop the rows which are contain zero in Area
index_names = case[ case['Area'] == 0 ].index 
case.drop(index_names, inplace = True) 
case.drop_duplicates(subset=None, keep='first', inplace=False)
#case
check = (case==0).any(axis=0)
case.isnull().sum()
desc = case.describe()


#DATA VISUALIZATION
#Production - Statewise
state = case.groupby("State_Name")['Production'].sum().reset_index().sort_values("Production",ascending=False)
state[:5]
state.tail(5)
sns.barplot('Production','State_Name',data=state)
#From the above plot ,Kerala is the highest crops producing state overall
Kerala = case[case["State_Name"]=="Kerala"]
Kerala.shape
#Find the highest crops producing district in kerala
kerala_dist = Kerala.groupby("District_Name")['Production'].sum().reset_index().sort_values("Production",ascending=False)
kerala_dist[:5]
kerala_dist.tail(5)
sns.barplot('Production','District_Name',data=kerala_dist)
#kozhicode is the highest crops producing district in kerala
#Find which crop is more productive in the state
kerala_crop = Kerala.groupby("Crop")['Production'].sum().reset_index().sort_values("Production",ascending=False)
kerala_crop[:5]
kerala_crop.tail(5)
sns.barplot('Production','Crop',data=kerala_crop)
#coconut is the most produced crop in kerala
#To find which year has produced the most in kerala
kerala_year = Kerala.groupby("Crop_Year")['Production'].sum().reset_index().sort_values("Production",ascending=False)
kerala_year[:5]
kerala_year.tail(5)
sns.barplot('Crop_Year','Production',data=kerala_year)
#2005 is the more produced year in kerala
#In which season kerala produced more
kerala_sea = Kerala.groupby("Season")['Production'].sum().reset_index().sort_values("Production",ascending=False)
kerala_sea[:5]
sns.barplot('Season','Production',data=kerala_sea)
sns.barplot('Season','Production',data=Kerala)
#kerala has produced more in whole year so, there is no specific reason
#we can do this for any states in the data set, i did the highest produced state.

#Production - yearwise(overall)
year = case.groupby('Crop_Year')['Production'].sum().reset_index().sort_values("Production",ascending=False)
year[:5]
sns.barplot("Crop_Year",'Production',data=year)
#2011 is the more produced year overall

#Production - cropwise(overall)
crop = case.groupby("Crop")['Production'].sum().reset_index().sort_values("Production",ascending=False)
crop[:5]
crop.tail(5)
sns.barplot('Production','Crop',data=crop)
#coconut is the most produced crop in country
#least produced 5 crops overall
crop[crop['Production']==0]

#Production - Season(overall)
sns.barplot('Season','Production',data=case)


#production by  Area(overall)
sns.regplot(x=case['Area'],y=case['Production'])
Area = case.groupby("State_Name")['Production','Area'].sum().reset_index().sort_values("Production",ascending=False)
Area[:5]
Area['Prod_by_area'] = Area['Production']/Area['Area']                                                             
sns.barplot("Prod_by_area","State_Name",data=Area)
#Kerala is the most productive state when we comapre in terms of production by area

#production - districtwise(overall)
dist = case.groupby("District_Name")['Production'].sum().reset_index().sort_values("Production",ascending=False)
dist[:5]
dist.tail(5)
sns.barplot('Production','District_Name',data=dist)
#production by area in districtwise(overall)
dist_area = case.groupby("District_Name")['Production','Area'].sum().reset_index().sort_values("Production",ascending=False)
dist_area[:5]
dist_area.tail(5)
dist_area['dist_Prod_by_area'] = dist_area['Production']/dist_area['Area']  
sns.barplot('dist_Prod_by_area','District_Name',data=dist_area)

#From the above plots,
         #highly produced state - kerala(statewise)
         #highly productive crop - coconut(cropwise)
         #highly productive year - 2011(yearwise)
         #highly produced state in terms 
               of production by area - kerala
         #highest produced season - wholeyear
      #statewise(highest produced state in country)
          #highly produced district in kerala - kozhicode
          #highly productive year in kerala - 2005
          #highly productive crop in kerla - coconut
          #highly productive season in kerala - whole year
            









