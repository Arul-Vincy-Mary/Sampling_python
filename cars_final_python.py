# problem statement :
#Predict the price of the used cars based on their attributes:



import os
os.chdir('D:/python')
import pandas as pd
import numpy as np
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
#reading data
cars1 = pd.read_csv("cars_sampled.csv")
cars=cars1.copy()
cars.head(5)
cars.tail(5)
cars.shape
cars.info()
cars.describe()

#supress scientific notation 
pd.options.display.float_format = '{:.3f}'.format
cars.describe()

#display maximum number of columns
pd.set_option('display.max_columns',100)
cars.describe()
 
#drop the unwanted columns
cars.columns
cars.drop(['dateCrawled','name','dateCreated','postalCode','lastSeen'], axis = 1, inplace = True) 
#drop duplicates and null values
cars.drop_duplicates(keep='first',inplace=True)
cars.isnull().sum()


#Identify the variables
#seller
seller = cars['seller'].value_counts()
pd.crosstab(cars['seller'],columns='count',normalize=True)
sns.countplot(x='seller',data=cars)
#does not affect price

#offerType
offer = cars['offerType'].value_counts()
pd.crosstab(cars['offerType'],columns='count',normalize=True)
sns.countplot(x='offerType',data=cars)
#does not affect price


#abtest
abtest = cars['abtest'].value_counts()
pd.crosstab(cars['abtest'],columns='count',normalize=True)
sns.countplot(x='abtest',data=cars)
sns.boxplot(x='abtest',y='price',data=cars)
#does not affect price

#vehicleType
vehicle = cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'],columns='count',normalize=True)
sns.countplot(x='vehicleType',data=cars)
sns.boxplot(x='vehicleType',y='price',data=cars)
#affects price

#gearbox
gearbox = cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'],columns='count',normalize=True)
sns.countplot(x='gearbox',data=cars)
sns.boxplot(x='gearbox',y='price',data=cars)
#affets price

#model
model = cars['model'].value_counts()
pd.crosstab(cars['model'],columns='count',normalize=True)
sns.countplot(x='model',data=cars)
sns.boxplot(x='model',y='price',data=cars)
#affects price
#many model

#kilometer
kilometer = cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'],columns='count',normalize=True)
sns.boxplot(x='kilometer',y='price',data=cars)
#affects price

#fuelType
fuel = cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'],columns='count',normalize=True)
sns.countplot(x='fuelType',data=cars)
sns.boxplot(x='fuelType',y='price',data=cars)
#affects price

#brand
brand = cars['brand'].value_counts()
pd.crosstab(cars['brand'],columns='count',normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand',y='price',data=cars)

#notRepairedDamage
damage = cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'],columns='count',normalize=True)
sns.countplot(x='notRepairedDamage',data=cars)
sns.boxplot(x='notRepairedDamage',y='price',data=cars)

#removing the columns which are doesn't affect the price
cars.drop(['seller','abtest','offerType'], axis =1, inplace = True) 




#price
price = cars['price'].value_counts().sort_index()
sns.boxplot(y=cars['price']) #many outliers and no box only line
price.head(5)
price.tail(5)
sum(cars['price']>150000)
sum(cars['price']<100)


#powerPs
power = cars['powerPS'].value_counts().sort_index()
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',data=cars)
sum(cars['powerPS']>500)
sum(cars['powerPS']<10)

#yearOfRegistration
year = cars['yearOfRegistration'].value_counts().sort_index()
sns.regplot(x='yearOfRegistration',y='price',data=cars)
sum(cars['yearOfRegistration']>2019)
sum(cars['yearOfRegistration']<1950)

#working range of data
cars = cars[(cars.yearOfRegistration<=2019)&
            (cars.yearOfRegistration>=1950)&
            (cars.price>=100)&
            (cars.price<=150000)&
            (cars.powerPS>=10)&
            (cars.powerPS<=500)]

#combine year and month to know the age of the car
cars['monthOfRegistration']/=12
#creating new column year1 by adding year and month
cars['year1']=(2019-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['year1']=round(cars['year1'],2)
cars['year1'].describe()

#drop month and year
 cars.drop(['yearOfRegistration','monthOfRegistration'],axis=1,inplace=True)
 

#visualization part for price,powerPS, Age
#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])
sns.regplot(x='powerPS',y='price',fit_reg=False,data=cars)

#year1
sns.distplot(cars['year1'])
sns.boxplot(y=cars['year1'])
sns.regplot(x='year1',y='price',fit_reg=False,data=cars)
#cars price higher are newer
#increases yea, price decreases except some vehicles

#correlation between the variables
cars2 = cars.select_dtypes(exclude=[object])
correlation = cars2.corr()
round(correlation,3)
cars2.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


#MODEL BUILDING
#Linear Regression
#Building a model using two types of data set
# 1. data obtained by omitting rows with any missing value
# 2. data obtained by imputing the missing value


#Sklearn - package to split data into train & test  
from sklearn.model_selection import train_test_split
# Sklearn - package to use linear regression
from sklearn.linear_model import LinearRegression
# Sklearn - package for performance metric 
from sklearn.metrics import mean_squared_error


#omit missing values
omit_missing_values = cars.dropna(axis=0)
#converting string into dummy variables
omit_missing_values = pd.get_dummies(omit_missing_values,drop_first=True)

#separating input and output
x1 = omit_missing_values.drop(['price'],axis=1,inplace=False)
y1 = omit_missing_values['price']

#plotting price
price1 = pd.DataFrame({'1.Before':y1 , "2.After":np.log(y1)})
price1.hist()

#transform price as a logarithmic value
y1 = np.log(y1)

# Splitting test & train as 30% and 70%
train_x, test_x, train_y, test_y = train_test_split(x1,y1,test_size=0.3,random_state=3)

#Baseline Model
"""
We are making a base model by using test data mean value
to keep as benchmark to compare with our regression model
"""

# finding the mean for test data value
base_pred = np.mean(test_y)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(test_y))

# finding the RMSE
base_RMSE =np.sqrt(mean_squared_error(test_y,base_pred))                           
print(base_RMSE)

#setting intercept as true
lgr = LinearRegression(fit_intercept=True)

# Fitting the train data for linear model
x = lgr.fit(train_x,train_y)

#Predicting the model on test data
predicted_lgr = lgr.predict(test_x)

# RMSE of the linear model
lr_mse= mean_squared_error(test_y,predicted_lgr)
lr_rmse = np.sqrt(lr_mse)
print(lr_rmse)

#R squared value
r2_test = x.score(test_x,test_y)
r2_train = x.score(train_x,train_y)
print(r2_test,r2_train)

#Residuals
residuals = test_y - predicted_lgr 
sns.regplot(x=predicted_lgr, y=residuals, fit_reg=False)
residuals.describe()


#MODEL BUILDING WITH IMPUTING THE MISSING VALUES

imput_missing_values = cars.apply(lambda x:x.fillna(x.median())\
                                    if x.dtypes == 'float'else\
                                     x.fillna(x.value_counts().index[0]))
#checking null values
imput_missing_values.isnull().sum()

#converting string into dummy variable
imput_missing_values = pd.get_dummies(imput_missing_values,drop_first=True)

#separating input and output
x2 = imput_missing_values.drop(['price'],axis=1,inplace=False)
y2 = imput_missing_values['price']

#plotting price
price1 = pd.DataFrame({'1.Before':y2 , "2.After":np.log(y2)})
price1.hist()

#transform price as a logarithmic value
y2 = np.log(y2)

# Splitting test & train as 30% and 70%
train1_x, test1_x, train1_y, test1_y = train_test_split(x2,y2,test_size=0.3,random_state=3)

#Baseline Model
# finding the mean for test data value
base_pred = np.mean(test1_y)
print(base_pred)

# Repeating same value till length of test data
base_pred = np.repeat(base_pred, len(test1_y))

# finding the RMSE
base_RMSE2 =np.sqrt(mean_squared_error(test1_y,base_pred))                           
print(base_RMSE2)

#setting intercept as true
lgr2 = LinearRegression(fit_intercept=True)

# Fitting the train data for linear model
y = lgr2.fit(train1_x,train1_y)

#Predicting the model on test data
predicted_lgr2 = lgr2.predict(test1_x)

# RMSE of the linear model
lr_mse1= mean_squared_error(test1_y,predicted_lgr2)
lr_rmse1 = np.sqrt(lr_mse1)
print(lr_rmse1)

#R squared value
r2_test1 = y.score(test1_x,test1_y)
r2_train1 = y.score(train1_x,train1_y)
print(r2_test1,r2_train1)

#Residuals
residuals2 = test1_y - predicted_lgr2 
sns.regplot(x=predicted_lgr2, y=residuals2, fit_reg=False)


#Results
print('Models built from data where missing values were omitted')
print('R squared value for train= %s' %r2_train)
print('R squared value for test=%s' %r2_test)
print('Base RMSE value of the data where missing values are omitted= %s' %base_RMSE)
print('RMSE value for test =%s' %lr_rmse)
print('\n\n')
print('Models built from data where missing values where imputed')
print('R squared value for train =%s' %r2_train1)
print('R squared value for test =%s' %r2_test1)
print('Base RMSE value of the data where missing values are imputed= %s' %base_RMSE2)
print('RMSE value for test =%s' %lr_rmse1)
