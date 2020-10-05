""" IMPORT THE RELEVANT LIBRARIES"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

""" LOAD THE DATA """

raw_data = pd.read_csv('C:/Users/sivam/OneDrive/Projects/Multi Linear Regression/Used cars/used cars.csv')
raw_data.head()

""" EXPLORING THE DESCRIPTIVE STATISTICS OF THE VARIABLES """

raw_data.describe(include ='all')
data = raw_data.drop(['Model'], axis =1) 
# its has a huge number of model, hard to create a regression with the amount of data
    
data.isnull()  # will give us data about missing values. true are the missing values
    
data.isnull().sum() 
# will give us the total missing values in each column if the missing value is less then 5% we can drop those columns.
    
data_no_mv = data.dropna(axis =0)
data_no_mv.describe(include='all')

""" EXPLORING THE PDF(PROBABILITY DISTRIBUTION FUNCTION). FOR OPTIMAL RESULTS WE WILL LOOKING FOR A NORMAL DISTRIBITION """

sns.distplot(data_no_mv['Price']) 
# price has a mean of 19552 and min of 600 and max of 300000. This means its has outliers. 
# one way to deal with this is to remove top 1% of the observation. the simplest way to do this is use 'quantile'.
sns.boxplot(data_no_mv['Price'])  
q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price'] <q]
data_1.describe(include ='all')
    
# repeat the process for the other variables aswell.
sns.distplot(data_1['Mileage'])
sns.boxplot(data_no_mv['Mileage']) 
q=data_1['Mileage'].quantile(0.99)
data_2=data_1[data_1['Mileage']<q]
    
sns.distplot(data_2['EngineV']) 
# engine volume values vary from 0.6 to 99.9 in the data where as in the real world the value lies between 0.6 to 6.5
sns.boxplot(data_no_mv['EngineV']) 
data_3 = data_2[data_2['EngineV']<6.5]
    
sns.distplot(data_3['Year']) 
# the plot shows the number of cars in early year are less, so we use greater than 0.01 quantile.
sns.boxplot(data_no_mv['Year']) 
q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year']>q]

data_cleaned = data_4.reset_index(drop=True)
data_cleaned.describe(include='all')

""" CHECKING THE OLS ASSUMPTIONS """
     
# check for linearity
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize = (15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price vs Year')
ax2.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax2.set_title('Price vs Mileage')
ax3.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax3.set_title('Price vs EngineV')
plt.show()
    
# the data is not linear, before performing regression we need to transform the data, we are gonna use log transformation.
    
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned
    
# create a scatter plot using log_price
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize = (15, 3))
ax1.scatter(data_cleaned['Year'], data_cleaned['log_price'])
ax1.set_title('log_price vs Year')
ax2.scatter(data_cleaned['Mileage'], data_cleaned['log_price'])
ax2.set_title('log_price vs Mileage')
ax3.scatter(data_cleaned['EngineV'], data_cleaned['log_price'])
ax3.set_title('log_price vs EngineV')
plt.show()
    
# Now that the data is linear we drop the price column.
    
data_cleaned = data_cleaned.drop(['Price'], axis = 1)

""" MULTICOLLINEARITY """

# the best way to check multicollinearity is through VIF (variance inflation factor) using statsmodels
    
variables = data_cleaned[['Year', 'Mileage', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
vif
    
# VIF = 1 : No multicollinearity
# 1 < VIF < 5 : Perfectly okay
# 5 or 6 or 10 < VIF : unacceptable. The value variables in this case we consider 10. so we eliminate 'Year'.
    
data_no_collinearity = data_cleaned.drop(['Year'], axis = 1)

""" CREATE DUMMY VARIABLES """

# categorical values cannot be used to create a regression. pd.get_dummies is used automatically convert categorical data to numerical dummies
    
data_dummies = pd.get_dummies(data_no_collinearity, drop_first= True)    
data_dummies.head()
data_processed = data_dummies
data_processed.head()

""" STANDARDISE AND LINEAR REGRESSION MODEL """
    
targets = data_processed['log_price']
inputs = data_processed.drop(['log_price'], axis = 1)
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size = 0.2, random_state = 29)
reg = LinearRegression()
reg.fit(inputs_scaled, targets)

""" CHECK THE REGRESSION """

# two to check the regression

# first method - plot the result of predicted values against observed values

y_hat = reg.predict(x_train)
plt.scatter(y_train,y_hat)
plt.xlabel('observed values - (y_train)')
plt.ylabel('predicted values - (y_hat)')
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()

# the model is 

# second method - residual plot

sns.distplot(y_train - y_hat)
plt.title('Residual distribution')

reg.score(x_train, y_train)

""" finding the weights & bias """

reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values, columns = ['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


""" Testing with residual & residual% """

y_hat_test = reg.predict(x_test)
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
y_test = y_test.reset_index(drop=True)
df_pf['observed'] = np.exp(y_test)
df_pf['Residuals'] = df_pf['observed'] - df_pf['Predictions']
df_pf['Residual%'] =np.absolute(df_pf['Residuals']/df_pf['observed']*100)
pd.set_option('display.float_format', lambda x: '%.2f' %x)
df_pf
df_pf.sort_values(by=['Residual%'])

