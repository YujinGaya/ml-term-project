#invite people for the Kaggle party
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
np.set_printoptions(threshold=np.nan)
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from scipy import stats
import warnings
from pandas import DataFrame
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
#%matplotlib inline

#bring in the six packs
col_list = ['contract date', 'latitude', 'longitude', 'altitude', '1st class region id', '2nd class region id',
'road id', 'apartment id', 'floor', 'angle', 'area', 'num of car to park', 'park area', 'whether external vehicle can enter parking lot', 
'average management fee', 'number of households', 'age', 'builder id', 'complete date' , 'built year', 'number of schools near the apartment' ,
'number of bus stations near apartment', 'number of subway stations near apartment', 'SalePrice']
df_train = pd.read_csv('./data_train.csv', names = col_list)
df_test = pd.read_csv('./data_test.csv', names = col_list)
df_test.pop('SalePrice')

df_train['complete date'] = pd.to_datetime(df_train['complete date'])
df_train['contract date'] = pd.to_datetime(df_train['contract date'])
df_test['complete date'] = pd.to_datetime(df_test['complete date'])
df_test['contract date'] = pd.to_datetime(df_test['contract date'])


X = df_train[['contract date', 'latitude', 'longitude', 'altitude', '1st class region id', '2nd class region id',
'road id', 'apartment id', 'floor', 'angle', 'area', 'num of car to park', 'park area', 'whether external vehicle can enter parking lot', 
'average management fee', 'number of households', 'age', 'builder id', 'complete date' , 'built year', 'number of schools near the apartment' ,
'number of bus stations near apartment', 'number of subway stations near apartment']]

X['complete date'] = pd.to_datetime(df_train['complete date']).dt.year
X['contract date'] = pd.to_datetime(df_train['contract date']).dt.month
df_test['complete date'] = pd.to_datetime(df_test['complete date']).dt.year
df_test['contract date'] = pd.to_datetime(df_test['contract date']).dt.month
df_train['complete date'] = pd.to_datetime(df_train['complete date']).dt.year
df_train['contract date'] = pd.to_datetime(df_train['contract date']).dt.month

y= df_train['SalePrice']




#fill missing data
rf = RandomForestRegressor(max_depth = 20, n_estimators = 1000)
simpimp = SimpleImputer(missing_values= np.nan, strategy='most_frequent')
simpimp2 = SimpleImputer(missing_values = np.nan, strategy = 'mean')


X = X.values

# builder id
X[:,17] = simpimp.fit_transform(np.reshape(X[:,17], (-1,1)))[:,0]
# bus stations 중요도가 낮다
X[:,21] = simpimp.fit_transform(np.reshape(X[:,21], (-1,1)))[:,0]
# households 중요도가 낮다
X[:,15] = simpimp.fit_transform(np.reshape(X[:,15], (-1,1)))[:,0]
# altitude  중요도가 낮으니 빼까
X[:,3] = simpimp2.fit_transform(np.reshape(X[:,3], (-1,1)))[:,0]
# number of car 중요!
X[:,11] = simpimp2.fit_transform(np.reshape(X[:,11], (-1,1)))[:,0]
# park area 중요!
X[:,12] = simpimp2.fit_transform(np.reshape(X[:,12], (-1,1)))[:,0]
# build year 꽤 중요, nan이 적다
X[:,19] = simpimp.fit_transform(np.reshape(X[:,19], (-1,1)))[:,0]
# completion 꽤 나름 중요
X[:,18] = simpimp.fit_transform(np.reshape(X[:,18], (-1,1)))[:,0]
# latitude
X[:,1] = simpimp.fit_transform(np.reshape(X[:,1], (-1,1)))[:,0]
# longitude
X[:,2] = simpimp.fit_transform(np.reshape(X[:,2], (-1,1)))[:,0]

#print(X)

#important!!
y = y.values

mask = np.isnan(X).any(axis=1)
#print(X[~mask])
X = X[~mask]
y = y[~mask]

X = np.delete(X[:, :], 6158, 0)
y = np.delete(y[:], 6158, 0)

X = X[:, :]
y = y[:]

#for i in range(np.size(X[0])):
#    print(np.size(X[0]))
#    print(X[:,i])
#for i in range(np.size(y)):
#    print(y[i])


rf.fit(X, y)
print(rf.feature_importances_)

# test
df_train = pd.read_csv('./data_train.csv', names = col_list)

df_train['complete date'] = pd.to_datetime(df_train['complete date'])
df_train['contract date'] = pd.to_datetime(df_train['contract date'])

X = df_train[['contract date', 'latitude', 'longitude', 'altitude', '1st class region id', '2nd class region id',
'road id', 'apartment id', 'floor', 'angle', 'area', 'num of car to park', 'park area', 'whether external vehicle can enter parking lot', 
'average management fee', 'number of households', 'age', 'builder id', 'complete date' , 'built year', 'number of schools near the apartment' ,
'number of bus stations near apartment', 'number of subway stations near apartment']]

X['complete date'] = pd.to_datetime(df_train['complete date']).dt.year
X['contract date'] = pd.to_datetime(df_train['contract date']).dt.month
df_train['complete date'] = pd.to_datetime(df_train['complete date']).dt.year
df_train['contract date'] = pd.to_datetime(df_train['contract date']).dt.month
y= df_train['SalePrice']

# fill out missing values
simpimp = SimpleImputer(missing_values= np.nan, strategy='most_frequent')
simpimp2 = SimpleImputer(missing_values = np.nan, strategy = 'mean')

X = X.values
y = y.values


# builder id
X[:,17] = simpimp.fit_transform(np.reshape(X[:,17], (-1,1)))[:,0]
# bus stations 중요도가 낮다
X[:,21] = simpimp.fit_transform(np.reshape(X[:,21], (-1,1)))[:,0]
# households 중요도가 낮다
X[:,15] = simpimp.fit_transform(np.reshape(X[:,15], (-1,1)))[:,0]
# altitude  중요도가 낮으니 빼까
X[:,3] = simpimp2.fit_transform(np.reshape(X[:,3], (-1,1)))[:,0]
# number of car 중요!
X[:,11] = simpimp2.fit_transform(np.reshape(X[:,11], (-1,1)))[:,0]
# park area 중요!
X[:,12] = simpimp2.fit_transform(np.reshape(X[:,12], (-1,1)))[:,0]
# build year 꽤 중요, nan이 적다
X[:,19] = simpimp.fit_transform(np.reshape(X[:,19], (-1,1)))[:,0]
# completion 꽤 나름 중요
X[:,18] = simpimp.fit_transform(np.reshape(X[:,18], (-1,1)))[:,0]
# latitude
X[:,1] = simpimp.fit_transform(np.reshape(X[:,1], (-1,1)))[:,0]
# longitude
X[:,2] = simpimp.fit_transform(np.reshape(X[:,2], (-1,1)))[:,0]

#df_test = simpimp.transform(df_test)
#y_pred = rf.predict(df_test)
y_pred = rf.predict(X)
#print(y_pred)
#print(rf.score(df_test, y))

# calculate performance
sum = 0.0
for i in range(np.size(y)):
    sum += np.absolute(1.0 - y_pred[i]/y[i])

perf = 1 - sum/np.shape(X)[0]
print(perf)