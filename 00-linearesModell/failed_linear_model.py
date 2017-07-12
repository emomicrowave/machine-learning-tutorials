# coding: utf-8

import script
import script
script.train_data
script.train_data.columns
get_ipython().magic('ls ')
script.train_data.columns
vec_x = script.train[['Assets', 'Sales', 'Profits', 'Cash_Flow', 'Employees']]
vec_x = script.train_data[['Assets', 'Sales', 'Profits', 'Cash_Flow', 'Employees']]
vec_x
vec_y = script.train_data['Market_Value']
vec_y.head(5)
np.linalg.lstsq(vec_x, vec_y)
import numpy as np
np.linalg.lstsq(vec_x, vec_y)
np.linalg.lstsq(vec_x, vec_y.T)
np.linalg.lstsq(vec_x.T, vec_y)
vec_x.shape
vex_y.shape
vec_y.shape
type(vec_y)
a = np.linalg.lstsq(vec_x, vec_y)
pd.Series(a)
import pandas as pd
pd.Series(a)
pd.Series(a).T
func = a[0]
func
target_y = []
for index, row in script.test_data.iterrows():
    target_y.appendrow(row * func)
    
for index, row in script.test_data.iterrows():
    target_y.append(row * func)
    
    
for index, row in script.test_data.iterrows():
    target_y.append(row * func)
    
    
    
func = pd.Series(func)
func
for index, row in script.test_data.iterrows():
    target_y.append(row * func)
    
testing = script.test_data[['Assets', 'Sales', 'Profits', 'Cash_Flow', 'Employees']]
for index, row in testing.iterrows():
    target_y.append(row * func)
    
    
testing.info
testing.describe
testing.info()
func.info()
func.describe()
for index, row in testing.iterrows():
    print(index, row)
    target_y.append(row * func)
    
for index, row in testing.iterrows():
    print(row)
    target_y.append(row * func)
    
for index, row in testing.iterrows():
    print(row)
    target_y.append(row.mul(func))
    
    
for index, row in testing.iterrows():
    target_y.append(row.mul(func))
    
    
for index, row in testing.iterrows():
    target_y.append(row.T.mul(func)) 
    
for index, row in testing.iterrows():
    print(row.T)
    target_y.append(row.T.mul(func)) 
    
    
for index, row in testing.iterrows():
    target_y.append(row.as_matrix().T * func.as_matrix())) 

    
    
for index, row in testing.iterrows():
    target_y.append(row.as_matrix().T * func.as_matrix())
    

    
    
target_y
target_y = []
for index, row in testing.iterrows():
    target_y.append(row.as_matrix() * func.as_matrix())
    
target_y
target_y = []
for index, row in testing.iterrows():
    target_y.append(np.multiply(row.as_matrix(), func.as_matrix()))
    
    
target_y
func
testing[:][0]
testing[:]['0']
testing[:]
testing[0]
testing
testing[:][69]
testing
testing.index = [x in range(len(testing))]
testing.index = [x for x in range(len(testing))]
testing[:][0]
testing[:,0]
testing[0,:]
testing.iloc(0)
testing.iloc[0]
testig
testing
testing.iloc[0].as_matrix()
testing.iloc[0].as_matrix() * func
testing.iloc[0].as_matrix() * func.T
testing.iloc[0].as_matrix() * func.as_matrix.T
testing.iloc[0].as_matrix() * func.as_matrix().T
testing.as_matrix().shape
testing.iloc[0].as_matrix().shape
testing.iloc[0].as_matrix().reshape([5,1])
testing.iloc[0].as_matrix().reshape([5,1]) * func
testing.iloc[0].as_matrix().reshape([5,1]) * func.as_matrix()
np.mult(testing.iloc[0].as_matrix().reshape([5,1]) , func.as_matrix())
np.multiply(testing.iloc[0].as_matrix().reshape([5,1]) , func.as_matrix())
np.multiply(testing.iloc[0].as_matrix().reshape([5,1]) , func.as_matrix().reshape([1,5]))
target_y = []
for index, row in testing.iterrows():
    target_y.append(np.dot(row.as_matrix(), func.as_matrix()))
    
target_y
target_y.index = testing.index
target_y = pd.Series(target_y)
target_y.index = testing.index
script.test_data.index = testing.index
script.test_data['MV_Predicitons'] = target_y
script.test_data
