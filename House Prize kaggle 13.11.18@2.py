# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 19:24:48 2018

@author: bharat
"""





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#os.chdir('G:\\DataScience\\kaggle\\house project')
os.chdir('D:\\naveen\\Kaggle house price project')







Test_data=pd.read_csv("test.csv")
Train_data=pd.read_csv("train.csv")
Test_data.shape
'''(1460, 81)'''
Test_data.columns


Test_data.isna().sum().sort_values(ascending=False)



Test_numcol = Test_data.select_dtypes(include=[np.number])
Test_catcol=Test_data.select_dtypes(exclude=[np.number])
Test_numcol=Test_numcol.fillna(Test_numcol.mean())
Test_catcol=Test_catcol.astype('str').replace('nan',Test_catcol.mode())
Test_numcol.isna().sum().sort_values(ascending=False)
Test_catcol.isna().sum().sort_values(ascending=False)
Test_numcol.columns
Test_numcol=Test_numcol.astype('int')
Test_catcol=Test_catcol.astype('str')



Test_catcol_dumm_varible=pd.get_dummies(data=Test_catcol,drop_first=True)
Test_catcol_dumm_varible=Test_catcol_dumm_varible.astype('int')
Test_dumm_col=list(Test_catcol_dumm_varible.columns)
Test_catcol_dumm_varible.info()
pd.concat([Test_numcol, Test_catcol_dumm_varible], axis=1, sort=False).isna().sum().sort_values(ascending=False)
Test_data_final=pd.concat([Test_numcol, Test_catcol_dumm_varible], axis=1, sort=False)







numcol = Train_data.select_dtypes(include=[np.number])
catcol=Train_data.select_dtypes(exclude=[np.number])
numcol=numcol.fillna(numcol.mean())
catcol=catcol.astype('str').replace('nan',catcol.mode())
numcol.isna().sum().sort_values(ascending=False)
catcol.isna().sum().sort_values(ascending=False)
numcol.columns
numcol=numcol.astype('int')
numcol.info()
catcol=catcol.astype('str')
catcol.info()


catcol_dumm_varible=pd.get_dummies(data=catcol,drop_first=True)
catcol_dumm_varible=catcol_dumm_varible.astype('int')
Train_dummy_col=list(catcol_dumm_varible)

catcol_dumm_varible=catcol_dumm_varible[Test_dumm_col]

Train_data_final=pd.concat([numcol, catcol_dumm_varible], axis=1, sort=False)
Train_data_final.SalePrice=np.log(Train_data_final.SalePrice)
Train_data_final.SalePrice

numcol.columns



sns.pairplot(data=Train_data,x_vars=['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',
       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
       'MiscVal', 'MoSold', 'YrSold'],y_vars=['SalePrice'])


sns.pairplot(numcol, palette="Set2", diag_kind="kde", height=2.5)

g = sns.PairGrid(numcol)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);


plt.figure(figsize = (7,5))

sns.pairplot(numcol)


sns.boxplot(x="SalePrice", data=numcol)
sns.swarmplot(x="SalePrice", data=numcol, color=".25")


    
    
for column in numcol:
    sns.boxplot(data=Train_data,x=column)
    plt.show()

for column in numcol:
    sns.violinplot(data=Train_data,x=column)
    plt.show()


for column in numcol:
    sns.distplot(numcol[column])
    plt.show()

for column in numcol:
    sns.jointplot(data=Train_data, x=column,y='SalePrice')
    plt.show()



for column in numcol:
    sns.jointplot(data=Train_data, x=column,y='SalePrice',kind='hex')
    plt.show()


for column in numcol:
    sns.jointplot(data=Train_data, x=column,y='SalePrice',kind='kde')
    plt.show()




for column in numcol:
    sns.jointplot(data=Train_data, x=column,y='SalePrice',kind='reg')
    plt.show()


for column in catcol:
    sns.countplot(data=Train_data, x=column)
    plt.show()

for column in catcol:
    sns.boxplot(data=Train_data,x=column,y='SalePrice')
    plt.show()

for column in catcol:
    sns.boxplot(data=Train_data,x=column,y='SalePrice',hue=column)
    plt.show()

for column in catcol:
    sns.violinplot(data=Train_data,x=column,y='SalePrice')
    plt.show()





for column in catcol:
    sns.violinplot(data=Train_data,x=column,y='SalePrice')
    plt.show()



for column in catcol:
    sns.relplot(data=Train_data,x=column,y='SalePrice',hue=column)
    plt.show()



for column in numcol:
    sns.relplot(data=Train_data,x=column,y='SalePrice',hue=column)
    plt.show()


for column in numcol:
    sns.relplot(data=Train_data,x=column,y='SalePrice',size='SalePrice')
    plt.show()


for column in numcol:
    sns.relplot(data=Train_data,x=column,y='SalePrice',kind='line')
    plt.show()
    


for column in catcol:
    sns.catplot(data=Train_data,x=column,y='SalePrice')
    plt.show()



for column in catcol:
    sns.catplot(data=Train_data,x=column,y='SalePrice',jitter=False)
    plt.show()


for column in catcol:
    sns.catplot(data=Train_data,x=column,y='SalePrice',kind='swarm')
    plt.show()


for column in catcol:
    sns.catplot(data=Train_data,x=column,y='SalePrice',kind='boxen')
    plt.show()
    

for column in catcol:
    g=sns.catplot(data=Train_data,x=column,y='SalePrice',kind='violin')
    sns.swarmplot(data=Train_data,x=column,y='SalePrice',color="k", size=3,ax=g.ax)
    plt.show()
    
for column in numcol:
    sns.regplot(data=Train_data,x=column,y='SalePrice')
    plt.show()
    

    
for column in numcol:
    sns.lmplot(data=Train_data,x=column,y='SalePrice')
    plt.show()




Train_data_final.drop(['Id'], axis=1, inplace=True)
Train_data_final.Id




y=Train_data_final.SalePrice.values
Train_data_final.drop(['SalePrice'], axis=1, inplace=True)
x=Train_data_final.values






from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(x)
x=scaler.transform(x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size =0.3,random_state=0)


from xgboost import XGBRegressor

xgb = XGBRegressor()

xgb.fit(x_train,y_train)
xgb.score(x,y)
xgb.score(x_train,y_train)
xgb.score(x_test,y_test)

#xgb.predict(Test_x)







#Test_data_final.drop(['MiscVal','3SsnPorch','Id','BsmtFinSF2'], axis=1, inplace=True)
Test_data_final.drop(['Id'], axis=1, inplace=True)
Train_data_final.columns.sort_values(ascending=True)
Test_data_final.columns.sort_values(ascending=True)

(Train_data_final.columns.sort_values(ascending=True)==Test_data_final.columns.sort_values(ascending=True)).sum()
Train_data_final.columns==Test_data_final.columns


from collections import Counter
Counter(Train_data_final.columns==Test_data_final.columns)
(Train_data_final.columns==Test_data_final.columns).sum()


Test_x=Test_data_final.values
Test_x=scaler.transform(Test_x)
xgb.predict(Test_x)











from sklearn.grid_search import GridSearchCV


param_test1 ={
        'n_estimators':[50,100],
        'min_child_weight': [9,10,11],
        'gamma': [0,0.1,1],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [1,5,10,15]
        }


model1 = GridSearchCV(xgb, param_grid=param_test1, n_jobs=-1)

model1.fit(x,y)
print("Best Hyper Parameters:",model1.best_params_)

model1.score(x,y)
model1.score(x_train,y_train)
model1.score(x_test,y_test)




predect=model1.predict(x)
diff=y-predect



plt.plot(y)
plt.plot(predect)
plt.plot(diff)


actual_y_test = np.exp(y)
actual_predicted = np.exp(predect)
diff = abs(actual_y_test - actual_predicted)

compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})
compare_actual = compare_actual.astype(int)
compare_actual.head(5)



plt.plot(actual_y_test)
plt.plot(actual_predicted)
plt.plot(diff)




from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pca_x=pca.fit_transform(x)



plt.scatter(x=pca_x,y=actual_y_test)
plt.scatter(x=pca_x,y=actual_predicted)



submission1=np.exp(xgb.predict(Test_x))


Test_data_Id=Test_data.Id

submission=pd.DataFrame(data = {'Id':Test_data_Id, 'SalePrice':submission1})

import os
os.chdir('C:\\Users\\hp\\Downloads')
submission.to_csv('SampleSubmission.csv',index = False)


data_submission=pd.read_csv("sample_submission.csv")
data_submission.columns
(data_submission.Id==submission.Id).value_counts()































