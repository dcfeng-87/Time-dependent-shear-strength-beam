
# core code for predicting the shear strength using AdaBoost
# v 0.1
# 2019-7-14

# define some necessary packages
import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from sklearn.model_selection import cross_val_predict
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib
matplotlib.rcParams['mathtext.fontset']='stix'
# load the full data set
dataset = np.loadtxt('corr_beam.csv', delimiter=",")

# define Input variables X and output variable y
# 13 variables as followed: f_c;b;h;rho_l;rho_v;f_y;f_yv;s;lambda;eta_l;eta_w;h0;A_v	

fc = dataset[:, 0]
b = dataset[:, 1]
h = dataset[:, 2]
rho_l = dataset[:, 3]
rho_v = dataset[:, 4]
fy = dataset[:, 5]
fyv = dataset[:, 6]
s = dataset[:, 7]
lambda_s = dataset[:, 8]
eta_l = dataset[:, 9]
eta_w = dataset[:, 10]
h0 = dataset[:, 11]

X = np.zeros(shape=(158,6))
X[:, 0] = lambda_s
X[:, 1] = h0/b
X[:, 2] = rho_l * fy / fc
X[:, 3] = rho_v * fyv / fc
X[:, 4] = eta_l
X[:, 5] = eta_w

y = dataset [0:158, 12]   # shear strength

# normalize the data sets
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the training-testing set into 10 folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)

# DT
# regr_1 = DecisionTreeRegressor(max_depth=80,max_leaf_nodes=100,min_samples_leaf=1,min_samples_split=2,random_state=2)

# RF
# regr_2 = RandomForestRegressor(n_estimators=200, max_depth=80,max_leaf_nodes=300,min_samples_leaf=1,min_samples_split=2,random_state=0)

# GBRT
regr_1 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,max_depth=2,max_leaf_nodes=5,min_samples_leaf=1,min_samples_split=2, random_state=0, loss='ls')

# AdaBoost
# regr_4 = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=80,max_leaf_nodes=300,min_samples_leaf=1,min_samples_split=2,random_state=0),	n_estimators=200, learning_rate=0.8)


scores = cross_val_score (regr_1, X_train, y_train, cv=10, scoring='r2', n_jobs = -1)
print('10-fold R^2:', scores.mean())

scores = cross_val_score (regr_1, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs = -1)
print('10-fold RMSE:', np.mean(np.sqrt( -scores)))

scores = cross_val_score (regr_1, X_train, y_train, cv=10, scoring='neg_mean_absolute_error', n_jobs = -1)
print('10-fold MAE:', -scores.mean())


# training the learner
regr_1.fit(X_train, y_train)

# predict the results
Z1 = regr_1.predict(X_train)
Z2 = regr_1.predict(X_test)


print("GBRT Training RMSE:", np.sqrt(mean_squared_error(y_train, Z1)), "MAE:", mean_absolute_error(y_train, Z1), "R2:", r2_score(y_train, Z1))
print("GBRT Testing RMSE:", np.sqrt(mean_squared_error(y_test, Z2)), "MAE:", mean_absolute_error(y_test, Z2), "R2:", r2_score(y_test, Z2))
##X_train = scaler.inverse_transform(X_train)
##print(X_train[:, 5])
##print(y_train)
##print(Z1)
##X_test = scaler.inverse_transform(X_test)
##print(X_test[:, 5])
##print(y_test)
##print(Z2)



##xx_test = [[1,1.3917,28.6691,3.1587,0,0],
##            [1,1.3917,28.6691,3.1587,0,0],
##            [1,1.3917,28.6691,3.1587,0,0],
##            [1,1.3917,28.6691,3.1587,0,0],
##            [1,1.3917,28.6691,3.1587,0,8.3746],
##            [1,1.3917,28.6691,3.1587,0,21.5831],
##             [1,1.3917,28.6691,3.1587,0,31.1602],
##            [1,1.3917,28.6691,3.1587,0,39.0631],
##            [1,1.3917,28.6691,3.1587,0,45.8741],
##             [1,1.3917,28.6691,3.1587,0,51.8749],
##            [1,1.3917,28.6691,3.1587,0,57.2313],
##            [1,1.3917,28.6691,3.1587,0,62.0518]]

##xx_test = [[2,1.3250,20.9625,1.6103,0,0],
##            [2,1.3250,20.9625,1.6103,0,0],
##            [2,1.3250,20.9625,1.6103,0,10.2427],
##             [2,1.3250,20.9625,1.6103,0,30.1401],
##            [2,1.3250,20.9625,1.6103,0,43.6124],
##            [2,1.3250,20.9625,1.6103,0,54.2122],
##             [2,1.3250,20.9625,1.6103,0,62.9282],
##            [2,1.3250,20.9625,1.6103,0,70.2340],
##            [2,1.3250,20.9625,1.6103,0,76.4075],
##            [2,1.3250,20.9625,1.6103,0,81.6314],
##            [2,1.3250,20.9625,1.6103,0,86.0349],
##            [2,1.3250,20.9625,1.6103,0,89.7143]]

xx_test = [[1.0000,0.8333,28.6691,3.1587,0,32.0500],
           [1.0000,1.2500,28.6691,3.1587,0,32.0500],
           [1.0000,1.6667,28.6691,3.1587,0,32.0500],
           [1.0000,2.0833,28.6691,3.1587,0,32.0500],
           [1.0000,2.5000,28.6691,3.1587,0,32.0500],
           [1.0000,2.9167,28.6691,3.1587,0,32.0500],
           [1.0000,3.3333,28.6691,3.1587,0,32.0500],
           [1.0000,3.7500,28.6691,3.1587,0,32.0500],
           [1.0000,4.1667,28.6691,3.1587,0,32.0500],
           [1.0000,4.5833,28.6691,3.1587,0,32.0500]]


xx_test = scaler.transform(xx_test)
zz = regr_1.predict(xx_test)
print(zz)

# predicted = cross_val_predict (regr_1, X_train, y_train, cv=10)
predicted = Z1
# AdaBoost results
X_inp1 = np.concatenate((X_train, X_test))
print(X_inp1.shape)
X_inp = scaler.inverse_transform(X_inp1)

y_ture = np.concatenate((y_train, y_test))
y_pre = np.concatenate((predicted, Z2))
print("GBRT RMSE:", np.sqrt(mean_squared_error(y_ture, y_pre)), "R2:", r2_score(y_ture, y_pre), "MAE:", mean_absolute_error(y_ture, y_pre))

ratio = y_pre/y_ture
print(np.mean(ratio), np.std(ratio, ddof=1))

writer = pd.ExcelWriter('ratio.xlsx',engin='openpyxl')
#book = load_workbook(writer.path)
#writer.book = book
df1 = pd.DataFrame({'eta_w':X_inp[:,5]/100})
df2 = pd.DataFrame({'y_ture':y_ture})
df3 = pd.DataFrame({'y_pre':y_pre})
df1.to_excel(writer,startcol=0,index=False)
df2.to_excel(writer,startcol=1,index=False)
df3.to_excel(writer,startcol=2,index=False)
writer.save()
writer.close()

# ##xx_test = np.array(xx_test).reshape(1, -1)
# xx_test = scaler.transform(xx_test)
# zz = regr_1.predict(xx_test)
# print(zz)

##print("Training R of this fold :", np.sqrt(r2_score(y_train, Z1)))
##print("Testing R of this fold:", np.sqrt(r2_score(y_test, Z2)))
##
##print("Training RMSE of this fold :", np.sqrt(mean_squared_error(y_train, Z1)))
##print("Testing RMSE of this fold:", np.sqrt(mean_squared_error(y_test, Z2)))
##
##print("Training R of this fold :", np.sqrt(r2_score(y_train, Z1)))
##print("Testing R of this fold:", np.sqrt(r2_score(y_test, Z2)))

# plot the results by AdaBoost
xx = np.linspace(0, 600, 100)
yy = xx

plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
plt.scatter(y_train, Z1, marker='s', s=80, c='r', edgecolor='grey', linewidth=0.5)
# plt.scatter(y_test, Z2, marker='s')

plt.grid()
# plt.legend(['y=x','Training set','Testing set'], loc = 'upper left', fontproperties(family='serif') )

plt.tick_params (axis='both',which='major',labelsize=14)

font1 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 18,
}

plt.axis('tight')
plt.xlabel('Tested shear strength (kN)', font1)
plt.ylabel('Predicted shear strength (kN)', font1)
plt.xlim([0, 600])
plt.ylim([0, 600])

plt.tight_layout()
plt.savefig('Fig4a.eps', dpi=600, bbox_inches = 'tight', format='eps')




plt.figure()
plt.plot(xx, yy, c='k', linewidth=2)
# plt.scatter(y_train, Z1, marker='s', s=50, c='r', edgecolor='grey', linewidth=1)
plt.scatter(y_test, Z2, marker='o', s=80, edgecolor='b', linewidth=0.5)

plt.grid()
# plt.legend(['y=x','Training set','Testing set'], loc = 'upper left', fontproperties(family='serif') )

plt.tick_params (axis='both',which='major',labelsize=14)

font1 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 18,
}

plt.axis('tight')
plt.xlabel('Tested shear strength (kN)', font1)
plt.ylabel('Predicted shear strength (kN)', font1)
plt.xlim([0, 600])
plt.ylim([0, 600])

plt.tight_layout()
plt.savefig('Fig4b.eps', dpi=600, bbox_inches = 'tight', format='eps')




##plt.show()
