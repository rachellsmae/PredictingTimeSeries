
# coding: utf-8

# In[238]:


# imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math

from sklearn.preprocessing import scale 
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold, train_test_split
from sklearn.linear_model import LinearRegression, LassoCV, Lasso, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn import feature_selection, tree
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from scipy import stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ## Part 1: Load Data

# In[249]:


# load data
X = pd.read_csv('X.csv', delimiter=',')
y = pd.read_csv('y.csv', delimiter=',')


# In[68]:


X.describe()


# In[69]:


# print out first 10 values of X
X.head(10)


# In[70]:


# determine the correlation between predictor variables before fitting any models
pd.DataFrame(X.corr()).head(10)


# In[71]:


# plot time series of y
fig= plt.figure(figsize=(20,5))
plt.plot(y)
plt.title('Time Series of y')
plt.show()


# The time series plot shows that there is no trend structure in the log returns of the stock.
# 
# Based on the correlation matrix of the predictor variables, it is clear that there is some correlation between the variables, especially those from the same time period. Furthermore, the high number of predictor variables (189) may reduce the accuracy of the model due to the curse of dimensionality. Therefore, I've decided to find the principal components to be used in all of my models.

# ## Part 2: Explore Various Regression Models and Find the Best to Predict  $x_{t+1}$

# #### Helper functions

# In[72]:


# reset index
def reset(df):
    return df.reset_index(drop=True)

# plot the actual vs predicted log returns (training)
def plot_actual_pred(y_test, y_pred):
    fig= plt.figure(figsize=(20,5))
    plt.plot(y_test, color='red', label='Actual')
    plt.plot(y_pred, color='blue', label='Predicted')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Log Returns')
    plt.title('Actual vs Predicted Log Returns (Test)')
    plt.show()


# In[250]:


# divide data into 0.7 training and 0.3 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train = reset(X_train)
X_test = reset(X_test)
y_train = reset(y_train)
y_test = reset(y_test)


# #### Forward Selection Regression

# In[111]:


forward = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=4)

results_for = forward.fit(X_train,y_train)
scores = results_for.scores_

ind = np.argpartition(scores, -20)[-20:]
X_for_train = X_train.iloc[:,ind]
X_for_train.head(10)


# In[112]:


for_regr = sm.OLS(y_train,X_for_train).fit()
print(for_regr.summary())


# In[113]:


y_for_pred = for_regr.predict(X_test.iloc[:,ind])
mse_for_regr = mean_squared_error(y_test, y_for_pred)
corr_for_regr = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_for_pred)], axis=1, ignore_index=True).corr().iat[0,1]
print(mse_for_regr)
print(corr_for_regr)
plot_actual_pred(y_test, y_for_pred)


# #### Lasso Regression

# In[114]:


# set alphas
alphas = 10**np.linspace(10, -2, 100) * 0.5

# fit lasso model
lassocv = LassoCV(alphas=None, cv=10, max_iter=10000, normalize=False)
lassocv.fit(X_train, np.array(y_train).ravel())

lasso = Lasso(max_iter=10000, normalize=True)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)

y_lasso_pred = lasso.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_lasso_pred)
corr_lasso = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_lasso_pred)], axis=1, ignore_index=True).corr().iat[0,1]

print(mse_lasso)
print(corr_lasso)
plot_actual_pred(y_test, y_lasso_pred)


# #### Regression Only Using Company 4's data

# In[115]:


X_train_4 = X_train.filter(regex=('comp4'))
X_test_4 = X_test.filter(regex=('comp4'))
X_train_4.head(10)


# In[116]:


company_regr = sm.OLS(y_train,X_train_4).fit()
print(company_regr.summary())


# In[118]:


y_company_pred = company_regr.predict(X_test_4)
mse_company_regr = mean_squared_error(y_test, y_company_pred)
corr_company_regr = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_company_pred)], axis=1, 
                              ignore_index=True).corr().iat[0,1]
print(mse_company_regr)
print(corr_company_regr)
plot_actual_pred(y_test, y_company_pred)


# In[119]:


forward2 = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=4)

results_for2 = forward2.fit(X_train_4,y_train)
scores2 = results_for2.scores_

ind2 = np.argpartition(scores2, -10)[-10:]
X_for_train2 = X_train_4.iloc[:,ind2]
X_test_42 = X_test_4.iloc[:,ind2]
X_for_train2.head(10)


# In[120]:


company_regr2 = sm.OLS(y_train,X_for_train2).fit()
print(company_regr2.summary())


# In[121]:


company_regr2 = sm.OLS(y_train,X_for_train2).fit()
y_company_pred2 = company_regr2.predict(X_test_42)
mse_company_regr2 = mean_squared_error(y_test, y_company_pred2)
corr_company_regr2 = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_company_pred2)], axis=1, 
                              ignore_index=True).corr().iat[0,1]
print(mse_company_regr2)
print(corr_company_regr2)
plot_actual_pred(y_test, y_company_pred2)


# #### Principal Component Analysis to reduce dimensionality and the correlation of data points

# In[122]:


# PCA 
pca = PCA()
X_reduced = pca.fit_transform(scale(X))

plt.plot(np.cumsum(pca.explained_variance_ratio_*100))
plt.title('Plot of Total Explained Variance')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative variance explained')
plt.show()


# The first 127 principal components explain 90% of the variance, so I will create a new dataset of only the first 127 principal components.

# In[123]:


X_reduced = pd.DataFrame(X_reduced[:,:127])
X_reduced.head(10)


# #### Regression model using principal components

# In[126]:


# cross-validation to determine the average performance of Model 1
mse = []
corr = []
    
regr = LinearRegression()
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)

mse_model = mean_squared_error(y_test, y_pred)
mse.append(mse_model)
        
corre = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_pred)], axis=1, ignore_index=True).corr().iat[0,1]
corr.append(corre)


# In[127]:


regr = LinearRegression()
regr.fit(X_train,y_train)
lm_y_pred = regr.predict(X_test)

print(np.mean(mse))
print(np.mean(corr))
plot_actual_pred(y_test, lm_y_pred)


# #### Random Forest Regression

# In[129]:


# n_estimators
mse_bag = np.array([])
num_trees_bag = np.array([])

trees = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

clf_bagging = RandomForestRegressor(warm_start=False, 
                                    oob_score=False,
                                    criterion='mse',
                                    random_state = 1)

for i in trees:
    clf_bagging.set_params(n_estimators=i)
    clf_bagging.fit(X_train, np.array(y_train).ravel())
    mse_bag = np.append(mse_bag, mean_squared_error(y_test, clf_bagging.predict(X_test)))
    num_trees_bag = np.append(num_trees_bag,i)


# In[ ]:


# max_depth
mse_bag2 = np.array([])
max_depth_bag = np.array([])
error_rate_bag = np.array([])

max_depth = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

clf_bagging2 = RandomForestRegressor(warm_start=False, 
                                    n_estimators=120,
                                    oob_score=True,
                                    max_features=None,
                                    criterion='mse',
                                    random_state = 1)


for i in max_depth:
    clf_bagging2.set_params(max_depth=i)
    clf_bagging2.fit(X_train, y_train.ravel())
    mse_bag2 = np.append(mse_bag2, mean_squared_error(y_test, clf_bagging2.predict(X_test)))
    error_rate_bag = np.append(error_rate_bag,1-clf_bagging2.oob_score_)
    max_depth_bag = np.append(max_depth_bag,i)


# In[ ]:


rf_final = RandomForestRegressor(warm_start=False, 
                                    n_estimators=120,
                                    oob_score=True,
                                    max_features=None,
                                    criterion='mse',
                                    random_state = 1)

rf_final.fit(X_train, y_train.ravel())

rf_y_pred = rf_final.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_y_pred)
rf_corr = pd.concat([pd.DataFrame(y_test),pd.DataFrame(rf_y_pred)], axis=1, ignore_index=True).corr().iat[0,1]

#mse_tests.append(rf_mse)
#corr_tests.append(rf_corr)


# In[ ]:


print(rf_mse)
print(rf_corr)
plot_actual_pred(y_test, rf_y_pred)


# In[241]:


# gradient boosting regressor 
gbrt=GradientBoostingRegressor(n_estimators=100) 
gbrt.fit(X_train, y_train) 
y_pred=gbrt.predict(X_test) 


# In[242]:


mean_squared_error(y_pred, y_test)


# In[243]:


corr_for_regr = pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_pred)], axis=1, ignore_index=True).corr().iat[0,1]
print(mse_for_regr)
print(corr_for_regr)
plot_actual_pred(y_test, y_pred)


# In[270]:


X_1train = pd.DataFrame(X_train.iloc[:,:8])
X_2train = pd.DataFrame(X_train.iloc[:,5:13])
X_3train = pd.DataFrame(X_train.iloc[:,8:16])
X_4train = pd.DataFrame(X_train.iloc[:,13:21])
    
X_1test = pd.DataFrame(X_test.iloc[:,:8])
X_2test = pd.DataFrame(X_test.iloc[:,5:13])
X_3test = pd.DataFrame(X_test.iloc[:,8:16])
X_4test = pd.DataFrame(X_test.iloc[:,13:21])
    
model1 = LinearRegression().fit(X_1train, y_train)
model2 = LinearRegression().fit(X_2train, y_train)
model3 = LinearRegression().fit(X_3train, y_train)
model4 = LinearRegression().fit(X_4train, y_train)
    
answer = []
summ = []
sum = 0
for i in range(0, len(X_test)):
    sum = model1.predict(X_1test)[i] + model2.predict(X_2test)[i] + model3.predict(X_3test)[i] + model4.predict(X_4test)[i]
    average = sum/4.0
    summ.append(sum)
    answer.append(x)


# In[275]:


plt.plot(summ)
plt.plot(y_test)


# In[276]:


mean_squared_error(summ, y_test)


# In[277]:


corr = pd.concat([pd.DataFrame(y_test),pd.DataFrame(summ)], axis=1, ignore_index=True).corr().iat[0,1]
print(corr)


# ## Part 3: Explore Various Classification Models and Find the Best to Predict  $x_{t+1} >= 0$

# In[ ]:


# set up data for classification
y_class = y.copy()
encode = lambda x: 1 if x >= 0 else 0
y_class['comp4_xt+1'] = y_class['comp4_xt+1'].map(encode)
y_class.head(10)


# #### Logistic Regression with All Variables

# In[146]:


# split data into test and train
X_train, X_test , y_class_train, y_class_test = train_test_split(X, y_class, test_size=0.3, random_state=1)
y_class_train = np.array(y_class_train).ravel()
y_class_test = np.array(y_class_test).ravel()

# fit logistic regression model
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_class_train)
pred = clf.predict_proba(X_test)


# In[ ]:


y_class_pred = clf.predict(X_test)
num_wrong_test = sum(abs(np.array(y_class_test).ravel() - y_class_pred))
print("Test number classified wrong: ", num_wrong_test)
print("Test percentage classified wrong: ", num_wrong_test/len(X_test))


# In[ ]:


fig= plt.figure(figsize=(20,5))
x2 = list(range(1, 752))
plt.scatter(x2, y_class_pred, color='blue', label='Predicted')
plt.scatter(x2, y_class_test, color='red', label='Actual')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Decrease/Increase')
plt.title('Actual vs Predicted (Test)')
plt.show()


# In[108]:


kfold = model_selection.KFold(n_splits=10, random_state=1)
clf_kf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
results = model_selection.cross_val_score(clf_kf, X, np.array(y_class).ravel(), cv=kfold)
print(results.mean())


# #### Logistic Regression with Only Company 4 Lags

# In[ ]:


X_4 = X.filter(regex=('comp4'))
X_train_4 = X_train.filter(regex=('comp4'))
X_test_4 = X_test.filter(regex=('comp4'))

# fit logistic regression model
clf2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train_4, y_class_train)
pred = clf2.predict_proba(X_test_4)
clf2.score(X_test_4, y_class_test)


# In[ ]:


y_class_pred = clf2.predict(X_test_4)
num_wrong_test = sum(abs(np.array(y_class_test).ravel() - y_class_pred))
print("Test number classified wrong: ", num_wrong_test)
print("Test percentage classified wrong: ", num_wrong_test/len(X_test_4))


# In[109]:


kfold = model_selection.KFold(n_splits=10, random_state=1)
results = model_selection.cross_val_score(clf_kf, X_4, np.array(y_class).ravel(), cv=kfold)
print(results.mean())


# #### Support Vector Machine

# In[100]:


model_svc = SVC(C=1E10)
model_svc.fit(X_train_4, np.array(y_class_train).ravel())


# In[ ]:


y_svc_pred = model_svc.predict(X_test_4)

num_wrong_test = sum(abs(np.array(y_class_test).ravel() - y_svc_pred))
print("Test number classified wrong: ", num_wrong_test)
print("Test percentage classified wrong: ", num_wrong_test/len(X_test))


# In[ ]:


fig= plt.figure(figsize=(20,5))
x2 = list(range(1, 752))
plt.scatter(x2, y_svc_pred, color='blue', label='Predicted')
plt.scatter(x2, y_class_test, color='red', label='Actual')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Decrease/Increase')
plt.title('Actual vs Predicted (Test)')
plt.show()


# In[105]:


kfold = model_selection.KFold(n_splits=10, random_state=1)
results = model_selection.cross_val_score(model_svc, X_4, np.array(y_class).ravel(), cv=kfold)
print(results.mean())


# #### Decision Tree

# In[ ]:


clf3 = tree.DecisionTreeClassifier(random_state = 1)
y_class_test = np.array(y_class_test).ravel()
y_class_train = np.array(y_class_train).ravel()

max_depth = [5, 10, 15, 20, 25, 30]
mse_clf3= []
max_depth_clf3 = []

for i in max_depth:
    clf3.set_params(max_depth=i)
    clf3.fit(X_train, y_class_train)
    mse_clf3 = np.append(mse_clf3, mean_squared_error(y_class_test, clf3.predict(X_test)))
    max_depth_clf3 = np.append(max_depth_clf3,i)


# In[ ]:


plt.plot(max_depth_clf3, mse_clf3, c='red');
plt.xlabel('max_depth');
plt.ylabel('Test MSE');
plt.title('Plot of Test MSE vs Max Depth')
plt.show()


# In[ ]:


clf3 = tree.DecisionTreeClassifier(max_depth =20,random_state = 1)
clf3 = clf.fit(X_train, y_class_train)

y_dt_pred = clf3.predict(X_test)

num_wrong_test = sum(abs(np.array(y_class_test).ravel() - y_dt_pred))
print("Test number classified wrong: ", num_wrong_test)
print("Test percentage classified wrong: ", num_wrong_test/len(X_test))


# #### Random Forest

# In[ ]:


# n_estimators
wrong = np.array([])
num_trees_bag2 = np.array([])

trees = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

clf_bagging2 = RandomForestClassifier(warm_start=False, 
                                    oob_score=True,
                                    random_state = 1)

for i in trees:
    clf_bagging2.set_params(n_estimators=i)
    clf_bagging2.fit(X_train, y_class_train)
    wrong = np.append(wrong, sum(abs(y_class_test - clf_bagging2.predict(X_test)))/len(X_test))
    num_trees_bag2 = np.append(num_trees_bag2,i)


# In[ ]:


# max_depth
wrong = np.array([])
max_depth_bag2 = np.array([])
error_rate_bag2 = np.array([])

max_depth = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

clf_bagging2 = RandomForestClassifier(warm_start=False, 
                                    n_estimators=50,
                                    oob_score=True,
                                    max_features=None,
                                    random_state = 1)


for i in max_depth:
    clf_bagging2.set_params(max_depth=i)
    clf_bagging2.fit(X_train, y_class_train)
    wrong = np.append(wrong, sum(abs(y_class_test - clf_bagging2.predict(X_test)))/len(X_test))
    max_depth_bag2 = np.append(max_depth_bag2,i)


# In[ ]:


clf4 = RandomForestClassifier(max_depth =5, n_estimators=50,random_state = 1)
clf4 = clf.fit(X_train, y_class_train)

y_rf_pred = clf4.predict(X_test)

num_wrong_test = sum(abs(y_class_test - y_rf_pred))
print("Test number classified wrong: ", num_wrong_test)
print("Test percentage classified wrong: ", num_wrong_test/len(X_test))


# In[ ]:


# voting ensemble
kfold = KFold(n_splits=10, random_state=1)

# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = tree.DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, np.array(y_class).ravel(), cv=kfold)
print(results.mean())


# In[90]:


# ada boost
num_trees = [10, 20, 30, 40, 50, 60, 70, 80]
res = []
for i in num_trees:
    kfold = model_selection.KFold(n_splits=10, random_state=1)
    model = AdaBoostClassifier(n_estimators=i, random_state=1)
    results = model_selection.cross_val_score(model, X, np.array(y_class).ravel(), cv=kfold)
    mean = results.mean()
    res.append(mean)


# In[92]:


plt.plot(num_trees, res)
plt.show()


# In[98]:


num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=1)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=1)
results = model_selection.cross_val_score(model, X, np.array(y_class).ravel(), cv=kfold)
print(results.mean())


# In[130]:


X.head(5)


# #### Trying windows

# In[167]:


X_1train = pd.DataFrame(X_train.iloc[:,:8])
X_2train = pd.DataFrame(X_train.iloc[:,5:13])
X_3train = pd.DataFrame(X_train.iloc[:,8:16])
X_4train = pd.DataFrame(X_train.iloc[:,13:21])

X_1test = pd.DataFrame(X_test.iloc[:,:8])
X_2test = pd.DataFrame(X_test.iloc[:,5:13])
X_3test = pd.DataFrame(X_test.iloc[:,8:16])
X_4test = pd.DataFrame(X_test.iloc[:,13:21])


# In[168]:


model1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_1train, y_class_train)
model2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_2train, y_class_train)
model3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_3train, y_class_train)
model4 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_4train, y_class_train)


# In[211]:


answer = []
for i in range(0, len(X_test)):
    sum = model1.predict(X_1test)[i] + model2.predict(X_2test)[i] + model3.predict(X_3test)[i] + model4.predict(X_4test)[i]
    average = sum/4.0
    if average >= 0.5:
        x = 1
    else:
        x = 0
    answer.append(x)


# In[229]:


sum = 0
for i in range(0, len(answer)):
    diff = abs(y_class_test[i] - answer[i])
    sum = sum + diff
print(sum)

print("Test number classified wrong: ", sum)
print("Test percentage classified wrong: ", sum/len(X_test))


# ## Part 4: Best Performing Models

# In[281]:


def predict_price(X_train, y_train, X_test):
    from sklearn import feature_selection
    import statsmodels.api as sm
    
    X_train_4 = X_train.filter(regex=('comp4'))
    X_test_4 = X_test.filter(regex=('comp4'))
    
    forward2 = feature_selection.SelectKBest(score_func=feature_selection.f_regression, k=4)

    results_for2 = forward2.fit(X_train_4,y_train)
    scores2 = results_for2.scores_

    ind2 = np.argpartition(scores2, -10)[-10:]
    X_for_train2 = X_train_4.iloc[:,ind2]
    X_test_42 = X_test_4.iloc[:,ind2]
    
    company_regr2 = sm.OLS(y_train,X_for_train2).fit()
    y_company_pred2 = company_regr2.predict(X_test_42)
    
    return np.array(y_company_pred2)


# In[235]:


def predict_dir(X_train,y_train,X_test):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    X_1train = pd.DataFrame(X_train.iloc[:,:8])
    X_2train = pd.DataFrame(X_train.iloc[:,5:13])
    X_3train = pd.DataFrame(X_train.iloc[:,8:16])
    X_4train = pd.DataFrame(X_train.iloc[:,13:21])
    
    X_1test = pd.DataFrame(X_test.iloc[:,:8])
    X_2test = pd.DataFrame(X_test.iloc[:,5:13])
    X_3test = pd.DataFrame(X_test.iloc[:,8:16])
    X_4test = pd.DataFrame(X_test.iloc[:,13:21])
    
    model1 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_1train, y_train)
    model2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_2train, y_train)
    model3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_3train, y_train)
    model4 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_4train, y_train)
    
    answer = []
    sum = 0
    for i in range(0, len(X_test)):
        sum = model1.predict(X_1test)[i] + model2.predict(X_2test)[i] + model3.predict(X_3test)[i] + model4.predict(X_4test)[i]
        average = sum/4.0
        if average >= 0.5:
            x = 1
        else:
            x = 0
        answer.append(x)
    
    return np.array(answer) 

