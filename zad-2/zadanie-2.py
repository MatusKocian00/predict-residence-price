#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn_evaluation import plot
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import sweetviz as sv


# In[2]:


train_data_path = "train_dummy.csv"
test_data_path = "test_dummy.csv"
train_data_no_dummy_path = 'train.csv'


# In[3]:


train_data = pd.read_csv(train_data_path, low_memory=False)
test_data = pd.read_csv(test_data_path, low_memory=False)
no_dummy = pd.read_csv(train_data_no_dummy_path, low_memory=False)
print(train_data.shape)
print(test_data.shape)


# In[35]:


my_report = sv.analyze(train_data, pairwise_analysis='off')
my_report.show_html()


# In[7]:


fig = px.bar(no_dummy, x='GarageCars', y='SalePrice', color='YearBuilt')
fig.write_image('year-cars-price.png')


# In[10]:


fig = px.scatter(no_dummy, x='TotRmsAbvGrd', y='GrLivArea', color='SalePrice')
fig.write_image('rms-liv-price.png')


# In[16]:


fig = px.scatter(no_dummy, x='Fireplaces', y='GrLivArea', color='SalePrice')
fig.write_image('liv-area-fire-price.png')


# In[20]:


fig = px.scatter(no_dummy, x='YrSold', y='SalePrice', color='SaleType')
fig.write_image('sale-yearsold-saletype.png')


# In[24]:


fig = px.scatter(no_dummy, x='YearBuilt', y='SalePrice', color='PoolArea')
fig.write_image('year-price-poolarea.png')


# In[30]:


fig = px.scatter(no_dummy, x='YearBuilt', y='SalePrice', color='HouseStyle')
fig.write_image('year-price-style.png')


# In[ ]:





# In[34]:


fig = px.scatter(no_dummy, x='BldgType', y='SalePrice', color='RoofStyle')
fig.write_image('roof-price-type.png')


# In[ ]:


#Scaling
scaler = MinMaxScaler()
scaler.fit(train_data)
scaled_train = scaler.transform(train_data)
scaled_train_data = pd.DataFrame(scaled_train, columns=train_data.columns)
scaled_test = scaler.transform(test_data)
scaled_test_data = pd.DataFrame(scaled_test,columns=test_data.columns)


# In[ ]:


print(scaled_train_data.shape)
print(scaled_test_data.shape)


# In[ ]:


plt.figure(figsize=(10,5))
plt.scatter(x=scaled_train_data["LotArea"], y=scaled_train_data['SalePrice'], )
plt.xlabel('Lot Area')
plt.ylabel('Price')
plt.savefig('lot-area-price.png')


# In[ ]:


scaled_train_data = scaled_train_data[scaled_train_data['LotArea'] < 0.25].reset_index(drop=True)
scaled_train_data.shape


# In[ ]:


plt.figure(figsize=(10,5))
plt.scatter(x=scaled_train_data["LotArea"], y=scaled_train_data['SalePrice'], )
plt.xlabel('Lot Area')
plt.ylabel('Price')
plt.savefig('lot-are-price-del.png')


# In[ ]:


#Vypisanie najviac korelujucich hodnot
corr = train_data.corr().abs()
highest_corr = corr.unstack()
sorted_highest_corr = highest_corr.sort_values(ascending=False).drop_duplicates()
sorted_highest_corr[:50]


# In[ ]:


fig = px.imshow(corr)
fig.write_html("corr_matrix.html")


# In[ ]:


scaled_train_data["LotArea"].hist()


# In[ ]:


X_train = scaled_train_data.drop('SalePrice', axis=1)
y_train = scaled_train_data['SalePrice']
X_columns = scaled_train_data.drop('SalePrice', axis=1).columns
X_test = scaled_test_data.drop('SalePrice', axis=1)
y_test = scaled_test_data['SalePrice']


# In[ ]:


print(np.shape(X_train))
print(np.shape(y_train))


# In[ ]:


param_grid = {'max_features': ['sqrt', 'log2',1.0],
              'ccp_alpha': [.01, .001],
              'max_depth' : [5, 6, 7],
              'criterion' :['absolute_error'],
              'min_samples_leaf': [1,2,3]
             }


# In[ ]:


regressor = DecisionTreeRegressor()
grid_search_tree = GridSearchCV(estimator=regressor,
                           param_grid=param_grid,
                           cv=5, verbose=4)
grid_search_tree.fit(X_train, y_train)


# In[ ]:


print("BEST ESTIMATOR: " + str(grid_search_tree.best_estimator_))
print("BEST SCORE: " + str(grid_search_tree.best_score_))


# In[ ]:


tree_results = pd.DataFrame(grid_search_tree.cv_results_)
tree_results.to_csv("tree_results.csv")


# In[ ]:


plt.figure(figsize=(10,5))
plot.grid_search(grid_search_tree.cv_results_, change="max_depth", kind='bar', sort=False)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig('tree_grid.png',bbox_inches="tight")


# In[ ]:


best_tree_regressor = DecisionTreeRegressor(ccp_alpha=0.001, criterion='absolute_error', max_depth=5,min_samples_leaf=2)
best_tree_regressor.fit(X_train, y_train)


# In[ ]:


y_pred = best_tree_regressor.predict(X_test)
plot.residuals(y_test, y_pred)
plt.savefig('residual-tree.png')


# In[ ]:


#tree_test_results
print("R2 score: "+ str(r2_score(y_true=y_test,y_pred=y_pred)))
print("MSE score: "+ str(mean_squared_error(y_pred=y_pred,y_true=y_test)))


# In[ ]:


text_representation = tree.export_text(best_tree_regressor)
feature_names = list(scaled_train_data.columns.values)

fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(best_tree_regressor,feature_names= feature_names,
                   filled=True)
fig.savefig('decisionTree.png')


# In[ ]:


param = {'kernel' : ('linear', 'poly', 'rbf'),'C' : [1,5,10,100],'gamma' : (0.1,0.01,0.001)},

svrGridSearch = GridSearchCV(estimator=SVR(),param_grid=param,
                             cv=5,
                             verbose=4,)

svrGridSearch.fit(X_train,y_train)


# In[ ]:


grid_scores = svrGridSearch.cv_results_
tree_results = pd.DataFrame(grid_scores)
#tree_results = tree_results.sort_values("rank_test_r2")
tree_results.to_csv("svm_results.csv")


# In[ ]:


print("BEST ESTIMATOR: " + str(svrGridSearch.best_estimator_))
print("BEST SCORE: " + str(svrGridSearch.best_score_))
print("BEST PARAMETERS" + str(svrGridSearch.best_params_))


# In[ ]:


svrGridSearchResults = svrGridSearch.cv_results_
ax = plot.grid_search(svrGridSearch.cv_results_, change="gamma", kind='bar', sort=False)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig('svr-grid.png',bbox_inches="tight")


# In[ ]:


from sklearn_evaluation import plot
bestSvr = SVR(C=1,gamma='auto',kernel='rbf', verbose=False)
bestSvr.fit(X_train,y_train)


# In[ ]:


y_pred = bestSvr.predict(X_test)
y_true = y_test
plot.residuals(y_true, y_pred)
plt.savefig('svr-residuals.png')


# In[ ]:


y_pred = bestSvr.predict(X_test)
print("R2 score: "+ str(r2_score(y_true=y_test,y_pred=y_pred)))
print("MSE score: "+ str(mean_squared_error(y_pred=y_pred,y_true=y_test)))


# In[ ]:


parameters = {
    'n_estimators': [100, 150, 200, 250, 300],
    'max_depth': [1,2,3,4],
}
randomForestRegressor = RandomForestRegressor(ccp_alpha=0.001, criterion='absolute_error',min_samples_leaf=2)

grid_search = GridSearchCV(estimator=randomForestRegressor,
                           param_grid=parameters,
                           cv=5, verbose=4)
grid_search.fit(X_train, y_train)


# In[ ]:


print("BEST FOREST ESTIMATOR: " + str(grid_search.best_estimator_))
print("BEST SCORE: " + str(grid_search.best_score_))
print("BEST PARAMETERS" + str(grid_search.best_params_))


# In[ ]:


plot.grid_search(grid_search.cv_results_, change="n_estimators", kind='bar', sort=False)
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig('random-f-grid.png',bbox_inches="tight")


# In[ ]:


bestRandomForest = RandomForestRegressor(max_depth=4,n_estimators=150,ccp_alpha=0.001, criterion='absolute_error',min_samples_leaf=2)
bestRandomForest.fit(X_train, y_train)


# In[ ]:


y_pred = bestRandomForest.predict(X_test)
plot.residuals(y_true=y_test, y_pred=y_pred)
plt.savefig('plt-residual.png')


# In[ ]:


print("R2 score: "+ str(r2_score(y_true=y_test,y_pred=y_pred)))
print("MSE score: "+ str(mean_squared_error(y_pred=y_pred,y_true=y_test)))


# In[ ]:


feat_importances = pd.Series(bestRandomForest.feature_importances_, index=X_train.columns).sort_values()
feat_importances.nlargest(20).plot(kind='barh')


# In[ ]:


fig = px.bar(feat_importances, orientation='h')
fig.write_html('importances.html')


# In[ ]:


X = X_test
y = y_test
y_pred = bestSvr.predict(X)
fig = px.scatter(x=y, y=y_pred, labels={'x': 'ground truth', 'y': 'prediction'}, title="Prediction vs Expected SVR")
fig.add_shape(
    type="line", line=dict(dash='dash'),
    x0=y.min(), y0=y.min(),
    x1=y.max(), y1=y.max()
)
fig.show()


# In[ ]:


import plotly.express as px

df = scaled_test_data

# Condition the model on sepal width and length, predict the petal width
df['prediction'] = bestSvr.predict(X_test)
df['residual'] = df['prediction'] - scaled_test_data['SalePrice']

fig = px.scatter(
    df, x='prediction', y='residual',
    marginal_y='violin', trendline='ols', title="Residual SVR"
)
fig.write_html('residual-trendline-svr.html')


# In[ ]:


import plotly.express as px

df = scaled_test_data

df['prediction'] = bestRandomForest.predict(X_test)
df['residual'] = df['prediction'] - scaled_test_data['SalePrice']

fig = px.scatter(
    df, x='prediction', y='residual',
    marginal_y='violin', trendline='ols', title="Residual Random Forrest"
)
fig.write_html('residual-trendline-forest.html')


# In[ ]:


import plotly.express as px

df = scaled_test_data

df['prediction'] = best_tree_regressor.predict(X_test)
df['residual'] = df['prediction'] - scaled_test_data['SalePrice']

fig = px.scatter(
    df, x='prediction', y='residual',
    marginal_y='violin', trendline='ols', title="Residual Tree"
)
fig.write_html('residual-trendline-tree.html')


# In[ ]:


fig = px.scatter_3d(train_data, x='TotRmsAbvGrd', y='GrLivArea', z='YearBuilt',
                    color='SalePrice', symbol='GarageCars')
fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
                                          ticks="outside"))
fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 20)
))
fig.write_html('3d.html')


# In[ ]:


import umap.umap_ as umap
X = X_train
reducer = umap.UMAP(n_components=3, min_dist=0.1, n_neighbors=50).fit(X)
umap_train_data = reducer.transform(X)


# In[ ]:


umap_train_data


# In[ ]:


df_umap = pd.DataFrame(umap_train_data)
df_umap[['SalePrice','GarageCars']] = train_data[["SalePrice","GarageCars"]]
fig = px.scatter_3d(df_umap, x=0, y=1, z=2,
                    color='SalePrice', symbol='GarageCars')
fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0,
                                          ticks="outside"))
fig.update_layout(legend=dict(title_font_family="Times New Roman",
                              font=dict(size= 20)
))
fig.write_html('3d_umap.html')


# In[ ]:


X = X_train
pca = PCA(n_components=3)
components = pca.fit_transform(X)

total_var = pca.explained_variance_ratio_.sum() * 100

fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=scaled_train_data['SalePrice'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
)
fig.write_html('3d_pca.html')


# In[ ]:


for i in range (0,255,5):
    if i == 1:
        continue
    pca = PCA(n_components=i, random_state=2020)
    components = pca.fit_transform(scaled_train_data)
    print("VARIANCE EXPLAINED BY ALL " + str(i) + " PRINCIPAL COMPONENTS = " + str(sum(pca.explained_variance_ratio_ *100)))


# In[ ]:


most_corr_columns = sorted_highest_corr[:2000].reset_index()
most_corr =[]
reducted_train_data = X_train.copy()
reducted_test_data = X_test.copy()
for row in most_corr_columns['level_0']:
    if row in reducted_train_data.columns:
        reducted_train_data = reducted_train_data.drop(row, axis=1)
        reducted_test_data = reducted_test_data.drop(row, axis=1)
        most_corr.append(row)

updated_most_corr = reducted_train_data.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
least_corr_train = X_train.drop(columns = most_corr)
most_corr_train = X_train.drop(columns = least_corr_train.columns)

least_corr_test = X_test.drop(columns = most_corr)
most_corr_test = X_test.drop(columns = least_corr_test.columns)

print("POCET NAJVIAC KORELUJUCICH STLPCOV PRI 1000 HODNOTACH ", len(most_corr_train.columns))
print("POCET NAJMENEJ KORELUJUCICH STLPCOV PRI 1000 HODNOTACH ", len(least_corr_train.columns))
print(least_corr_train.shape)
print(most_corr_train.shape)
print(least_corr_test.shape)
print(most_corr_test.shape)


# In[ ]:


#4
pca4 = PCA(n_components=4, random_state=2020)
pca4.fit(most_corr_train)
train_components4 = pca4.transform(most_corr_train)
test_components4 = pca4.transform(most_corr_test)
print("VARIANCE EXPLAINED BY ALL " + str(4) + " PRINCIPAL COMPONENTS = " + str(sum(pca4.explained_variance_ratio_ *100)))


# In[ ]:


#5
pca5 = PCA(n_components=5, random_state=2020)
pca5.fit(most_corr_train)
train_components5 = pca5.transform(most_corr_train)
test_components5 = pca5.transform(most_corr_test)
print("VARIANCE EXPLAINED BY ALL " + str(5) + " PRINCIPAL COMPONENTS = " + str(sum(pca5.explained_variance_ratio_ *100)))


# In[ ]:


#6
pca6 = PCA(n_components=6, random_state=2020)
pca6.fit(most_corr_train)
train_components6 = pca6.transform(most_corr_train)
test_components6 = pca6.transform(most_corr_test)
print("VARIANCE EXPLAINED BY ALL " + str(6) + " PRINCIPAL COMPONENTS = " + str(sum(pca6.explained_variance_ratio_ *100)))


# In[ ]:


#7
pca7 = PCA(n_components=7, random_state=2020)
pca7.fit(most_corr_train)
train_components7 = pca7.transform(most_corr_train)
test_components7 = pca7.transform(most_corr_test)
print("VARIANCE EXPLAINED BY ALL " + str(7) + " PRINCIPAL COMPONENTS = " + str(sum(pca7.explained_variance_ratio_ *100)))


# In[ ]:


#10
pca10 = PCA(n_components=10, random_state=2020)
pca10.fit(most_corr_train)
train_components10 = pca10.transform(most_corr_train)
test_components10 = pca10.transform(most_corr_test)
print("VARIANCE EXPLAINED BY ALL " + str(10) + " PRINCIPAL COMPONENTS = " + str(sum(pca10.explained_variance_ratio_ *100)))


# In[ ]:


pca15 = PCA(n_components=15, random_state=2020)
pca15.fit(most_corr_train)
train_components15 = pca15.transform(most_corr_train)
test_components15 = pca15.transform(most_corr_test)
print("VARIANCE EXPLAINED BY ALL " + str(15) + " PRINCIPAL COMPONENTS = " + str(sum(pca15.explained_variance_ratio_ *100)))


# In[ ]:


train_reductions = [train_components4,train_components5,train_components6,train_components7,train_components10,train_components15]


# In[ ]:


test_reductions = [test_components4,test_components5,test_components6,test_components7,test_components10,test_components15]


# In[ ]:


def Average(lst):
    return sum(lst) / len(lst)

def train_with_reduction(train_p, test_p, model):
    reducted_train_data_pca = least_corr_train.copy()
    train_pca_reduction = pd.DataFrame(train_p)
    ready_train = pd.concat([reducted_train_data_pca,train_pca_reduction], axis=1, join='inner')
    ready_train.columns = ready_train.columns.map(str)

    reducted_test_data_pca = least_corr_test.copy()
    test_pca_reduction = pd.DataFrame(test_p)
    ready_test = pd.concat([reducted_test_data_pca,test_pca_reduction], axis=1, join='inner')
    ready_test.columns = ready_train.columns.map(str)

    train_result_best_random_forest_pca = cross_validate(model,ready_train,y_train,scoring=["r2"], return_estimator=True, return_train_score=True)
    r2test = []
    for i in range(4):
        y_predicted = train_result_best_random_forest_pca['estimator'][i].predict(ready_test)
        test_result = r2_score(y_test,y_predicted)
        r2test.append(test_result)
    print("Pocet priznakov: " + str(len(ready_train.columns)) + " Pocet dimenzii redukovanej podmnoziny: " + str(len(train_pca_reduction.columns)))
    print('\n')
    return [len(train_pca_reduction.columns),train_result_best_random_forest_pca, Average(r2test)]


# In[ ]:


fit_times = []
dimensions = []
r2score = []
test_r2 =[]
for (train, test) in zip(train_reductions, test_reductions):
     result = train_with_reduction(train,test, bestRandomForest)
     fit_times.append(result[1]['fit_time'].mean())
     dimensions.append(result[0])
     r2score.append(result[1]['train_r2'].mean())
     test_r2.append(result[2])
print(fit_times)
print(dimensions)
print(r2score)
print(test_r2)


# In[ ]:


df = pd.DataFrame(list(zip(dimensions, fit_times, r2score,test_r2)),
               columns =['dimension', 'fit_time','r2score','test_r2'])
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Bar(x=dimensions, y=fit_times, name="Mean Fit time",marker=dict(color='rgb(34,163,192)'))
    ,secondary_y=False
)

fig.add_trace(
    go.Scatter(x=dimensions, y=r2score, name="Mean Train R2 score"),
    secondary_y=True,
)
fig.add_trace(
    go.Scatter(x=dimensions, y=test_r2, name="Mean test R2 score"),
    secondary_y=True
)

# Add figure title
fig.update_layout(
    title_text="R2 Score with Fit times based on reductions"
)

# Set x-axis title
fig.update_xaxes(title_text="Dimensions")

# Set y-axes titles
fig.update_yaxes(title_text="Fit time", secondary_y=False)
fig.update_yaxes(title_text="R2 score", secondary_y=True)

fig.write_html('reducted_dimension.html')
fig.write_image('reducted_dimension.png')

