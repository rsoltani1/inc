# Import python modules
import numpy as np
import kaggle
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import neighbors

# Read in train and test data

def read_data_fb():
	print('Reading facebook dataset ...')
	train_x = np.loadtxt('../../Data/data.csv', delimiter=',')
	train_y = np.loadtxt('../../Data/labels.csv', delimiter=',')
	train_kaggle = np.loadtxt('../../Data/kaggle_data.csv', delimiter=',')

	return (train_x, train_y, train_kaggle)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

############################################################################
train_x, train_y, kaggle_x   = read_data_fb()
print('Train=', train_x.shape)
print('Kaggle=', kaggle_x.shape)

######### normalizing features
train_x2=StandardScaler().fit_transform(train_x)
kaggle_x2=StandardScaler().fit_transform(kaggle_x)


########### Question 1: Cross Validation on a Tree
### Make a tree
mytree = tree.DecisionTreeRegressor()

### Setting values for maximum depth of the tree
p_grid = {'max_depth':[3,6,9,12,15]}
print('Performing CV on the Tree ...')
besttree = GridSearchCV(estimator=mytree, param_grid=p_grid, cv=5,return_train_score=True,scoring='neg_mean_absolute_error')
besttree.fit(train_x,train_y)
print('Done with CV on the Tree')
print('Best tree has a depth of:', besttree.best_params_)
print('out of sample errors:', abs(besttree.cv_results_['mean_test_score']))
print('max depth:', [3,6,9,12,15])

####### Finding the runtime which is the sum of fitting time and scoring time
treecv_runtimes=500*(besttree.cv_results_['mean_fit_time']+besttree.cv_results_['mean_score_time'])

#### Plotting time time versus depth of the tree
plt.plot([3,6,9,12,15],treecv_runtimes)
plt.ylabel('Cross Validation Runtime (ms)')
plt.xlabel('Maximum Debth the Tree')
plt.show()

###### Observing the best error for the tree comming from the CV
print('Error from CV:',(besttree.cv_results_['mean_test_score'].max())*(-1))
#tree_res=besttree.cv_results_


##### Kaggelizing the output to see Test MAE:
file_name = '../Predictions/besttree.csv'
print('Writing output to ', file_name)
predicted_y=besttree.predict(kaggle_x)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing')


######### Question 2, Cross Validation on KNN Regressor
###### Making KNN
myknn = neighbors.KNeighborsRegressor()

##### Possible value for number of neighbors
p_grid = {'n_neighbors':[3,5,10,20,25]}

# PErforming Cross Validation

print('performing CV on KNN ...')
bestknn = GridSearchCV(estimator=myknn, param_grid=p_grid, cv=5,return_train_score=True,scoring='neg_mean_absolute_error')
bestknn.fit(train_x,train_y)
print("Done with CV on KNN")

########## Printing outputs
print('out of sample errors:', abs(bestknn.cv_results_['mean_test_score']))
print('number of neighbors:', [3,5,10,20,25])
print('The number of neighbors of the best KNN:', bestknn.best_params_)

##### Kaggelizing the output
predicted_y=bestknn.predict(kaggle_x)
file_name = '../Predictions/bestknn.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')

###### KNN with Manhattan Distance

myknn2 = neighbors.KNeighborsRegressor(n_neighbors=5,metric='manhattan')
myknn2.fit(train_x,train_y)
predicted_y=myknn2.predict(kaggle_x)
file_name = '../Predictions/bestknnmanhattan.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')

###### KNN with Chebyshev Distance
myknn3 = neighbors.KNeighborsRegressor(n_neighbors=5,metric='chebyshev')
myknn3.fit(train_x,train_y)
predicted_y=myknn3.predict(kaggle_x)
file_name = '../Predictions/bestknnchebyshev.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')


##### Question 3
########### Ridge Regression, CV

from sklearn import linear_model
myridge = linear_model.Ridge()
p_grid = {'alpha':[1e-6,1e-4,1e-2,1,10]}
bestridge = GridSearchCV(estimator=myridge, param_grid=p_grid, cv=5,return_train_score=True,scoring='neg_mean_absolute_error')
bestridge.fit(train_x2,train_y)
print("Done with CV on Rdige")
#print(bestridge.cv_results_)
print(bestridge.best_params_)
print('out of sample errors:', abs(bestridge.cv_results_['mean_test_score']))
print('alpha:', [1e-6,1e-4,1e-2,1,10])

########## Lasso
from sklearn import linear_model
mylasso = linear_model.Lasso(tol=1)
p_grid = {'alpha':[1e-6,1e-4,1e-2,1,10]}
bestlasso = GridSearchCV(estimator=mylasso, param_grid=p_grid, cv=5,return_train_score=True,scoring='neg_mean_absolute_error')
bestlasso.fit(train_x2,train_y)
print("Done with CV on Lasso")
#print(bestridge.cv_results_)
print(bestlasso.best_params_)
print('out of sample errors:', abs(bestlasso.cv_results_['mean_test_score']))
print('alpha:', [0.000001,0.0001,0.01,1,10])

########## Best Lasso For Kaggle
mylasso2 = linear_model.Lasso(alpha=10,tol=1)
mylasso2.fit(train_x2,train_y)
predicted_y=mylasso2.predict(kaggle_x2)
file_name = '../Predictions/bestlasso.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')

### We find the indices that sort the absolute value of the wieghts:
exweights_temp=np.argsort(abs(mylasso2.coef_))
exw=exweights_temp[0:4]+1
print('we can exclude the following features:',exw)



##### Question 4, SVR
from sklearn.svm import SVR
mysvr = SVR(kernel='poly')
p_grid = {'degree':[1,2]}
bestsvr1 = GridSearchCV(estimator=mysvr, param_grid=p_grid, cv=5,return_train_score=True,scoring='neg_mean_absolute_error')
bestsvr1.fit(train_x2,train_y)
print("Done with CV on Polynomial SVM")

print('out of sample errors for polynomial SVM:', abs(bestsvr1.cv_results_['mean_test_score']))
print('degree of the polynomial', [1,2])

from sklearn.model_selection import cross_val_score
svr3 = SVR(kernel='rbf')
scores = cross_val_score(svr3, train_x2, train_y, cv=5,scoring='neg_mean_absolute_error')
print("Done with CV on RBF SVM")
print('out of sample errors for RBF SVM:', abs((scores.mean())))

bestsvr=SVR(kernel='poly',degree=1)
bestsvr.fit(train_x2,train_y)
predicted_y=bestsvr.predict(kaggle_x2)
file_name = '../Predictions/bestsvr.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')



### Question 5
from sklearn.neural_network import MLPRegressor
mynn=MLPRegressor(max_iter=400)
p_grid = {'hidden_layer_sizes':[(10,),(20,),(30,),(40,)]}
#p_grid = {'hidden_layer_sizes':[(20,)]}
bestnn = GridSearchCV(estimator=mynn, param_grid=p_grid, cv=5,return_train_score=True,scoring='neg_mean_absolute_error')
bestnn.fit(train_x2,train_y)
print("Done with CV on NN")
print(bestnn.best_params_)
print('out of sample errors for NN', abs(bestnn.cv_results_['mean_test_score']))
print('number of neurons in the single hidden layer:', [10,20,30,40])

bestnn.fit(train_x2,train_y)
predicted_y=bestnn.predict(kaggle_x2)
file_name = '../Predictions/bestnn.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')


### Question 6
from sklearn.neural_network import MLPRegressor
mynnk=MLPRegressor(max_iter=400)
p_grid = {'hidden_layer_sizes':[(10,20,30,40,50,60,70,80,90,100),(10,20,30,40)]}
bestnnk = GridSearchCV(estimator=mynnk, param_grid=p_grid, cv=10,return_train_score=True,scoring='neg_mean_absolute_error')
bestnnk.fit(train_x2,train_y)
print("Done with CV on NN for Kaggle")
print(bestnnk.best_params_)
bestnnk.fit(train_x2,train_y)
predicted_y=bestnnk.predict(kaggle_x2)
predicted_y=predicted_y.clip(min=0)
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')

### Also see the code here which resulted the best kaggle
############# This found the best output
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(100, input_shape=(52,)),
    Activation('softmax'),
    Dense(40),
    Activation('sigmoid'),
    Dense(1),
    Activation('relu')
])

model.compile(optimizer='sgd',
              loss='mean_absolute_error')

model.fit(train_x2, train_y, epochs=200, batch_size=52)

predicted_y=model.predict(kaggle_x2)
predicted_y=predicted_y.clip(min=0)
file_name = '../Predictions/bestbest.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
print('Done with writing.')
