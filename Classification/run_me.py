import numpy as np
import timeit
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_svmlight_file

#### Reading the data and coverting them to arrays
data = load_svmlight_file("HW2.train.txt",multilabel=True)
data2 = load_svmlight_file("HW2.test.txt",multilabel=True)
data3 = load_svmlight_file("HW2.kaggle.txt",multilabel=True)

### Features of the training set
Features_Train=data[0].todense()
### Features of the test set
Features_Test=data2[0].todense()
### Features of the kaggle test set
Kaggle_Feautres=data3[0].todense()
### Classes of trainig set instances
Class_Train= np.asarray(data[1])
### Classe of test set instances
Class_Test= np.asarray(data2[1])


########### Random Forest
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
start = timeit.default_timer()
clf_rf.fit(Features_Train, Class_Train.ravel())
stop = timeit.default_timer()
print("Fitting runtime for random forest:")
print(stop - start)

start = timeit.default_timer()
rfp=clf_rf.predict(Features_Test)
stop = timeit.default_timer()

print("Prediction runtime for tree:")
print(stop - start)

rfac=accuracy_score(rfp, Class_Test.ravel())
print("Random Forest Accuracy")
print(rfac)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier()
#start = timeit.default_timer()
#clf = clf.fit(Features_Train,Class_Train)
#stop = timeit.default_timer()
#print("Fitting runtime for tree:")
#print(stop - start)
#start = timeit.default_timer()
#tree_pred=clf.predict(Features_Test)
# #stop = timeit.default_timer()
#tree_kaggle=clf.predict(Kaggle_Feautres)
#print("Prediction runtime for tree:")
#print(stop - start)
#tree_ac=accuracy_score(tree_pred, Class_Test)
#print("Tree Accuracy")
#print(tree_ac)

### Logistic regression
from sklearn.linear_model import LogisticRegression
mylr = LogisticRegression()
start = timeit.default_timer()
mylr.fit(Features_Train,Class_Train.ravel())
stop = timeit.default_timer()

print("Fitting runtime for Logistic Regression:")
print(stop - start)

start = timeit.default_timer()
logreg_pred=mylr.predict(Features_Test)
stop = timeit.default_timer()
print("Prediction runtime for Logistic Regression:")
print(stop - start)

lr_ac=accuracy_score(logreg_pred, Class_Test)
print("Logistic Regression Accuracy")
print(lr_ac)

#### Neural Networks
from sklearn.neural_network import MLPClassifier
nclf = MLPClassifier()

start = timeit.default_timer()
nclf.fit(Features_Train,Class_Train.ravel())
stop = timeit.default_timer()

print("Fitting runtime for Neural Net:")
print(stop - start)

start = timeit.default_timer()
neupred=nclf.predict(Features_Test)
stop = timeit.default_timer()

print("Prediction runtime for Neural Net:")
print(stop - start)
nn_ac=accuracy_score(neupred, Class_Test)
print("Neural Network Accuracy")
print(nn_ac)


### Find the best random forest by Corss Validation
#from sklearn.model_selection import GridSearchCV
#p_grid = {'n_estimators': [50,100,150],'max_depth':[10, 30, 50, 70, None]}
#bestrf = GridSearchCV(estimator=clf_rf, param_grid=p_grid, cv=10)

## or use
bestrf = RandomForestClassifier(n_estimators=150,max_depth=70)



######### Fitting best random forest
bestrf.fit(Features_Train, Class_Train.ravel())
rfp2=bestrf.predict(Features_Test)
rfac2=accuracy_score(rfp2, Class_Test.ravel())
print('Best random forest (by CV) accuracy:')
print(rfac2)


########### Find the best random forest by re-substitution
#from sklearn.model_selection import GridSearchCV
#p_grid = {'n_estimators': [50,100,150],'max_depth':[10, 30, 50, 70, None]}
#bestrf2 = GridSearchCV(estimator=clf_rf, param_grid=p_grid, cv=[(slice(None), slice(None))],return_train_score=True)
#bestrf2.best_estimator_

## or use
bestrf2 = RandomForestClassifier(n_estimators=100,max_depth=30)


bestrf2.fit(Features_Train, Class_Train.ravel())
rfp3=bestrf2.predict(Features_Test)
rfac3=accuracy_score(rfp3, Class_Test.ravel())
print('Best random forest (by resubstitution) accuracy:')
print(rfac3)



######### Now we find the most accurate one by using merging training and test data and running CV on them to find the best parameters:
Features_merge=np.concatenate((Features_Train, Features_Test), axis=0)
Class_merge=np.concatenate((Class_Train, Class_Test), axis=0)

from sklearn.model_selection import GridSearchCV
#p_grid = {'n_estimators': [300],'max_depth':[30]}
#bestrf10 = GridSearchCV(estimator=clf_rf, param_grid=p_grid, cv=10,return_train_score=True)

#or use
bestrf10 = RandomForestClassifier(n_estimators=200,max_depth=30)

bestrf10.fit(Features_merge, Class_merge.ravel())

################################### Kaggle Output Generation
######### Ebedding kaggle.py
import csv
import numpy as np

#Save prediction vector in Kaggle CSV format
#Input must be a Nx1, 1XN, or N long numpy vector
def kaggleize(predictions,file):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	kaggle_predictions = np.hstack((ids,predictions)).astype(int)
	writer = csv.writer(open(file, 'w'))
	writer.writerow(['# id','Prediction'])
	writer.writerows(kaggle_predictions)


#### Generating Kaggle outputs using the provided
neur_kaggle=nclf.predict(Kaggle_Feautres)
lr_kaggle=mylr.predict(Kaggle_Feautres)
rf_kaggle=clf_rf.predict(Kaggle_Feautres)
bestrf_kaggle=bestrf.predict(Kaggle_Feautres)
bestrf2_kaggle=bestrf2.predict(Kaggle_Feautres)
bestrf10_kaggle=bestrf10.predict(Kaggle_Feautres)

#### Generating Kaggle fiels
kaggleize(predictions=neur_kaggle,file="kaggle_neur.csv")
kaggleize(predictions=lr_kaggle,file="kaggle_lr.csv")
kaggleize(predictions=rf_kaggle,file="kaggle_rf.csv")
kaggleize(predictions=bestrf_kaggle,file="kaggle_bestrf.csv")
kaggleize(predictions=bestrf2_kaggle,file="kaggle_bestrf2.csv")
kaggleize(predictions=bestrf10_kaggle,file="kaggle_bestrf10.csv")

