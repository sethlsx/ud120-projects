#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier(min_samples_split=40)
time0 = time()
clf.fit(features_train, labels_train)
print 'The training time is: {}'.format(round(time()-time0,3)), 's'
time0 = time()
pred = clf.predict(features_test)
print 'The prediction time is: {}'.format(round(time()-time0,3)), 's'
acc = accuracy_score(labels_test, pred)
print "The accuracy of decision tree method is: {}".format(acc)

#print the numbers of features of the data

print 'There are {} features in the original data.'.format(len(features_train[0]))



#########################################################


