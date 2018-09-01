
# coding: utf-8

# In[78]:

#!/usr/bin/python
import time
start_time = time.time()

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier


# In[79]:

#------------------------------------------------------------------
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

poi = ['poi']
financial_features = ['salary','bonus','deferral_payments','deferred_income',
                      'exercised_stock_options','expenses','long_term_incentive',
                      'other','restricted_stock','total_payments','total_stock_value']
email_features = ['from_messages','from_poi_to_this_person','from_this_person_to_poi',
                  'shared_receipt_with_poi','to_messages']
features_list =  poi + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
print "Size dataset: " , len(data_dict)
poi = 0
not_poi = 0
for k in data_dict:
    if data_dict[k]['poi'] == True:
        poi += 1
    if data_dict[k]['poi'] == False:
        not_poi += 1
        
print "Number of POI: " , poi
print "Number of not POI: " , not_poi
print "Number of financial Features: " , len(financial_features)
print "Number of email Features: " , len(email_features)


# In[80]:

#-----------------------------------------------------------------    
### Task 2: Remove outliers

# remove 'TOTAL' from dictionary
del data_dict['TOTAL']

# remove 'THE TRAVEL AGENCY IN THE PARK' from dictionary
del data_dict['THE TRAVEL AGENCY IN THE PARK']

# remove negative values from 'restricted_stock'
for person in data_dict:
        if data_dict[person]['restricted_stock'] < 0 and data_dict[person]['restricted_stock'] != 'NaN':
            data_dict[person]['restricted_stock'] = 'NaN'

# remove negative values from 'deferral_payments'
for person in data_dict:
        if data_dict[person]['deferral_payments'] < 0 and data_dict[person]['deferral_payments'] != 'NaN':
            data_dict[person]['deferral_payments'] = 'NaN'

# remove negative values from 'total_stock_value'
for person in data_dict:
        if data_dict[person]['total_stock_value'] < 0 and data_dict[person]['total_stock_value'] != 'NaN':
            data_dict[person]['total_stock_value'] = 'NaN'
            
# Remove 'restricted_stock_deferred' and 'loan_advances' from the features, few relevant data available

# Remove 'director_fee' because there is only non-POI data


# In[81]:

# Checking if had some person without value
not_NaN_data = {}
for key in data_dict:
    not_NaN_feature = 0
    for feature in data_dict[key]:
        if data_dict[key][feature] != 'NaN':
            not_NaN_feature += 1
    not_NaN_data[key] = not_NaN_feature

for k in not_NaN_data:
    if not_NaN_data[k] == 1:
        print k
        print data_dict[k]


# In[82]:

# remove 'THE TRAVEL AGENCY IN THE PARK' from dictionary
del data_dict['LOCKHART EUGENE E']


# In[83]:

#------------------------------------------------------------------
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

### The messages to and from POI are an absolute measure, let's create new features that are a ratio of the total messages.
def new_feature_ratio(new_feature, numerator, denominator):    
    for key in data_dict:
        if data_dict[key][denominator] != 'NaN' and data_dict[key][numerator] != "NaN":
            data_dict[key][new_feature] = float(data_dict[key][numerator]) / float(data_dict[key][denominator])
        else:
            data_dict[key][new_feature] = "NaN"
    features_list.append(new_feature)
    
### Feature - 'from_this_person_to_poi_ratio'
new_feature_ratio('from_this_person_to_poi_ratio', 'from_this_person_to_poi', 'from_messages')

### Feature - 'from_poi_to_this_person_ratio'
new_feature_ratio('from_poi_to_this_person_ratio', 'from_poi_to_this_person', 'to_messages')
    
### Feature - 'bonus_ratio'
new_feature_ratio('bonus_ratio', 'bonus', 'salary')


# In[84]:

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
my_dataset = data_dict

### Put features with "long tail" in log10 scale
features_list_log = ['salary','bonus','deferral_payments','exercised_stock_options',
                     'expenses','long_term_incentive','other','restricted_stock',
                     'total_payments','total_stock_value', 'from_messages', 
                     'from_poi_to_this_person', 'from_this_person_to_poi', 
                     'shared_receipt_with_poi', 'bonus_ratio']
features_list_log = []
for n in range(1,len(features_list_log)):
    for person in my_dataset:
        if my_dataset[person][features_list_log[n]] != "NaN":
            if my_dataset[person][features_list_log[n]] >= 0:
                if my_dataset[person][features_list_log[n]] == 0:
                    my_dataset[person][features_list_log[n]] = 0
            else:
                my_dataset[person][features_list_log[n]] = np.log10(my_dataset[person][features_list_log[n]]*-1)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Put all features in same reange (0,1)
for n in range(0,len(features[0])):
    feature = []
    for person in range(0,len(features)):
        feature.append(features[person][n])
    feature = np.array(feature).reshape(-1,1)
    feature = scaler.fit_transform(feature)
    for person in range(0,len(features)):
        features[person][n] = feature[person]


# In[85]:

#-----------------------------------------------------------------
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.model_selection import GridSearchCV

from sklearn.naive_bayes import GaussianNB
clf_NB = GaussianNB()

from sklearn import svm
from sklearn.svm import SVC
parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 10], 'degree': [2,10]}
svr = svm.SVC()
clf_SVM = GridSearchCV(svr, parameters, scoring = 'f1')

from sklearn.tree import DecisionTreeClassifier
parameters = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'min_samples_split':[2,200]}
svr = DecisionTreeClassifier()
clf_tree = GridSearchCV(svr, parameters, scoring = 'f1')

from sklearn.ensemble import RandomForestClassifier
parameters = {'n_estimators': [2,20], 'criterion':('gini', 'entropy'), 'min_samples_split':[2,200]}
svr = RandomForestClassifier()
clf_randon_forest = GridSearchCV(svr, parameters, scoring = 'f1')

classifiers = {"clf_NB": clf_NB,
               "clf_SVM": clf_SVM,
               "clf_tree": clf_tree,
               "clf_randon_forest": clf_randon_forest}


#----------------------------------------------------------------
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

### Using K-fold
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.pipeline import make_pipeline

def train_test_StratifiedKFold(clf, k_best, features):
    # Enter a classifier and the number of the k-best features and the function return
    # the classifier and validation metrics
    acc = []
    pre = []
    rec = []
    f = []
    skf = StratifiedKFold(2, shuffle=True)
    for train_index, test_index in skf.split(features, labels):
        features_train = [features[ii] for ii in train_index] 
        labels_train = [labels[ii] for ii in train_index]
        features_test = [features[ii] for ii in test_index]
        labels_test = [labels[ii] for ii in test_index]
        
        skb = SelectKBest(f_classif, k = k_best)
        pipe = make_pipeline(skb, clf)
        pipe.fit(features_train, labels_train)
        labels_pred = pipe.predict(features_test)
        acc.append(accuracy_score (labels_test, labels_pred))
        pre_rec_f = precision_recall_fscore_support (labels_test, labels_pred)
        try:
            pre.append(pre_rec_f[0][1])
        except:
            pass
        try:
            rec.append(pre_rec_f[1][1])
        except:
            pass
        try:
            f.append(pre_rec_f[2][1])
        except:
            pass
    return [pipe, np.mean(acc), np.mean(pre), np.mean(rec), np.mean(f)]


# In[86]:

#---------------------------------------------------------
# Now we will test the best classifiers

best_clf = [None, None, None]

# We will test all combination of the 4 algoritms and k-best features (k from 1 to 19). 
# For each metric (accuracy, precision, recall and f) we will print the best combination.
# We will try 5 times to be sure to choose the best combination
for test in range(1,6):
    max_acc = [0, 'NaN', 'NaN']
    max_pre = [0, 'NaN', 'NaN']
    max_rec = [0, 'NaN', 'NaN']
    max_f = [0, 'NaN', 'NaN']
    for algor in classifiers:
        for k_best in range(1, 17): #20):
            preview_clf, acc, pre, rec, f = train_test_StratifiedKFold(classifiers[algor], k_best, features)
            if acc > max_acc[0]:
                max_acc = [acc, algor, k_best]
            if pre > max_pre[0]:
                max_pre = [pre, algor, k_best]
            if rec > max_rec[0]:
                max_rec = [rec, algor, k_best]
            if f > max_f[0]:
                max_f = [f, algor, k_best]
                best_clf = ['k-best', max_f, preview_clf]

    print ""
    print "Test k-best ", test
    print 'Accuracy: ', max_acc
    print 'Precision: ', max_pre
    print 'Reccal: ', max_rec
    print 'f Score: ', max_f

### We will do the same but decomponding the features using PCA (n° of componnents 1 to 19)
from sklearn.decomposition import PCA

for test in range(1,6):
    max_acc = [0, 'NaN', 'NaN']
    max_pre = [0, 'NaN', 'NaN']
    max_rec = [0, 'NaN', 'NaN']
    max_f = [0, 'NaN', 'NaN']
    for algor in classifiers:
        for n_comp in range(1, 17): #20):
            pca = PCA(n_components = n_comp)
            pipe = make_pipeline(pca, classifiers[algor])
            #pca_features = pca.fit_transform(features)
            preview_clf, acc, pre, rec, f = train_test_StratifiedKFold(pipe, "all", features)
            if acc > max_acc[0]:
                max_acc = [acc, algor, n_comp]
            if pre > max_pre[0]:
                max_pre = [pre, algor, n_comp]
            if rec > max_rec[0]:
                max_rec = [rec, algor, n_comp]
            if f > max_f[0]:
                max_f = [f, algor, n_comp]
            if f > best_clf[1][0]:
                best_clf = ['PCA', max_f, preview_clf]

    print ""
    print "Test PCA", test
    print 'Accuracy: ', max_acc
    print 'Precision: ', max_pre
    print 'Reccal: ', max_rec
    print 'f Score: ', max_f


# In[87]:

#--------------------------------------------------------
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print "f classi: ", best_clf[1]
print "K-best or PCA: ", best_clf[0]
print "Classifier:"
print best_clf[2]


### The best classifier is
clf = best_clf[2]


dump_classifier_and_data(clf, my_dataset, features_list)
print ""
test_classifier(clf, my_dataset, features_list, folds = 1000)

print ""
print("--- %s seconds ---" % (time.time() - start_time))

