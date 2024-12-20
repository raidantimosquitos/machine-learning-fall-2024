'''
ex3svc.py
This script defines the binaryClass_SVM() function. Basically splits the input dataset
into train and test subsets, these are fitted through the Support Vector Machine to
perform binary classification (setosa vs not-setosa). Finally accuracy as the final
performance metric is returned.
'''

# third-party libraries
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import torch

def binaryClass_SVM(dataset):
    train, test = train_test_split(dataset, test_size=0.2, random_state=1234)
    n_train_samples, n_train_features = len(train), len(train[0][0])
    n_test_samples, n_test_features = len(test), len(test[0][0])

    print('Splitting the dataset in train/test on a 80/20 proportion: ')
    print(f'Train dataset for classification shape: {n_train_samples}, {n_train_features}')
    print(f'Test dataset for classification shape: {n_test_samples}, {n_test_features}')

    train_feat_only = torch.stack([train[i][0] for i in range(len(train))])
    train_targ_only = torch.stack([train[i][1] for i in range(len(train))])
    test_feat_only = torch.stack([test[i][0] for i in range(len(test))])
    test_targ_only = torch.stack([test[i][1] for i in range(len(test))])
    clf = svm.SVC()
    clf.fit(train_feat_only, train_targ_only)

    y_pred = clf.predict(test_feat_only)

    acc = accuracy_score(test_targ_only, y_pred)

    print("\nSupport Vector Machine Classifier, test metrics: ")    
    print(f"\tAccuracy: {acc:.4f}")
    print()
