import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import lightgbm as lgb
from sklearn import metrics
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix
import shap
import joblib
import pickle
from imblearn.over_sampling import SMOTE


def gb_model(df_train_x,df_train_y,df_test_x,df_test_y):
    model = lgb.LGBMClassifier(learning_rate=0.01,num_leaves=100,max_depth=15, early_stopping_rounds=10, num_iterations=3000, random_state=42)

    model.fit(df_train_x,df_train_y,eval_set=[(df_train_x,df_train_y),(df_test_x,df_test_y)],
            eval_metric='binary')
    return model

model_method=gb_model
density=sys.argv[1]
fa=sys.argv[2]

def exp(density,fa):
    cols_names=['area', 'perimeter', 'neighbours', 'max neighbour distance',
        'min neighbour distance', 'max vertices distance',
        'min vertices distance', 'max vertices-point distance',
        'min vertices-point distance', 'distance to center', 'activity',
        'particle type']

    input_file=f"phia{density}/particles-features-{density}-Fa{fa}.txt"
    data = pd.read_csv(input_file, delimiter=' ',names=cols_names)

    target='activity'
    smote=SMOTE(sampling_strategy=0.5) #that's how we balance the data

    df_train, df_test = train_test_split(data, random_state=50, test_size=0.3)
    df_train_y = df_train[target].copy().astype('int')
    df_train.drop(columns=target, inplace=True)
    df_train_x = df_train
    df_train_x,df_train_y=smote.fit_resample(df_train_x,df_train_y)

    df_test_y = df_test[target].copy().astype('int')
    df_test.drop(columns=target, inplace=True)
    df_test_x = df_test
    df_test_x,df_test_y=smote.fit_resample(df_test_x,df_test_y)
    model=model_method(df_train_x,df_train_y,df_test_x,df_test_y)
    features=list(df_test_x.columns)

    joblib.dump(value=[model,features,target,df_test_y,df_test_x],filename=f"phia{density}/gb-model-{density}-Fa{fa}-balanced.pkl")
    #We create a .pkl file that saves all the useful values of the model so we can later score it and make predictions without training it another time
    return 1

exp(density,fa)

#If we want to obtain models for a list of fa we can run:
# fa_list=[50]
# for fa in fa_list:
#     exp(density,fa)

