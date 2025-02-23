import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
# import matplotlib
# matplotlib.use('Agg')
import lightgbm as lgb
from sklearn import metrics
from numpy.ma import MaskedArray
import sklearn.utils.fixes
sklearn.utils.fixes.MaskedArray = MaskedArray
from sklearn.model_selection import train_test_split
#from sklearn.metrics import plot_confusion_matrix
import shap

if sys.argv[1] is None :
    print("Required the number of cores as input")
    sys.exit(1)

# Whether to keep a single core free for the 'control script'
ONE_CORE_FREE = True
# How many CPU cores to use
# Should be set by running this file like `nn_optimize.py $SLURM_NTASKS <training input file> <test input file>`
if ONE_CORE_FREE:
    N_JOBS = max(1, int(sys.argv[1]) - 1)
else:
    N_JOBS = int(sys.argv[1])
print("Number of cores: ", str(N_JOBS))
# Force running with a limited number of threads
os.environ['OPENBLAS_NUM_THREADS'] = str(N_JOBS - 1)


# **************** Loading the data
# For the example I am usign a table in csv format, but if your data are larger you can use other formats compatible with pandas, like .pickle
print("Start loading data file ")

cols_names=['area', 'perimeter', 'neighbours', 'max neighbour distance',
       'min neighbour distance', 'max vertices distance',
       'min vertices distance', 'max vertices-point distance',
       'min vertices-point distance', 'distance to center', 'activity',
       'particle type']

density=sys.argv[2]
fa=sys.argv[3]

data = pd.read_csv(f"phia{density}/particles-features-{density}-Fa{fa}.txt", delimiter=' ',names=cols_names)


# *** Split the available data in training and test
# the model will be trained only using the training data
# so we can evaluate the performance on a different (test) set, that is new for the ML model
df_train, df_test = train_test_split(data, random_state=50, test_size=0.3)

print('------------ BEGIN TRAIN DATAFRAME COLUMNS ------------------')
print(df_train.columns)
print('------------- END TRAIN DATAFRAME COLUMNS -------------------')




print("create x/y dataframes (train set)")
# In this particulat proble we want to identify if a specific particle 'is_active' or not
# so we have to do a binary classification of the column 'is_active'
df_train_y = df_train['activity'].copy().astype('int')
df_train.drop(columns='activity', inplace=True)
df_train_x = df_train

print("create x/y dataframes (test set)")
df_test_y = df_test['activity'].copy().astype('int')
df_test.drop(columns='activity', inplace=True)
df_test_x = df_test


# *********************************************************************************
# * The ML model it actually just two lines!
# I am providing you with two examples that you can select (In the paper we used LGBMClassifier)
model_flag=1
if model_flag==0:
    # notice that there are several model parameters (=hyperparameters) that can be tuned.
    # In particular you have:
    #   - hidden_layer_size: the number of neuron in each of the hidden layer.
    #                        In the example we are using 2 hidden layer where the first is composed by one neuron for input,
    #                        while the second is 50 neurons.
    #   - activation: the activation function to interpose between each neuron. 'relu' should be your standard choiche
    #   - solver: the algorithm for the training. I almost always use 'adam'
    # More details are available at: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    
    print('\n\n*** Using a Multi-layer-perceptron')
    # Definition of the model
    model = MLPClassifier(random_state=1, max_iter=100, hidden_layer_sizes=(len(df_train_x.columns),50), activation = 'relu',solver='adam',validation_fraction=0.2, early_stopping=True)
    
    model.fit(df_train_x,df_train_y)
elif model_flag==1:
    print('\n\n*** Using a gradient boosting algorithm')
        # This is a more advanced algorithm that is explained here:https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
    model = lgb.LGBMClassifier(learning_rate=0.01,num_leaves=100,max_depth=15, early_stopping_rounds=10, num_iterations=3000, random_state=42)
    
    model.fit(df_train_x,df_train_y,eval_set=[(df_train_x,df_train_y),(df_test_x,df_test_y)],
              eval_metric='binary')
# *********************************************************************************
 
print('\n\n***Training accuracy {:.4f}'.format(model.score(df_train_x,df_train_y)))
print('***Testing accuracy {:.4f}\n'.format(model.score(df_test_x,df_test_y)))

print(metrics.classification_report(df_test_y,model.predict(df_test_x)))

# ******************************************************************************
# ******************************************************************************
# To evaluate the model you can plot
#   (1) Feature importances: https://inria.github.io/scikit-learn-mooc/python_scripts/dev_features_importance.html
#                            it basically tells you which feature the model is using more for its predictions
#   (2) Train/Test score: it reports the accuracy of the training model in classifing the training and the test sets.
#                         If only the training score is improving, it means that the model is overfitting.
#                         This is bad beacuse then the model it will not be able to make predictions (it has only 'memorized' the training set)
#   (3) Confusion matrix: https://en.wikipedia.org/wiki/Confusion_matrix


# Plot feature importance

lgb.plot_importance(model, max_num_features=15, color='orange')
plt.savefig(f'phia{density}/importance-3D-voronoi-phi{density}-Fa{fa}.png', format='png', bbox_inches='tight', transparent=False)
plt.show(block=True)  

# Plot metric evolution

lgb.plot_metric(model)
plt.show(block=True)  