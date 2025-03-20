import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
import sys
import joblib
import numpy as np

#Returns accuracy,auc,f1_score
def score(y,pred):
    return accuracy_score(y,pred),roc_auc_score(y,pred),f1_score(y,pred)


def plot_score_Fa_depend(fa_list,density):
    accuracy_list=[]
    auc_list=[]
    f1_list=[]

    for fa in fa_list:
        model,features,target,df_test_y,df_test_x=joblib.load(f"phia{density}/gb-model-{density}-Fa{fa}.pkl")   
        pred=model.predict(df_test_x)
        accuracy,auc,f1=score(df_test_y,pred)
        accuracy_list.append(accuracy)
        auc_list.append(auc)
        f1_list.append(f1)
    
    plt.scatter(fa_list,accuracy_list)
    plt.scatter(fa_list,auc_list)
    plt.scatter(fa_list,f1_list)
    plt.legend(['Accuracy','ROC AUC score', 'F1 score'])
    plt.xlabel('Fa value')
    plt.ylabel('Score')
    plt.show()
    output_file='scoreFadependent.txt'
    df=pd.DataFrame(columns=[fa_list,accuracy_list,auc_list,f1_list])
    df.to_csv(output_file, sep=" ", index=False)

    return 1
    
density=sys.argv[1]
fa_list=[100,85,75,60,50,40,20,15,10,5,0.5]
plot_score_Fa_depend(fa_list,density)

