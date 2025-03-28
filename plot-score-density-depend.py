import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score
import sys
import joblib

#Returns accuracy,auc,f1_score
def score(y,pred):
    return accuracy_score(y,pred),roc_auc_score(y,pred),f1_score(y,pred)


def plot_score_density_depend(fa,density_list):
    accuracy_list=[]
    auc_list=[]
    f1_list=[]

    for density in density_list:
        model,features,target,df_test_y,df_test_x=joblib.load(f"phia{density}/gb-model-{density}-Fa{fa}-balanced.pkl")   
        pred=model.predict(df_test_x)
        accuracy,auc,f1=score(df_test_y,pred)
        accuracy_list.append(accuracy)
        auc_list.append(auc)
        f1_list.append(f1)
    
    plt.scatter(density_list,accuracy_list)
    plt.scatter(density_list,auc_list)
    plt.scatter(density_list,f1_list)
    plt.legend(['Accuracy','ROC AUC score', 'F1 score'])
    plt.show()
    output_file='scorephiadependent-balanced.txt'
    df=pd.DataFrame(columns=[density_list,accuracy_list,auc_list,f1_list])
    df.to_csv(output_file, sep=" ", index=False)

    return 1
    
fa=sys.argv[1]

#Include your density_list
density_list=[0.008,0.1,0.2]
plot_score_density_depend(fa,density_list)