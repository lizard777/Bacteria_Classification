

import pandas as pd
import matplotlib.pyplot as plt

#for machine learning practice
"""""
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classificiation_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svs import SVC
"""

import seaborn as sb


#express genes as heatmap to see patterns if any 
def initial_explatory(x): 
    print(x.head())

    x = x.nunique().sort_values(ascending=False)
    print(x)


"""

    plt.figure(figsize=(5,15))
    
    sb.heatmap(x,linewidths = 0.5, linecolor = 'black', square = True, cmap = 'RdBu')
    plt.xlabel('bacterial species', size = 15)
    plt.ylabel('gene location', size = 15)
    plt.title('gene vs. bacteria species', size = 15)
    plt.xticks(rotation = 25)
    plt.yticks(rotation = 0)
    plt.savefig('test.png')
    plt.show()
"""








data_set = pd.read_csv('test_dataset.csv',sep = ",")
initial_explatory(data_set)

