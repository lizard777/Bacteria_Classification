

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

#must split the data to test and train data 
train = pd.read_csv('train_dataset.csv',sep = ',')
print(train.head())

test = pd.read_csv('test_dataset.csv',sep = ",")
print(test.head())

#initial_explatory(data_set)

#find unique falues of the train data 

find_uniques = train.nunique().sort_values(ascending = False)
print(find_uniques)
print(train.info())

#dropping row id from the data set as we are only interested in the expressed sequences 

train = train.drop(['row_id'],axis = 1)
print(train.head())

test = test.drop(['row_id'], axis = 1)
print(test.head())

#reducing the memory usage 
#def reduce_mem_usage(props):
 #   start_mem_usg = props.memory_usage()


#for machine learning practice
y = train.pop('target')
x = train
#test size is 20%
#random_state
#shuffle
#statify
x_train, x_test,y_train,y_test = train_test_split( x, y, test_size=0.2, random_state=42,shuffle=True, stratify=y)

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



