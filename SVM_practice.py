from cProfile import label
from math import gamma
from turtle import color
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/home/siddhant/Downloads/cell_samples.csv")
#print(df.tail())             #prints last 5 rows of data to check if reading is happening right or not
#print(df.shape)              #prints the shape of the data, i.e. number of (rows, columns)
#print(df.size)               #prints the file size on computer (in bytes)
#print(df.count())            #prints no of values in each column (699)
sizes = df['Class'].value_counts()
#print(sizes)                 #value split between 2 and 4 (benign and malignant cancers resp.)

#specifying benign and malignant data frames(df) as a subset of out main df(data frame) with respective Class values
benign_df = df[df['Class']==2][0:200] #will take values till 200 from the data
malignant_df = df[df['Class']==4][0:200]

#now plotting
#plotting a scatter plot for any two columns (  help(benign_df.plot) will tell about all the possible plottings   )
axes = benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='benign') 
malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='malignant', ax = axes)
#axes will specify indivicual plot area, ax=axes will plot both on same graph
#plt.show()                   #shows the graph

#selecting unwanted columns
#print(df.dtypes)      #shows that column BareNuc has the datatype "object" , not "int64" and thus mathematical operations can't be performed on it, we need to convert

#now to convert those object values to integer
df = df[pd.to_numeric(df['BareNuc'],errors='coerce').notnull()]              #this will form respective integer values from object form
df['BareNuc'] = df['BareNuc'].astype('int')                                  #this will replace the values with integer form
#print(df.dtypes)

#data cleaning is done
#time for data sorting and removing unwanted columns

df.drop(['ID'], axis=1, inplace=True)
#print(df.dtypes)

#specifying x and y values
Y = df["Class"].values
X = df.drop(labels=["Class"], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 20)

#Defining the model and training.
from sklearn import svm                                                       #specifically we will use SVC, support vector classifier
classifier = svm.SVC(kernel='linear', gamma='auto', C=2) 
#kernel's func is to view the data from a different perspective to make a suitable hyperplane to differentiate data
#gamma is a kernel coeff hyperparameter that specifies till how far data points to be considered
# C poses penalty on incorrectly placed data points (here, 2 units of penalty for such data points)
classifier.fit(X_train,Y_train)
Y_predict = classifier.predict(X_test)

from sklearn import metrics                                                   #it compares test and prdicted values and gives an accuracy rate
print("Accuracy = ", metrics.accuracy_score(Y_test, Y_predict))

#now if we want all precision categories as well, i.e. classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_predict))



