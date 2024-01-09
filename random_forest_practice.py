import pandas as pd
from matplotlib import pyplot as ptl
import numpy as np

df = pd.read_csv("/home/siddhant/Downloads/123.csv")
#print(df.head()) #prints first 5 rows of data to check if reading is happening right or not

#now to see the split between good and bad values of productivity
sizes = df['Productivity'].value_counts()
#print(sizes)


#now dropping irrelevent data
df.drop(['Images_Analyzed'], axis=1, inplace=True) #axis value 1 says it to drop column adn 0 says to drop row, inplace tells it to modify the original data frame
df.drop(['User'], axis=1, inplace=True)
#print(df.head())

#Handle missing values, if needed
#df = df.dropna()  #Drops all rows with at least one null value.

#Convert non-numeric to numeric

df.Productivity[df.Productivity == 'Good'] = 1
df.Productivity[df.Productivity == 'Bad'] = 0
#print(df.head())

#Y is the dependent data, what we need to predict i.e. Productivity column
Y = df['Productivity'].values
#but Y is not in integer form, so converting it
Y = Y.astype('int')

#X in the independent data columns, i.e rest 3 columns other that Productivity
X = df.drop(labels=["Productivity"], axis=1)
#print(X.head())

#spliting data into train and test datas
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.30, random_state = 20)
#random_state can be any integer and it is used as a seed to randomly split dataset.
#By doing this we work with same test dataset evry time, if this is important.
#random_state=None splits dataset randomly every time

#Defining the model and training.
#RandomForestRegressor is for regression type of problems. 
#For classification we use RandomForestClassifier.
#Both yield similar results except for regressor the result is float
#and for classifier it is an integer. 
#Let us use classifier since this is a classification problem

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, random_state = 30)

# Train the model on training data
model.fit(X_train, Y_train)

#TESTING THE MODEL BY PREDICTING ON TEST DATA
#AND CALCULATE THE ACCURACY SCORE

prediction_test = model.predict(X_test)
#print(Y_test, prediction_test) #to check the values of test and predicted values

from sklearn import metrics #it compares test and prdicted values and gives an accuracy rate
print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

#we can even find out importance of different fields/columns in prediction in RandomForest             #extra
feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
#print(feature_imp)

