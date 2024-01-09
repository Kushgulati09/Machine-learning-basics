import pandas as pd
import numpy as np

df = pd.read_csv('/home/siddhant/Downloads/spam.csv')

df.Category.replace({'ham':0, 'spam':1}, inplace= True)
#print(df.head())
null_count = df['Message'].isnull().sum()
#print(null_count)
#no null values
#all message content is in text, needs to be converted to integers 
#it will be done using countvectorizer() technique which takes into consideration of frequency of words used and converts into an array with number of frequencies of all the different words used

X= df['Message']
Y= df['Category']

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=5)

#conversion


from sklearn.feature_extraction.text import CountVectorizer
#v = CountVectorizer()
#X_train_count= v.fit_transform(X_train.values) 
#print(X_train_count.toarray()[:3])

from sklearn.naive_bayes import MultinomialNB
#model = MultinomialNB()
#model.fit(X_train_count, Y_train)

#X_test_count = v.transform(X_test)
#print(model.score(X_test_count, Y_test))



#but using all these count variables again and again can be exempted using pipeline functionality from sklearn.
#The pipeline will perform multiple steps in a single call and remove the need for intermediate variables like X_train_count and X_test_count.

from sklearn.pipeline import Pipeline
clf= Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, Y_train)
print(clf.score(X_test, Y_test))