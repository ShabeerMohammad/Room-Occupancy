
#import the required packages
import pandas as pd
from pandas import concat
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

import pickle

# load all data
data1 = pd.read_csv('datatest.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data2 = pd.read_csv('datatraining.txt', header=0, index_col=1, parse_dates=True, squeeze=True)
data3 = pd.read_csv('datatest2.txt', header=0, index_col=1, parse_dates=True, squeeze=True)

#let's concatenate the data
data = concat([data1,data2,data3])

data.head(2)
data.info()
data.describe
data.dtypes
#drop row number
data.drop('date',axis=1,inplace=True)

#save aggregated dataset
data.to_csv('combined.csv')

data = pd.read_csv('combined.csv',
                   header=0,
                   index_col=0,
                   parse_dates=True,
                   squeeze=True)

values  = data.values

#split the data into inputs and outputs
X,y = values[:,:-1],values[:,-1]

#split the dataset
trainX,testX,trainy,testy = train_test_split(X,y,test_size=0.3,shuffle=False,random_state=1)
  
# define the model
model = LogisticRegression()

# fit the model on the training set
model.fit(trainX, trainy)

# predict the test set
yhat = model.predict(testX)
# evaluate model skill
score = accuracy_score(testy, yhat)
print(score)    

#save the model
filename = 'logistic-regression.pkl'

pickle.dump(model,open(filename,'wb'))
