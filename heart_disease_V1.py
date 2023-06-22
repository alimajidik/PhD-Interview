'''
dataset =>https://www.kaggle.com/datasets/venkatkarthick/heartcsv
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
age,sex,cp,trtbps,chol,fbs,restecg,thalachh,exng,oldpeak,slp,caa,thall,target
age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg - resting electrocardiographic results
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
target - have disease or not (1=yes, 0=no)

'''
# Importing all libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# 1). Imports and Reading Dataset
BASE_FOLDER = './heart_disease/'
dataset = pd.read_csv(f'{BASE_FOLDER}/heart.csv',delimiter=',',engine='python',encoding='latin')
print("\n #### Total number of rows and columns:")
print("(Rows, columns): " + str(dataset.shape))
 #print(dataset.columns)
print("\n #################")
#display the data
print('Sample instances from the dataset are given below')
print(dataset.head())
print("\n #################")
 #print(dataset.info())
#summarizes the count, mean, standard deviation, min, and max for numeric variables.
dataset.describe()
# age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
    #print(dataset.head(5))
# Display the Missing Values
print(dataset.isna().sum())
print("\n ######################")

countNoDisease = len(dataset[dataset.target == 0])
countHaveDisease = len(dataset[dataset.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(dataset.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(dataset.target))*100)))

'''
ercentage of Patients Haven't Heart Disease: 45.54%
Percentage of Patients Have Heart Disease: 54.46%
'''
print("\n")
sns.countplot(x='sex', data=dataset, palette="mako_r")
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

print("\n")
countFemale = len(dataset[dataset.sex == 0])
countMale = len(dataset[dataset.sex == 1])
print("\n Percentage of Female Patients: {:.2f}%".format((countFemale / (len(dataset.sex))*100)))
print("\n Percentage of Male Patients: {:.2f}%".format((countMale / (len(dataset.sex))*100)))
print("\n ##############\n")
'''
Percentage of Female Patients: 31.68%
Percentage of Male Patients: 68.32%
'''
print(dataset.groupby('target').mean())
print("\n ##############\n")

pd.crosstab(dataset.age,dataset.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

pd.crosstab(dataset.sex,dataset.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()

plt.scatter(x=dataset.age[dataset.target==1], y=dataset.thalach[(dataset.target==1)], c="red")
plt.scatter(x=dataset.age[dataset.target==0], y=dataset.thalach[(dataset.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()

pd.crosstab(dataset.slope,dataset.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()

pd.crosstab(dataset.fbs,dataset.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

pd.crosstab(dataset.cp,dataset.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

#Creating Model for Logistic Regression¶
'''
@ Splitting the data into train test

Next, I performed train-test-split and scaled our data. To review, 
we must train-test-split our data in order to evaluate our model and
 scaling our data normalizes the range of values in our independent variables.

For comparing and testing the viability of our model we need to split the heart.csv file
 into test and train data so that we can have a clear picture of how well our model
   is peforming and the overfitting stats. 
'''
# We split the data into training and testing set:
y = dataset["target"] # y is target
X = dataset.drop('target',axis=1) # X is data

#print (type(X))
#print("*****")
#Data Standardisation
# splitting the dataset into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state = 42)

print ('\n The total number of Training Data :', Y_train.shape)
print ('\n The total number of Test Data :', Y_test.shape)

#Naive Bayes Classifier: 
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
model = GaussianNB() # get instance of model
model.fit(X_train, Y_train) # Train/Fit model
#print('\n######### X test in GaussianNB model ##########\n')
#print(X_test)
y_predicted = model.predict(X_test) # get y predictions
#print('\n######### y predicted data in GaussianNB model ##########\n')
#print(y_predicted)
print('\n The accuracy of this GaussianNB model is:')
print(classification_report(Y_test, y_predicted)) #  output accuracy The accuracy of this model was 87%



# Get user input for data to predict
user_input = {}
for column in X.columns:
    value = input(f"Enter the value for {column}: ")
    user_input[column] = value

# Create a DataFrame for user input
user_data = pd.DataFrame(user_input, index=[0])

    #data = ([57,1,2,150,168,0,1,174,0,1.6,2,0,2] )
# Predict the output for user input
prediction = model.predict(user_data)
# Print the prediction
if prediction[0] == 0:
    print("The model predicts that the input does not indicate heart disease.")
else:
    print("The model predicts that the input indicates heart disease.")
print('\n!####################')
#print(" Prediction of previous attack probability status with user input is:",prediction )
#print('\n!####################')

# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
model2= MultinomialNB() # get instance of model
model2.fit(X_train,Y_train)# Train/Fit model
y_predicted2 = model2.predict(X_test)
#printing accuracy, Confusion matrix, Precision and Recall
from sklearn import metrics
print('\n The accuracy of the classifer with is MultinomialNB Model',metrics.accuracy_score(Y_test,y_predicted2))

# Fine-tune the model using ROC AUC score
from sklearn.metrics import roc_auc_score
y_pred_prob = model2.predict_proba(X)[:, 1]
auc_score = roc_auc_score(y, y_pred_prob)
print("ROC AUC score:", auc_score)


