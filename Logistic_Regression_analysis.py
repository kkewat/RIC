import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics

matplotlib.style.use('ggplot')
plt.figure(figsize=(9,9))

#definr sigmoidal function\
def sigmoid(t):
    return(1/(1+np.e**(-t)))

plot_range = np.arange(-6,6,0.1)

y_values = sigmoid(plot_range)

#Plot curve
plt.plot(plot_range,y_values,color="red")
titanic_train = pd.read_csv('D://New folder//titanic_train.csv')
char_cabin = titanic_train["cabin"].astype(str)  #Convert Cabin to str
new_Cabin = np.array([cabin[0] for cabin in char_cabin])

titanic_train["cabin"] = pd.Categorical(new_Cabin)

new_age_var = np.where(titanic_train["age"].isnull(),28,titanic_train["age"])

titanic_train["age"] = new_age_var

label_encoder = preprocessing.LabelEncoder()

encoded_sex = label_encoder.fit_transform(titanic_train["sex"])

log_model = linear_model.LogisticRegression()

#Train the model
log_model.fit(X=pd.DataFrame(encoded_sex),y=titanic_train["survived"])

print(log_model.intercept_)

print(log_model.coef_)

#Make Predictions
preds = log_model.predict_proba(X=pd.DataFrame(encoded_sex))
preds = pd.DataFrame(preds)
preds.columns=["Death_prob","Survival_prob"]

pd.crosstab(titanic_train["sex"],preds.loc[:,"Survival_prob"])

encoded_class = label_encoder.fit_transform(titanic_train["pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_train["cabin"])

train_features = pd.DataFrame([encoded_class,encoded_cabin,encoded_sex,titanic_train["age"]]).T

#Initializing titanic regression midel
log_model = linear_model.LogisticRegression()

#Train the model
log_model.fit(X=train_features,y=titanic_train["survived"])
#Checke ntrained model
print(log_model.intercept_)

print(log_model.coef_)
preds = log_model.predict(X=train_features)

pd.crosstab(preds,titanic_train["survived"])

log_model.score(X=train_features,y=titanic_train["survived"])

metrics.confusion_matrix(y_true=titanic_train["survived"],y_pred=preds)

print(metrics.classification_report(y_true=titanic_train["survived"],y_pred=preds))

titanic_test = pd.read_csv('D:/New folder/titanic_train.csv')

char_cabin = titanic_test["cabin"].astype(str)
new_Cabin = np.array([0] for cabin in char_cabin)
titanic_test['cabin'] = pd.Categorical(new_Cabin)

new_age_var = np.where(titanic_test["age"].isnull(),28,titanic_test["age"])
titanic_test["age"] = new_age_var

encoded_sex = label_encoder.fit_transform(titanic_test["sex"])
encoded_class = label_encoder.fit_transform(titanic_test["pclass"])
encoded_cabin = label_encoder.fit_transform(titanic_test["cabin"])

test_features = pd.DataFrame([encoded_class,encoded_cabin,encoded_sex,titanic_test["age"]]).T

test_preds = log_model,predict(X=test_features)

submission = pd.DataFrame({"PassengerId":titanic_test["PassengerId"],"Survived":test_preds})

submission.to_csv('D:/New folder/tutorial_logreg_submission.csv',index=False)
