'''
    Name: Prashant Kumar
    Roll No.: B19101
    Mobile Number: 8700350173

'''
#----------------------------------Import Libraries---------------------------------#
import pandas as pd # importing all necessary Libraries 
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


#----------------------------------Reading file-------------------------------------#

df = pd.read_csv(r"seismic_bumps1.csv")   # Reading seismic_bumps1.csv file using panda
attributes = list(df.head(0)) # list of all attributes
df.drop(attributes[8:16],axis=1,inplace=True) # droping unnecessary attributes
attr = list(df.head(0)) # list of all attributes

#---------------------------------Spliting Data or Data Preprocessing-------------------------------------#

gkk = df.groupby('class') # grouping on the basis of class
[X_train ,X_test, X_label_train, X_label_test] = [pd.DataFrame() # initialising all variables with empty dataframe
                    ,pd.DataFrame(),pd.DataFrame(),pd.DataFrame()] 

for i,j in gkk:
    [a_train, b_test, a_label_train, b_label_test]=train_test_split(j.copy(),
        j['class'], test_size=0.3, random_state=42,shuffle=True) # spliting data 70% train and 30% test of each class
    
    [X_train, X_test, X_label_train, X_label_test] = [pd.concat([X_train,a_train]), # concatinate the data of each class
        pd.concat([X_test,b_test]),pd.concat([X_label_train,a_label_train]),
        pd.concat([X_label_test,b_label_test])]
 
X_test = X_test.drop('class',axis=1)  # test data contain class attribute so drop it
X_train.to_csv('seismic-bumps-train.csv',index=False) # saving train data into csv file
X_test.to_csv('seismic-bumps-test.csv',index=False) # savind test data into csv file

#-----------------------------------Defining all necessary functions------------------------#

def K_NN(k,train_data,test_data): # define K-NN function 
    model = KNeighborsClassifier(n_neighbors=k) # using K-NN classifier for particular k
    model.fit(train_data,X_label_train.values.ravel()) # fitting train data 
    predicted= model.predict(test_data) # prdicting class
    conf_mat = confusion_matrix(X_label_test,predicted)  # forming confusion matrix
    accuracy = accuracy_score(X_label_test,predicted) # finding accuracy using accuracy_score function
    return accuracy, conf_mat # returning accuracy and confusion matrix

def show_confMat_accu(train_data,test_data): # function to showing confusion matrix and accuracy
    k_high_accuracy = 1 # variable to find k which has max accuracy
    high_accuracy = 0 # variable to find high accuracy
    for k in range(1,6,2):
        print("\nK = ",k)
        accu, conf_mat = K_NN(k,train_data,test_data) # calling K-NN function
        print("Confusion Matrix:")
        print(conf_mat) # printing confusion_matrix
        print("Accuracy: ",accu) # printing accuracy
        if accu > high_accuracy: # finding max accuracy
            high_accuracy = accu
            k_high_accuracy = k
    print(f"\nHigh Accuracy is for k = {k_high_accuracy} is {high_accuracy}")
    return high_accuracy # returning high accuracy

def min_max_scaling(df,minValue,maxValue,train_data): # function to normalized data
    df_norm = df.copy()# copy the dataframe
    attr = list(df_norm.head(0)) # list of attributes
    for column in attr:  # normalizing data using min-max scaling
        min_df = train_data[column].min() # minimum value of train data
        max_df = train_data[column].max() # maximum value of test data
        df_norm[column] = (((df_norm[column] - min_df) / (max_df - min_df))*(maxValue-minValue))+minValue    
    return df_norm # returning normalized data

def likelihood(sample,mean,covariance): # defining likelihood function
    # calculating exponential factor
    expo_part = math.exp((-1/2)*(np.dot(np.dot((sample-mean),np.linalg.inv(covariance)),(sample-mean).transpose())))
    # calculating pre-exponential factor
    non_exp_part = 1/(((2*math.pi)**(len(mean)/2))*(np.linalg.det(covariance)**0.5))
    return non_exp_part*expo_part # reyurning likelihood probability

print("\n-----------------------------------QUSETION 1-------------------------------------")
print("K-nearest neighbour (K-NN) classifier")
q1_accu = show_confMat_accu(X_train.iloc[:,:-1],X_test)  # calling function


print("\n-----------------------------------QUSETION 2-------------------------------------")
print("K-nearest neighbour (K-NN) classifier on normalized data")
normal_train_data = min_max_scaling(X_train.iloc[:,:-1],0,1,X_train.iloc[:,:-1]) # normalizing train data
normal_train_data.to_csv('seismic-bumps-train-Normalized.csv',index=False) # saving csv file of train data
normal_test_data = min_max_scaling(X_test,0,1,X_train.iloc[:,:-1]) # normalizing test data
normal_test_data.to_csv('seismic-bumps-test-Normalized.csv',index=False) # saving csv file of test data
q2_accu = show_confMat_accu(normal_train_data,normal_test_data) # calling function

print("\n-----------------------------------QUSETION 3-------------------------------------")
print("Bayes classifier:")
gkk = X_train.groupby('class') #grouping data on the basis of class
mean = [] # list contaning mean vector of each class 
covariance = [] # list contain covariance matrix of each class
cls_row = [] # list containg length of each class
for i,j in gkk:
    mean.append(np.array(j.iloc[:,:-1].mean().tolist())) # finding mean vector of the data
    covariance.append(np.cov(j.iloc[:,:-1].transpose())) # finding covariance matrix of data
    cls_row.append(len(j.iloc[:,1])) # finding number of rows in data
predict = [] # list contain prediction of test data
for i in X_test.index: # looping over test data
    sample = (list(X_test.loc[i]))  # list of tuple
    # calculating  Evedience P(x)  
    den = (likelihood(sample,mean[0],covariance[0]))*(cls_row[0]/sum(cls_row))+ (likelihood(sample,mean[1],covariance[1])*(cls_row[1]/sum(cls_row)))        
    # calculating P(C0/x) i.e posterior probability of class 0
    prob_cls0 = (likelihood(sample,mean[0],covariance[0])*(cls_row[0]/sum(cls_row)))/den
    # calculating P(C1/x) i.e posterior probability of class 1
    prob_cls1 = (likelihood(sample,mean[1],covariance[1])*(cls_row[1]/sum(cls_row)))/den
    if prob_cls0 > prob_cls1: # comparing probability and then assigning classs
        predict.append(0)
    else:
        predict.append(1)
conf_mat = confusion_matrix(X_label_test,predict) # finding confusion matrix
q3_accu = accuracy_score(X_label_test,predict) # finding accuracy of Bay's classifier
print("\nConfusion Matrix:\n",conf_mat)
print("Accuracy:",q3_accu)


print("\n-----------------------------------QUSETION 4-------------------------------------")
data = {"":['K-NN Classifier','K-NN on normalized data','Bayes Classifier'],
        "Accuracy":[q1_accu,q2_accu,q3_accu]} # forming dataframe of accuracy of each method
data = pd.DataFrame(data)
data = data.set_index([""])
print(data)

