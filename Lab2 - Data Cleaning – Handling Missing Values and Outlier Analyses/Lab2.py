
'''
    Name: Prashant Kumar
    Roll No.: B19101
    Mobile Number: 8700350173

'''
import pandas as pd # importing all necessary Libraries 
from matplotlib import pyplot as plt 
import numpy as np
miss_df = pd.read_csv(r"pima_indians_diabetes_miss.csv") # Reading pima_indians_diabetes_miss.csv file using panda
original_df = pd.read_csv(r"pima_indians_diabetes_original.csv") # Reading pima_indians_diabetes_original.csv file using panda
attributes = list(miss_df.head()) # grabing attributes
def Graph(freq,title,xlabel,ylabel): # defining funtion to plot graph
    y = np.arange(len(attributes))
    plt.bar(y,freq,align='center',width=0.12,color='red',alpha=0.8)
    plt.scatter(y,freq,color='blue',alpha=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(y,attributes)
    plt.show()
    
def Q1(): # function that count null values of each attribute and plot the graph using Graph() function
	CountNull = miss_df.isnull().sum();Graph(CountNull,"SOLUTION OF Q1",'Attribute','Frequency of Missing Values')

def Q2():
    print("\n_______2(a)._______\n")
    print("Delete (drop) the tuples (rows) having equal to or more than one third of attributes with missing values")
    dropRow=[] # list contains rows that are to drop 
    for i in miss_df.index: # using loop to find the rows that contain null values greater than 3
        if miss_df.loc[i].isnull().sum()>=3 :
            dropRow.append(i)
    print("* Total rows deleted:",len(dropRow)) # printing number of rows to delete 
    print("* Row number of deleted tuples:",*dropRow)
    print("\n_______2(b)._______\n")
    print("Drop the tuples (rows) having missing value in the target (class) attribute")
    miss_df.drop(dropRow,inplace=True) # droping tuples of specific rows
    row_emptyClass=miss_df[miss_df['class'].isnull()].index.tolist() # making list that contain null value of attribute class 
    miss_df.drop(row_emptyClass,inplace=True)  # drping tuples on the basis of rows that has empty value in class attribute
    print("* Total rows deleted:",len(row_emptyClass))   
    print("* Row number of deleted tuple:",*row_emptyClass)

def Q3():
    print("\n* The number of missing values in each attributes")
    nullvalue= miss_df.isnull().sum() # getting total null values of each attribute
    data = {"Attribute":attributes,"Total null values":nullvalue}
    data = pd.DataFrame(data) # forming data in tabular form
    data= data.set_index(["Attribute"])
    print(data)
    print("\n* Total Missing Value in file:",sum(nullvalue))

l=[] # list that contain list of rows of each attribute that has missing values 
def nullRows(s): #using this funtion we find the list of row of an attribute that has empty value
    a=(miss_df[miss_df[s].isnull()].index.tolist())
    l.append(a)

def DataFrame(t): # function for calculation mean, median, mode , standard deviation of each attribute of the data
    data = {"Attributes":attributes,"mean":t.mean().tolist(),
                "median":t.median().tolist(),
                "mode":t.mode().iloc[0].tolist(),
                "std":t.std().tolist()}
    dataFrame = pd.DataFrame(data) # forming data in tabular form
    dataFrame = dataFrame.set_index(["Attributes"])
    print(dataFrame)

def RMSE(attribute,nullRows,data): # function to calculate RMSE of an attribute
    rms=0
    for i in nullRows:
        rms += (data[attribute].loc[i]-original_df[attribute].loc[i])**2  # using formula of RMSE
    if rms == 0:
        return rms
    return (rms/len(nullRows))**0.5

def showRMSE(t): # funtion to show the RMSE of each attribute in a tabular form using dataframe
    rmselist=[]
    for i in range(len(attributes)):
        rmselist.append(RMSE(attributes[i],l[i],t))
    data = {"Attribute":attributes,"RMSE":rmselist}
    data = pd.DataFrame(data) # forming data in a tabular form
    data = data.set_index(["Attribute"])
    print(data)
    Graph(rmselist,'','Attribute','RMSE') # using Graph function to draw the graph between RMS and attribute
    
def Q4(): 
    print("\n_______4(a)._______")
    print("Null Value of Data filled using mean\n\n(i).\n")
    print("* Missing Data: Mean, median, mode and standard deviation ")
    fix_using_mean = miss_df.fillna(round(miss_df.mean())) # filling null value of data using mean of each attribute        
    DataFrame(fix_using_mean) # using DataFrame Function to show mean,median,mode etc. of missing data
    print("\n* Original Data: Mean, median, mode and standard deviation ")
    DataFrame(original_df) # using DataFrame Function to show mean,median,mode etc. of original data
    print("\n(ii).\n\n* Calculated RMSE of each attribute")
    showRMSE(fix_using_mean)  # using showRMSE function to show RMSE of data
    print()
    print("_______4(b)._______")
    print("Null Value of Data filled using linear interpolation\n\n(i).\n")
    print("* Missing Data: Mean, median, mode and standard deviation")
    fix_using_interpolate = miss_df.fillna(round(miss_df.interpolate())) # filling null value of data using interpolation in each attribute 
    DataFrame(fix_using_interpolate) # using DataFrame Function to show mean,median,mode etc. of missing data
    print("\n*Original Data: Mean, median, mode and standard deviation ")
    DataFrame(original_df) # using DataFrame Function to show mean,median,mode etc. of original data
    print("(ii).")
    print("\n(ii).\n\n* Calculated RMSE of each attribute")
    showRMSE(fix_using_interpolate) # using showRMSE function to show RMSE of data

def outliner(data,attr): # funtio to calculate the outliner of an attribute 
    minimum = 2.5 * np.percentile(data[attr], 25) - 1.5 * np.percentile(data[attr], 75)
    maximum = 2.5 * np.percentile(data[attr], 75) - 1.5 * np.percentile(data[attr], 25)
    return pd.concat((data[attr][data[attr] < minimum], data[attr][data[attr] > maximum])) # returning data in a tabular form that conatin only outliner  

def boxplot(data): # function to ploting boxploit of Age and BMI boxplot 
    fig,axs=plt.subplots(1,2,figsize=(5,6)) 
    data.boxplot(column=['Age'],ax=axs[0])
    data.boxplot(column=['BMI'],ax=axs[1])
    fig.tight_layout(pad=2.0)
    plt.show()

def Q5():
    print("_______5(a)._______")
    print("Replacing the missing values by interpolation method")
    print("* Box Plot of attribute 'Age' and 'BMI'")
    #a = miss_df.fillna(round(miss_df.interpolate())) # filling null value of data using interpolation in each attribute
    miss_df.fillna(round(miss_df.interpolate()),inplace=True)
    print(f"(i). Outliner in Age: {outliner(miss_df,'Age').values}") # printing outliner of Age attribute using outlier function
    print(f"(ii). Outliner in BMI: {outliner(miss_df,'BMI').values}") # printing outliner of BMI attribute using outlier function
     # printing outliner of BMI attribute using outlier function
    
    #a.drop(outliner(miss_df,'Age').index,inplace=True)
    #print(len(a['Age']),len(miss_df['Age']))
    #print(f"(ii). Outliner in age: {outliner(a,'Age')}")
    #boxplot(miss_df)
    #miss_df['Age'][outliner(miss_df,"Age").index] = a['Age'].median()
     # ploting boxplot
    boxplot(miss_df)
    print("_______5(b)._______")
    print("Replacing the outliners using median of data")
    print("* Box Plot of attribute 'Age' and 'BMI'")
    miss_df['Age'][outliner(miss_df,"Age").index] = miss_df['Age'].median() # replacing the outliner using median of the attribute
    miss_df['BMI'][outliner(miss_df,"BMI").index] = miss_df['BMI'].median()
    print(f"(i). Outliner in Age: {outliner(miss_df,'Age').values}") # printing outliner of Age attribute using outlier function
    print(f"(ii). Outliner in BMI: {outliner(miss_df,'BMI').values}")
    boxplot(miss_df) # ploting boxplot

print("===========================SOLUTION OF QUESTION 1===========================")
Q1()
print("\n==========================SOLUTION OF QUESTION 2============================")
Q2()
print("\n==========================SOLUTION OF QUESTION 3============================")
Q3()  
for i in attributes:
    nullRows(i)
print("\n==========================SOLUTION OF QUESTION 4============================")
Q4()
print("\n==========================SOLUTION OF QUESTION 5============================")
Q5()