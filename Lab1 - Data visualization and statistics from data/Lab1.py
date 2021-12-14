
'''
    Name: Prashant Kumar
    Roll No.: B19101
    Mobile Number: 8700350173

'''


from matplotlib import pyplot as plt # importing all necessary Libraries 
import pandas as pd
dataframe = pd.read_csv(r"landslide_data3.csv")    # Reading landslide_data3.csv file using panda

# defining all function for calculating mean, median, mode, max, min, standard deviation using pandas library 
def Mean(s): 
    return (dataframe[s].mean(axis=0))   
def Median(s):
    return (dataframe[s].median(axis=0))
def Maxi(s):
    return (dataframe[s].max(axis=0))
def Mini(s):
    return (dataframe[s].min(axis=0))
def Std(s):
    return (dataframe[s].std(axis=0))
def Mode(s):
    return (dataframe[s].mode()[0])

#-------------------------QUESTION 1-----------------------------------#

def Q1 ():
    attributes=['temperature','humidity','pressure','rain','lightavgw/o0','lightmax','moisture']    
    MeanList=[];MedianList=[];ModeList=[];MaxList=[];MinList=[];StdList=[]
    for i in attributes: # using loop to call all function for each attributes
        MeanList.append(Mean(i))
        MedianList.append(Median(i))
        ModeList.append(Mode(i))
        MaxList.append(Maxi(i))
        MinList.append(Mini(i))
        StdList.append(Std(i))
    data1 = {"Attributes":attributes, # formaing data in tabular form
            "Mean": MeanList,
            "Median": MedianList,
            "Mode": ModeList}
    data2 = {"Attributes":attributes, # formaing data in tabular form
            "Max": MaxList,
            "Min": MinList,
            "Std":StdList}
    ans = pd.DataFrame(data1)
    ans = ans.set_index(['Attributes'])# setting attributes as an index
    print(ans)
    ans = pd.DataFrame(data2)
    ans = ans.set_index(['Attributes'])# setting attributes as an index
    print(ans)

#--------------------------QUESTION 2----------------------------------#


def Scatter_plot (s,xAttribute): # defining function to plot scatter graph of particular attribute
	fig,axs=plt.subplots(2,3,figsize=(15,9))
	colors = ['red','gold','green','blue','lightcoral','lightskyblue']
	count=0
	for r in range(2):
	    for i in range(3):
	        dataframe.plot.scatter(x=xAttribute,y=s[count],ax=axs[r,i],color=colors[count],grid=True)# using scatter function ploting scatter plot 
	        count+=1
	        fig.tight_layout(pad=3.0)
	plt.show() 
	

def Q2 ():

	#-----------------2a-----------------------#
    print()
    print('2(a).  Scatter Plot btw Rain and other attribute')
    attrForRain = ['temperature','humidity','pressure','lightavgw/o0','lightmax','moisture']# list of attribute to plot with rain
    Scatter_plot(attrForRain,'rain') # calling function to plot scatter graph 

	#-----------------2b-----------------------#
    print()
    print('2(b).  Scatter Plot btw Temperature and other attribute')
    attrForTemp = ['humidity','pressure','rain','lightavgw/o0','lightmax','moisture'] # list of attribute to plot with temperature
    Scatter_plot(attrForTemp,'temperature') # calling function to plot scatter graph 

#--------------------------QUESTION 3---------------------------------#

def Correlation (s,corrWith): # defining function to calculate correlation 
	Corr=[]
	for i in s:
	    Corr.append(dataframe[corrWith].corr(dataframe[i]))    # using corr function to calculate correlation coefficient between two attributes
	data = {"Attribute":s,"Correlation with "+corrWith:Corr}
	ans = pd.DataFrame(data)
	ans = ans.set_index(['Attribute'])# setting attributes as an index
	print(ans)

def Q3 ():

	#-----------------3a------------------#
    print()
    print('3(a).  Correlation btw Rain and other attribute')
    corrAttrWithRain=['temperature','humidity','pressure','lightavgw/o0','lightmax','moisture'] # list of attribute to calculate correlation with rain
    Correlation(corrAttrWithRain,'rain') # calling function to calculate correlation

	#-----------------3b------------------#
    print()
    print('3(b).  Correlation btw Temperature and other attribute')
    corrAttrWithTemp=['humidity','pressure','rain','lightavgw/o0','lightmax','moisture']# list of attribute to calculate correlation with temperature
    Correlation(corrAttrWithTemp,'temperature') # calling function to calculate correlation


#--------------------------QUESTION 4------------------------------#

def Q4 ():
	fig,axs=plt.subplots(1,2,figsize=(10,5)) 
	dataframe['rain'].plot.hist(ax=axs[0],color='red',grid=True,title="HISTOGRAM OF RAIN") # ploting histogram of rain using hist function
	dataframe['moisture'].plot.hist(ax=axs[1],color='blue',grid=True,title="HISTOGRAM OF MOISTURE") # ploting histogram of temperature using hist function
	plt.show()

#-------------------------QUESTION 5 -------------------------------#

def Q5 ():
	gkk = dataframe.groupby('stationid') # here i am using group by function for grouping the data
	fig,axs=plt.subplots(2,5,figsize=(15,7)) 
	count=0
	colors = ['red','blue','green','black','lightcoral','lightskyblue','gold','#ac1bc1','#d11554','#16c918']
	r = 0; c= 0
	for s,j in gkk: # using for loop ploting histogram of rain for all 10 station  
	    j['rain'].plot.hist(ax=axs[r,c],color=colors[count],title=("HISTOGRAM OF RAIN FOR "+s.upper()),grid=True)
	    count+=1
	    c+=1
	    if c==5 :
	        c=0
	        r+=1
	    fig.tight_layout(pad=2.0)
	plt.show()

#----------------------------QUESTION 6-------------------------------#

def Q6 ():
    print("BOXPLOT OF RAIN AND MOISTURE")
    fig,axs=plt.subplots(1,2,figsize=(5,10)) 
    dataframe.boxplot(column=['rain'],ax=axs[0]) # ploting boxplot of rain using boxplot function
    dataframe.boxplot(column=['moisture'],ax=axs[1])# ploting boxplot of moisture using boxplot function
    fig.tight_layout(pad=2.0)
    plt.show()

print("-----------------------------------SOLUTION 1-------------------------------------")
Q1()
print()
print("-----------------------------------SOLUTION 2-------------------------------------")
Q2()
print()
print("-----------------------------------SOLUTION 3-------------------------------------")
Q3()
print()
print("-----------------------------------SOLUTION 4-------------------------------------")
Q4()
print()
print("-----------------------------------SOLUTION 5-------------------------------------")
Q5()
print()
print("-----------------------------------SOLUTION 6-------------------------------------")
Q6()