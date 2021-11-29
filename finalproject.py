from __future__ import division
from PIL import ImageTk,Image 
import pandas as pd
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from sklearn.svm import SVC
import seaborn as sns
import numpy as np
import seaborn as sns
import math
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import os
import numpy as np
import pickle
import sys
import pyttsx3
from termcolor import colored
import datetime
import time 
from datetime import timedelta
import csv
from tkinter import *
from gtts import gTTS 
import math
from tkinter import *
from gtts import gTTS 

# create tkinter window 
root = Tk() 

# styling the frame which helps to 
# make our background stylish 
frame1 = Frame(root, 
			bg = "#1f0036", 
			height = "150") 

# plcae the widget in gui window 
frame1.pack(fill = X) 


frame2 = Frame(root, 
			bg = "#1f0036", 
			height = "750") 
frame2.pack(fill=X) 


# styling the label which show the text 
# in our tkinter window 
label = Label(frame1, text = "COPD Analysis System", 
			font = "bold, 30", 
			bg = "#f9f7cb") 

label.place(x = 180, y = 70) 



# entry is used to enter the text 
name_entry = Entry(frame2, width = 45, 
			bd = 4, font = 14) 

name_entry.place(x = 130, y = 52) 
name_entry.insert(0, "") 

engine = pyttsx3.init()

def pyttsx3(text):
	# obtain voice property
	voices = engine.getProperty('voices')
	# voice id 1 is for female and 0 for male
	engine.setProperty('voice', voices[0].id)
	# convert to audio and play
	engine.say(text)
	engine.runAndWait()

def submit(): 
  
    name=name_entry.get() 

    if name=="hi":
        pyttsx3("Welcome to COPD Analysis System! Please wait to get the results")


        play()
    else:
        exit(0)



def play():
    import matplotlib.pyplot as plt


    df=pd.read_csv('data.csv')
    df.columns = ['Diagnosis','Imaginary_min','Imaginary_avg','Real_min','Real_avg','Gender','Age','Smoking']
    df.head() #first 5 rows
    df[df['Diagnosis'] == 'COPD'].isnull().sum()
    df[df['Diagnosis'] == 'HC'].isnull().sum()
    df[df['Diagnosis'] == 'Infected'].isnull().sum()
    df[df['Diagnosis'] == 'Asthma'].isnull().sum()
    df.isnull().sum()
    df.duplicated()
    plt.figure(figsize=(14,14))
    sns.heatmap(df.corr(),annot=True)
    dfn=pd.notnull(df['Imaginary_avg'])
    dfnt=df[dfn]
    dfnt
    G0=df[df['Gender']==0].count().Gender
    G1=df[df['Gender']==1].count().Gender
    s1=df[df['Smoking']==1].count().Smoking
    s2=df[df['Smoking']==2].count().Smoking
    s3=df[df['Smoking']==3].count().Smoking
    df1=df[df['Diagnosis'] == 'COPD']
    y=df1.count().Diagnosis
    print(y)
    x=df.count().Diagnosis
    print(x)
    pcopd=y/x
    ng0c=df1[df1['Gender']==0].count().Gender
    ng1c=df1[df1['Gender']==1].count().Gender
    ns1c=df1[df1['Smoking']==1].count().Smoking
    ns2c=df1[df1['Smoking']==2].count().Smoking
    ns3c=df1[df1['Smoking']==3].count().Smoking
    pg=ng0c/G0
    print(pg)
    pg=ng1c/G1
    print(pg)
    ps=ns1c/s1
    print(ps)
    ps=ns2c/s2
    print(ps)
    ps=ns3c/s3
    print(ps)
    df2=df[df['Diagnosis'] == 'HC']
    y1=df2.count().Diagnosis
    print(y1)
    x1=df.count().Diagnosis
    print(x1)
    ng0h=df2[df2['Gender']==0].count().Gender
    ng1h=df2[df2['Gender']==1].count().Gender
    ns1h=df2[df2['Smoking']==1].count().Smoking
    ns2h=df2[df2['Smoking']==2].count().Smoking
    ns3h=df2[df2['Smoking']==3].count().Smoking
    pgh=ng0h/G0
    print(pgh)
    pgh=ng1h/G1
    print(pgh)
    psh=ns1h/s1
    print(psh)
    psh=ns2h/s2
    print(psh)
    psh=ns3h/s3
    print(psh)
    phc=y1/x1
    df2=df[df['Diagnosis'] == 'Asthma']
    y2=df2.count().Diagnosis
    print(y2)
    x2=df.count().Diagnosis
    print(x2)
    ng0a=df2[df2['Gender']==0].count().Gender
    ng1a=df2[df2['Gender']==1].count().Gender
    ns1a=df2[df2['Smoking']==1].count().Smoking
    ns2a=df2[df2['Smoking']==2].count().Smoking
    ns3a=df2[df2['Smoking']==3].count().Smoking
    pga=ng0a/G0
    print(pga)
    pga=ng1a/G1
    print(pga)
    psa=ns1a/s1
    print(psa)
    psa=ns2a/s2
    print(psa)
    psa=ns3a/s3
    print(psa)
    p_asthama=y2/x2
    df3=df[df['Diagnosis'] == 'Infected']
    y3=df3.count().Diagnosis
    print(y3)
    x3=df.count().Diagnosis
    print(x3)
    ng0i=df3[df3['Gender']==0].count().Gender
    ng1i=df3[df3['Gender']==1].count().Gender
    ns1i=df3[df3['Smoking']==1].count().Smoking
    ns2i=df3[df3['Smoking']==2].count().Smoking
    ns3i=df3[df3['Smoking']==3].count().Smoking
    pgi=ng0i/G0
    print(pgi)
    pgi=ng1i/G1
    print(pgi)
    psi=ns1i/s1
    print(psi)
    psi=ns2i/s2
    print(psi)
    psi=ns3i/s3
    print(psi)
    p_infected=y3/x3
    dfcopd=dfnt[dfnt['Diagnosis']=='COPD']
    mean_age=dfcopd.Age.mean()
    std_age=dfcopd.Age.std()
    mean_img_min=dfcopd.Imaginary_min.mean()
    std_img_min=dfcopd.Imaginary_min.std()
    mean_img_avg=dfcopd.Imaginary_avg.mean()
    std_img_avg=dfcopd.Imaginary_avg.std()
    mean_real_min=dfcopd.Real_min.mean()
    std_real_min=dfcopd.Real_min.std()
    mean_real_avg=dfcopd.Real_avg.mean()
    std_real_avg=dfcopd.Real_avg.std()
    dfhc=dfnt[dfnt['Diagnosis']=='HC']
    mean_age_hc=dfhc.Age.mean()
    std_age_hc=dfhc.Age.std()
    mean_img_min_hc=dfhc.Imaginary_min.mean()
    std_img_min_hc=dfhc.Imaginary_min.std()
    mean_img_avg_hc=dfhc.Imaginary_avg.mean()
    std_img_avg_hc=dfhc.Imaginary_avg.std()
    mean_real_min_hc=dfhc.Real_min.mean()
    std_real_min_hc=dfhc.Real_min.std()
    mean_real_avg_hc=dfhc.Real_avg.mean()
    std_real_avg_hc=dfhc.Real_avg.std()
    dfasthama=dfnt[dfnt['Diagnosis']=='Asthma']
    mean_age_asthama=dfasthama.Age.mean()
    std_age_asthama=dfasthama.Age.std()
    mean_img_min_asthama=dfasthama.Imaginary_min.mean()
    std_img_min_asthama=dfasthama.Imaginary_min.std()
    mean_img_avg_asthama=dfasthama.Imaginary_avg.mean()
    std_img_avg_asthama=dfasthama.Imaginary_avg.std()
    mean_real_min_asthama=dfasthama.Real_min.mean()
    std_real_min_asthama=dfasthama.Real_min.std()
    mean_real_avg_asthama=dfasthama.Real_avg.mean()
    std_real_avg_asthama=dfasthama.Real_avg.std()
    dfinfected=dfnt[dfnt['Diagnosis']=='Infected']
    mean_age_infected=dfinfected.Age.mean()
    std_age_infected=dfinfected.Age.std()
    mean_img_min_infected=dfinfected.Imaginary_min.mean()
    std_img_min_infected=dfinfected.Imaginary_min.std()
    mean_img_avg_infected=dfinfected.Imaginary_avg.mean()
    std_img_avg_infected=dfinfected.Imaginary_avg.std()
    mean_real_min_infected=dfinfected.Real_min.mean()
    std_real_min_infected=dfinfected.Real_min.std()
    mean_real_avg_infected=dfinfected.Real_avg.mean()
    std_real_avg_infected=dfinfected.Real_avg.std()
    def normpdf(x, mean, sd):
        var = float(sd)**2
        denom = (2*math.pi*var)**.5
        num = math.exp(-(float(x)-float(mean))*(float(x)-float(mean))/(2*var))
        return num/denom
    p=[]
    for i in range(len(dfcopd)):
        #print(dfcopd.iloc[i, 0], dfcopd.iloc[i, 5])
        p_img_min=normpdf(dfcopd.iloc[i,1],mean_img_min,std_img_min)
        p_img_avg=normpdf(dfcopd.iloc[i,2],mean_img_avg,std_img_avg)
        p_real_min=normpdf(dfcopd.iloc[i,3],mean_real_min,std_real_min)
        p_real_avg=normpdf(dfcopd.iloc[i,4],mean_real_avg,std_real_avg)
        if(dfcopd.iloc[i,5]==0):
            pg=ng0c/G0
        if(dfcopd.iloc[i,5]==1):
            pg=ng1c/G1
        p_age=normpdf(dfcopd.iloc[i,6],mean_age,std_age)
        if(dfcopd.iloc[i,7]==1):
            ps=ns1c/s1
        if(dfcopd.iloc[i,7]==2):
            ps=ns2c/s2
        if(dfcopd.iloc[i,7]==3):
            ps=ns3c/s3
        px=float(p_img_min*p_img_avg*p_real_min*p_real_avg*pg*p_age*ps*pcopd)
        #print(px)
        p.append(px)
    x=np.mean(p)
    print(x)
    p1=[]
    for i in range(len(dfhc)):
        p_img_min_hc=normpdf(dfhc.iloc[i,1],mean_img_min_hc,std_img_min_hc)
        p_img_avg_hc=normpdf(dfhc.iloc[i,2],mean_img_avg_hc,std_img_avg_hc)
        p_real_min_hc=normpdf(dfhc.iloc[i,3],mean_real_min_hc,std_real_min_hc)
        p_real_avg_hc=normpdf(dfhc.iloc[i,4],mean_real_avg_hc,std_real_avg_hc)
        if(dfhc.iloc[i,5]==0):
            pg_hc=ng0h/G0
        if(dfhc.iloc[i,5]==1):
            pg_hc=ng1h/G1
        p_age_hc=normpdf(dfhc.iloc[i,6],mean_age_hc,std_age_hc)
        if(dfhc.iloc[i,7]==1):
            ps_hc=ns1h/s1
        if(dfhc.iloc[i,7]==2):
            ps_hc=ns2h/s2
        if(dfhc.iloc[i,7]==3):
            ps_hc=ns3h/s3
        px_hc=float(p_img_min_hc*p_img_avg_hc*p_real_min_hc*p_real_avg_hc*pg_hc*p_age_hc*ps_hc*phc)
        #print(px_hc)
        p1.append(px_hc)
    x_hc=np.mean(p1)
    print(x_hc)
    p2=[]
    for i in range(len(dfasthama)):
        p_img_min_asthama=normpdf(dfasthama.iloc[i,1],mean_img_min_asthama,std_img_min_asthama)
        p_img_avg_asthama=normpdf(dfasthama.iloc[i,2],mean_img_avg_asthama,std_img_avg_asthama)
        p_real_min_asthama=normpdf(dfasthama.iloc[i,3],mean_real_min_asthama,std_real_min_asthama)
        p_real_avg_asthama=normpdf(dfasthama.iloc[i,4],mean_real_avg_asthama,std_real_avg_asthama)
        if(dfasthama.iloc[i,5]==0):
            pg_asthama=ng0a/G0
        if(dfasthama.iloc[i,5]==1):
            pg_asthama=ng1a/G1
        p_age_asthama=normpdf(dfasthama.iloc[i,6],mean_age_asthama,std_age_asthama)
        if(dfasthama.iloc[i,7]==1):
            ps_asthama=ns1a/s1
        if(dfasthama.iloc[i,7]==2):
            ps_asthama=ns2a/s2
        if(dfasthama.iloc[i,7]==3):
            ps_asthama=ns3a/s3
        px_asthama=float(p_img_min_asthama*p_img_avg_asthama*p_real_min_asthama*p_real_avg_asthama*pg_asthama*p_age_asthama*ps_asthama*p_asthama)
        p2.append(px_asthama)
    x_asthama=np.mean(p2)
    print(x_asthama)
    
    p3=[]
    for i in range(len(dfinfected)):
        p_img_min_infected=normpdf(dfinfected.iloc[i,1],mean_img_min_infected,std_img_min_infected)
        p_img_avg_infected=normpdf(dfinfected.iloc[i,2],mean_img_avg_infected,std_img_avg_infected)
        p_real_min_infected=normpdf(dfinfected.iloc[i,3],mean_real_min_infected,std_real_min_infected)
        p_real_avg_infected=normpdf(dfinfected.iloc[i,4],mean_real_avg_infected,std_real_avg_infected)
        if(dfinfected.iloc[i,5]==0):
            pg_infected=ng0i/G0
        if(dfinfected.iloc[i,5]==1):
            pg_infected=ng1i/G1
        p_age_infected=normpdf(dfinfected.iloc[i,6],mean_age_infected,std_age_infected)
        if(dfinfected.iloc[i,7]==1):
            ps_infected=ns1i/s1
        if(dfinfected.iloc[i,7]==2):
            ps_infected=ns2i/s2
        if(dfinfected.iloc[i,7]==3):
            ps_infected=ns3i/s3
        px_infected=float(p_img_min_infected*p_img_avg_infected*p_real_min_infected*p_real_avg_infected*pg_infected*p_age_infected*ps_infected*p_infected)
        p3.append(px_infected)
    x_infected=np.mean(p3)
    print(x_infected)
    
    df.shape #rows and columns
    from scipy.stats import invgauss
    df=df.replace(np.nan,0)
    count=0
    for i in range(len(df)):
        if(df.iloc[i,0]=='COPD'):
            if(df.iloc[i,1]==0):
                if(df.iloc[i,5]==0):
                    pg=ng0c/G0
                if(df.iloc[i,5]==1):
                    pg=ng1c/G1
                if(df.iloc[i,7]==1):
                    ps=ns1c/s1
                if(df.iloc[i,7]==2):
                    ps=ns2c/s2
                if(df.iloc[i,7]==3):
                    ps=ns3c/s3
                m=math.log(x)-math.log(pcopd)-(math.log(normpdf(df.iloc[i,6],mean_age,std_age))+math.log(ps)+math.log(pg))
                
                n=math.exp(m)
            
                df.iloc[i,1]=invgauss.rvs(n,mean_img_min,std_img_min)
                df.iloc[i,2]=invgauss.rvs(n,mean_img_avg,std_img_avg)
                df.iloc[i,3]=invgauss.rvs(n,mean_real_min,std_real_min)
                df.iloc[i,4]=invgauss.rvs(n,mean_real_avg,std_real_avg)
        if(df.iloc[i,0]=='HC'):
            if(df.iloc[i,1]==0):
                if(df.iloc[i,5]==0):
                    pg_hc=ng0h/G0
                if(df.iloc[i,5]==1):
                    pg_hc=ng1h/G1
                if(df.iloc[i,7]==1):
                    ps_hc=ns1h/s1
                if(df.iloc[i,7]==2):
                    ps_hc=ns2h/s2
                if(df.iloc[i,7]==3):
                    ps_hc=ns3h/s3
                m_hc=math.log(x_hc)-math.log(phc)-(math.log(normpdf(df.iloc[i,6],mean_age_hc,std_age_hc))+math.log(ps_hc)+math.log(pg_hc))
            
                n_hc=math.exp(m_hc)
                
                df.iloc[i,1]=invgauss.rvs(n_hc,mean_img_min_hc,std_img_min_hc)
                df.iloc[i,2]=invgauss.rvs(n_hc,mean_img_avg_hc,std_img_avg_hc)
                df.iloc[i,3]=invgauss.rvs(n_hc,mean_real_min_hc,std_real_min_hc)
                df.iloc[i,4]=invgauss.rvs(n_hc,mean_real_avg_hc,std_real_avg_hc)
        if(df.iloc[i,0]=='Asthma'):
            if(df.iloc[i,1]==0):
                if(df.iloc[i,5]==0):
                    pg_asthama=ng0a/G0
                if(df.iloc[i,5]==1):
                    pg_asthama=ng1a/G1
                if(df.iloc[i,7]==1):
                    ps_asthama=ns1a/s1
                if(df.iloc[i,7]==2):
                    ps_asthama=ns2a/s2
                if(df.iloc[i,7]==3):
                    ps_asthama=ns3a/s3
                m_asthama=math.log(x_asthama)-math.log(p_asthama)-(math.log(normpdf(df.iloc[i,6],mean_age_asthama,std_age_asthama))+math.log(ps_asthama)+math.log(pg_asthama))
                
                n_asthama=math.exp(m_asthama)
            
                df.iloc[i,1]=invgauss.rvs(n_asthama,mean_img_min_asthama,std_img_min_asthama)
                df.iloc[i,2]=invgauss.rvs(n_asthama,mean_img_avg_asthama,std_img_avg_asthama)
                df.iloc[i,3]=invgauss.rvs(n_asthama,mean_real_min_asthama,std_real_min_asthama)
                df.iloc[i,4]=invgauss.rvs(n_asthama,mean_real_avg_asthama,std_real_avg_asthama)
        if(df.iloc[i,0]=='Infected'):
            if(df.iloc[i,1]==0):
                if(df.iloc[i,5]==0):
                    pg_infected=ng0i/G0
                if(df.iloc[i,5]==1):
                    pg_infected=ng1i/G1
                if(df.iloc[i,7]==1):
                    ps_infected=ns1i/s1
                if(df.iloc[i,7]==2):
                    ps_infected=ns2i/s2
                if(df.iloc[i,7]==3):
                    ps_infected=ns3i/s3
                m_infected=math.log(x_infected)-math.log(p_infected)-(math.log(normpdf(df.iloc[i,6],mean_age_infected,std_age_infected))+math.log(ps_infected)+math.log(pg_infected))
                n_infected=math.exp(m_infected)
                df.iloc[i,1]=invgauss.rvs(n_infected,mean_img_min_infected,std_img_min_infected)
                df.iloc[i,2]=invgauss.rvs(n_infected,mean_img_avg_infected,std_img_avg_infected)
                df.iloc[i,3]=invgauss.rvs(n_infected,mean_real_min_infected,std_real_min_infected)
                df.iloc[i,4]=invgauss.rvs(n_infected,mean_real_avg_infected,std_real_avg_infected)
            
            
            
    df.isnull().sum()
    fig = plt.figure()
    fig.patch.set_facecolor('grey')
    plt.pie(df['Diagnosis'].value_counts(),colors=['red','black','blue','green'],labels=['HC','Asthama','Infected','COPD'])
    plt.show()
    column_names = ['Imaginary_min','Imaginary_avg','Real_min','Real_avg','Gender','Age','Smoking','Diagnosis']
    
    training_data = df.reindex(columns=column_names)
    
    
    sns.countplot(x = "Gender",hue="Diagnosis", data = df)
    df.head()
    outputs = training_data.iloc[: , -1]
    inputs = training_data.iloc[: , :-1]
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(inputs,outputs,test_size=0.30,random_state=2)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    from sklearn.neighbors import KNeighborsClassifier
    Knn=KNeighborsClassifier(n_neighbors=6,metric='euclidean')
    
    ## apply the knn object on the dataset(training phase)
    Knn.fit(X_train, y_train)
    y_train_pred=Knn.predict(X_train)
    y_train_pred
    from sklearn.metrics import accuracy_score
    
    scores=[]
    for k in range(1,20):
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        pred_test = knn_model.predict(X_test)
        scores.append(accuracy_score(y_test, pred_test))
    scores
    
    
    print("For k = {} accuracy is {}".format(scores.index(max(scores))+1,max(scores)))
    final_model=KNeighborsClassifier(n_neighbors=1,metric='euclidean')
    final_model.fit(X_train,y_train)
    final_train_pred=final_model.predict(X_train)
    final_train_pred
    knn_train=accuracy_score(y_train,final_train_pred)
    print(knn_train)
    final_test_pred=final_model.predict(X_test)
    final_test_pred
    knn_test=accuracy_score(y_test,final_test_pred)
    print(knn_test)
    #Import svm model
    from sklearn import svm
    
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel
    
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    clf.fit(X_test,y_test)
    
    ##syntax:objname.ppredict(input_values)
    y_pred_train=clf.predict(X_train)
    
    svm_train=accuracy_score(y_train,y_train_pred)
    print(svm_train)
    ##syntax:objname.ppredict(input_values)
    y_pred_test=clf.predict(X_test)
    svm_test=accuracy_score(y_test,y_pred_test)
    print(svm_test)
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    dt.fit(X_train,y_train)
    pred = dt.predict(X_train)
    dt_train=accuracy_score(y_train,pred)
    print(dt_train)
    test_pred=dt.predict(X_test)
    print(test_pred)
    
    
    dt_test=accuracy_score(y_test,test_pred)
    print(dt_test)
    # Fitting Naive Bayes to the Training set  
    from sklearn.naive_bayes import GaussianNB  
    classifier = GaussianNB()   
    classifier.fit(X_train,y_train)
    pred = classifier.predict(X_train)
    
    nb_train=accuracy_score(y_train,pred)
    print(nb_train)
    test_pred=dt.predict(X_test)
    
    
    nb_test=accuracy_score(y_test,test_pred)
    print(nb_test)
    from sklearn.ensemble import GradientBoostingClassifier
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    pred=gbc.predict(X_train)
    
    gb_train=accuracy_score(y_train,pred)
    print(gb_train)
    test_pred=gbc.predict(X_test)
    gb_test=accuracy_score(y_test,test_pred)
    print(gb_test)
    acc=[]
    acc.append(knn_train*100)
    acc.append(knn_test*100)
    acc.append(svm_train*100)
    acc.append(svm_test*100)
    acc.append(nb_train*100)
    acc.append(nb_test*100)
    acc.append(dt_train*100)
    acc.append(dt_test*100)
    acc.append(gb_train*100)
    acc.append(gb_test*100)
    print(' knn traning data accuracy = {}\n knn testing data accuracy = {}\n svm traning data accuracy = {} \n svm testing data accuracy = {}\n NB training data accuracy = {} \n NB testing data accuracy = {} \n DT training data accuracy = {} \n DT testing data accuracy = {} \n GB training data accuracy = {} \n GB testing data accuracy = {}'.format(acc[0],acc[1],acc[2],acc[3],acc[4],acc[5],acc[6],acc[7],acc[8],acc[9]))
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (10,5))
    algo=['KNN_train','KNN_test','svm_train','svm_test','nb_train','nb_test','dt_train','dt_test','gb_train','gb_test']
    
    plt.bar(algo,acc,color=['red','green','red','green','red','green','red','green','red','green'])
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.show()
    testSet = [[-311,-305,-422,-445,1,67,1]]
    test = pd.DataFrame(testSet)
    
    result1=gbc.predict(test)
    print('The final prediction on the random test set 1 is',result1)
    pyttsx3("The final prediction on the random test set 1 is"+str(result1))


    testSet2 = [[-323,-301,-405,-466,0,92,2]]
    test = pd.DataFrame(testSet2)
    result2=gbc.predict(test)
    
    print('knn Prediction on the second random test set 2 is:',result2)
    pyttsx3("The final prediction on the random test set 2 is"+str(result2))






btn = Button(frame2, text = "SUBMIT", 
        			width = "15", pady = 10, 
        			font = "bold, 15", 
        			command = submit, bg='#f9f7cb') 

btn.place(x = 250, 
		y = 130
        ) 

# give a title 
root.title("COPD Analysis System") 

# we can not change the size 
# if you want you can change 
root.geometry("650x550+350+200") 

# start the gui 
root.mainloop()

