# import python packages for data manipulation
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import categorical_encoding as ce
import featuretools as ft
from featuretools.tests.testing_utils import make_ecommerce_entityset
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def split_data(file,target):
    df=pd.read_csv(file)
    X_train,X_test,y_train,y_test=train_test_split(df.drop(target,axis=1),df.deal)
    train = pd.concat([X_train,y_train],axis=1)
    test = pd.concat([X_test,y_test],axis=1)
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    train.drop('index',axis=1,inplace=True)
    test.drop('index',axis=1,inplace=True)
    
    return train, test
    
def filter_outliers(dataframe,threshold):
    """
    Input a data frame and have all outliers filtered to a certain and custom threshold of standard deviations
    """
    dataframe = dataframe[(np.abs(stats.zscore(dataframe)) <= threshold).all(axis=1)]
    return dataframe

def target_encode(df,target,col):
    
    dummy_dict = {}
    dummy_df = df[[col,target]].groupby(col,as_index=False).mean()
    for i in range(len(dummy_df)):
        dummy_dict[dummy_df.iloc[i,0]]=dummy_df.iloc[i,1]
    df[col]=df[col].map(lambda x: dummy_dict[x])
    
    return df[col]

def preprocess_data(data,scaler=StandardScaler()):
    
    data.deal=(data.deal).astype(int)
    
    data['Multiple Entreprenuers']=(data['Multiple Entreprenuers']).astype(int)
    
    for i in range(len(data)):
        data.location.iloc[i]=data.location[i].split(',')[1]
        
    data['website_length']=0
    for i in range(len(data)):
        try:
            data['website_length'][i]=len(data.website[i])
        except:
            data['website_length'][i]=0
            
    data.website_length=pd.cut(data.website_length,bins=3,labels=['short','medium','long'])
    
    data.columns = data.columns.str.replace(' ', '_')
    
    data.website_length=data.website_length.astype(str)
    
    df_filter_num = filter_outliers(data.select_dtypes(include=[int,float]),2)
    
    df_filter_num['asked_size']=pd.cut(df_filter_num.askedFor,bins=3,labels=['small','medium','large'])
    df_filter_num['value_size']=pd.cut(df_filter_num.askedFor,bins=3,labels=['small','medium','large'])
    
    df_filter_cat = data[data.index.isin(df_filter_num.index)].select_dtypes(include='O')
    
    df_filter=pd.concat([df_filter_cat,df_filter_num],axis=1)
    
    df_filter.reset_index(inplace=True)
    
    df_filter.drop('index',axis=1,inplace=True)
    
    df_filter.website.fillna('none',inplace=True)
    
    data = df_filter
    
    for i in range(len(data)):
        if data.website[i]=='none':
            data.website[i]=0
        else:
            data.website[i]=1
    
    data['sharks']=None
    
    for i in range(len(data)):
        data['sharks'][i]=[]
    
    for i in range(len(data)):
        data['sharks'][i].append([data.shark1[i],data.shark2[i],data.shark3[i],data.shark4[i],data.shark5[i]])
    
    for i in range(len(data)):
        data.sharks[i]=data.sharks[i][0]
                  
    data.sharks.astype(str).unique()
                  
    sharks_dict={
    'group_1':['Barbara Corcoran', 'Robert Herjavec', "Kevin O'Leary", 'Daymond John', 'Kevin Harrington'],
    'group_2':['Barbara Corcoran', 'Robert Herjavec', "Kevin O'Leary", 'Daymond John', 'Mark Cuban'],
    'group_3':['Barbara Corcoran', 'Robert Herjavec', "Kevin O'Leary", 'Jeff Foxworthy', 'Daymond John'],
    'group_4':['Lori Greiner', 'Robert Herjavec', "Kevin O'Leary", 'Daymond John', 'Mark Cuban'],
    'group_5':['Lori Greiner', 'Barbara Corcoran', 'Robert Herjavec', "Kevin O\'Leary", 'Mark Cuban'],
    'group_6':['Lori Greiner', "Kevin O'Leary", 'Daymond John', 'Mark Cuban', 'John Paul DeJoria'],
    'group_7':['Lori Greiner', 'Steve Tisch', "Kevin O'Leary", 'Daymond John', 'Mark Cuban'],
    'group_8':['Lori Greiner', "Kevin O'Leary", 'Daymond John', 'Mark Cuban', 'Nick Woodman']
    }
                  
    data['shark_group']=None
                  
    for i in range(len(data)):
        if data.sharks[i]==sharks_dict['group_1']:
            data.shark_group[i]=1
        elif data.sharks[i]==sharks_dict['group_2']:
            data.shark_group[i]=2
        elif data.sharks[i]==sharks_dict['group_3']:
            data.shark_group[i]=3
        elif data.sharks[i]==sharks_dict['group_4']:
            data.shark_group[i]=4
        elif data.sharks[i]==sharks_dict['group_5']:
            data.shark_group[i]=5
        elif data.sharks[i]==sharks_dict['group_6']:
            data.shark_group[i]=6
        elif data.sharks[i]==sharks_dict['group_7']:
            data.shark_group[i]=7
        else:
            data.shark_group[i]=8
                  
    data.drop(['description','sharks','title','episode-season','entrepreneurs'],axis=1,inplace=True)
                  
    for col in data.select_dtypes(exclude=[int,float]).columns:
        data[col]=target_encode(data,'deal',col)
        
    for col in data.columns:
        data[col]=data[col].astype(float)
    
    data.drop(['shark_group','season','value_size','shark1','shark2','shark3','shark4','askedFor'],axis=1,inplace=True)
    
    for col in data.drop('deal',axis=1):
        data[col]=scaler.fit_transform(data[[col]])
                  
    return data
                  
def classification_model(train,test,target,mod,pal):
    X_train = train.drop(target,axis=1)
    y_train = train[target]
    X_test = test.drop(target,axis=1)
    y_test = test[target]
    model=mod
    model.fit(X_train,y_train)
    pred=mod.predict(X_test)
    plt.figure(figsize=(15,8))
    sns.heatmap(confusion_matrix(y_test,pred),cmap=pal)
    print(classification_report(y_test,pred))
                  
def log_reg_coef_graph(X,y):
    logreg=LogisticRegression()
    logreg.fit(X,y)
    fi=pd.DataFrame(logreg.coef_)
    fi.columns=X.columns
    fi_1=fi.T.sort_values(by=0,ascending=False)
    with sns.axes_style({'axes.facecolor':'k'}):
        fi_1.plot(kind='barh',figsize=(15,8),color='skyblue')
        plt.tight_layout()
        plt.title('feature importances with sign')
        plt.show()