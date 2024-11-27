import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTEN

def prepare(test_size=0.1,four_class = True):
    df = pd.read_csv('./Datasets/Final/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df['TotalCharges'] =  pd.to_numeric(df['TotalCharges'],errors='coerce')
    df.loc[df['TotalCharges'].isnull(),'TotalCharges'] = df.loc[df['TotalCharges'].isnull(),'MonthlyCharges']
    
    df.drop(columns='customerID',axis=1,inplace=True)

    df['customer_value'] = df['Churn']
    df['customer_churn'] = df['Churn']


    df['customer_value'].loc[df['TotalCharges'] >= df['TotalCharges'].median()] = "High"
    df['customer_value'].loc[df['TotalCharges'] < df['TotalCharges'].median()] = "Low"


    df['customer_churn'].loc[(df['customer_value'] == "Low") & (df['Churn'] == "No")] = "Low_NoChurn"
    df['customer_churn'].loc[(df['customer_value'] == "Low") & (df['Churn'] == "Yes")] = "Low_Churn"
    df['customer_churn'].loc[(df['customer_value'] == "High") & (df['Churn'] == "No")] = "High_NoChurn"
    df['customer_churn'].loc[(df['customer_value'] == "High") & (df['Churn'] == "Yes")] = "High_Churn"
    
    # df['customer_churn'].loc[(df['customer_value'] == "Low") & (df['Churn'] == "No")] = 0
    # df['customer_churn'].loc[(df['customer_value'] == "Low") & (df['Churn'] == "Yes")] = 1
    # df['customer_churn'].loc[(df['customer_value'] == "High") & (df['Churn'] == "No")] = 2
    # df['customer_churn'].loc[(df['customer_value'] == "High") & (df['Churn'] == "Yes")] = 3

    X = df[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
           'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
           'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
           'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
           'MonthlyCharges', 'TotalCharges']]
    if four_class == True:
        y = df['customer_churn']
    else:
        y = df['Churn']
    
    return train_test_split(X,y,test_size=test_size,random_state=0)

def preprocess_pipeline():
    # preprocessing pipeline
    #1
    MonthlyCharges_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    #2
    TotalCharges_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    #3
    gender_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #4
    Partner_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #5
    Dependents_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #6
    PhoneService_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #7
    MultipleLines_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #8
    InternetService_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #9
    OnlineSecurity_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #10
    OnlineBackup_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #11
    DeviceProtection_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #12
    TechSupport_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #13
    StreamingTV_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #14
    StreamingMovies_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #15
    Contract_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #16
    PaperlessBilling_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder())
    ])
    #17
    PaymentMethod_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('MonthlyCharges_transformer', MonthlyCharges_transformer, ['MonthlyCharges']),
            ('TotalCharges_transformer', TotalCharges_transformer, ['TotalCharges']),
            ('gender_transformer', gender_transformer, ['gender']),
            ('Partner_transformer', Partner_transformer, ['Partner']),
            ('Dependents_transformer', Dependents_transformer, ['Dependents']),
            ('PhoneService_transformer', PhoneService_transformer, ['PhoneService']),
            ('MultipleLines_transformer', MultipleLines_transformer, ['MultipleLines']),
            ('InternetService_transformer', InternetService_transformer, ['InternetService']),
            ('OnlineSecurity_transformer', OnlineSecurity_transformer, ['OnlineSecurity']),
            ('OnlineBackup_transformer', OnlineBackup_transformer, ['OnlineBackup']),
            ('DeviceProtection_transformer', DeviceProtection_transformer, ['DeviceProtection']),
            ('TechSupport_transformer', TechSupport_transformer, ['TechSupport']),
            ('StreamingTV_transformer', StreamingTV_transformer, ['StreamingTV']),
            ('StreamingMovies_transformer', StreamingMovies_transformer, ['StreamingMovies']),
            ('Contract_transformer', Contract_transformer, ['Contract']),
            ('PaperlessBilling_transformer', PaperlessBilling_transformer, ['PaperlessBilling']),
            ('PaymentMethod_transformer', PaymentMethod_transformer, ['PaymentMethod']),
        ],remainder='passthrough',)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
    ])
    
    return pipeline

def transform(X):
    pipe = preprocess_pipeline()
    return pipe.fit_transform(X)

def up_sample(X_train,y_train):
    smote = SMOTEN(random_state=0)
    return smote.fit_resample(X_train,y_train)