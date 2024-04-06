import joblib
import pandas as pd
import math
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, ComprehensiveFCParameters
from sklearn.preprocessing import StandardScaler

import matplotlib.pylab as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

features = ['', 'glucose', 'finger', 'carbs', 'dose']

patients = ['540', '544', '552', '567', '584', '596', '559', '563', '570', '575', '588', '591']
split = 'train'

patient = patients[0]
df_0 = joblib.load('/home/mnawawy/Explainability/2020data/'+patient+'.'+split+'.pkl')
patient = patients[1]
df_1 = joblib.load('/home/mnawawy/Explainability/2020data/'+patient+'.'+split+'.pkl')
patient = patients[2]
df_2 = joblib.load('/home/mnawawy/Explainability/2020data/'+patient+'.'+split+'.pkl')
patient = patients[3]
df_3 = joblib.load('/home/mnawawy/Explainability/2020data/'+patient+'.'+split+'.pkl')
patient = patients[4]
df_4 = joblib.load('/home/mnawawy/Explainability/2020data/'+patient+'.'+split+'.pkl')
patient = patients[5]
df_5 = joblib.load('/home/mnawawy/Explainability/2020data/'+patient+'.'+split+'.pkl')
patient = patients[6]
df_6 = joblib.load('/home/mnawawy/Explainability/2018data/'+patient+'.'+split+'.pkl')
patient = patients[7]
df_7 = joblib.load('/home/mnawawy/Explainability/2018data/'+patient+'.'+split+'.pkl')
patient = patients[8]
df_8 = joblib.load('/home/mnawawy/Explainability/2018data/'+patient+'.'+split+'.pkl')
patient = patients[9]
df_9 = joblib.load('/home/mnawawy/Explainability/2018data/'+patient+'.'+split+'.pkl')
patient = patients[10]
df_10 = joblib.load('/home/mnawawy/Explainability/2018data/'+patient+'.'+split+'.pkl')
patient = patients[11]
df_11 = joblib.load('/home/mnawawy/Explainability/2018data/'+patient+'.'+split+'.pkl')

df_0.finger.fillna(df_0.glucose, inplace=True)
df_0.glucose.fillna(df_0.finger, inplace=True)

df_1.finger.fillna(df_1.glucose, inplace=True)
df_1.glucose.fillna(df_1.finger, inplace=True)

df_2.finger.fillna(df_2.glucose, inplace=True)
df_2.glucose.fillna(df_2.finger, inplace=True)

df_3.finger.fillna(df_3.glucose, inplace=True)
df_3.glucose.fillna(df_3.finger, inplace=True)

df_4.finger.fillna(df_4.glucose, inplace=True)
df_4.glucose.fillna(df_4.finger, inplace=True)

df_5.finger.fillna(df_5.glucose, inplace=True)
df_5.glucose.fillna(df_5.finger, inplace=True)

df_6.finger.fillna(df_6.glucose, inplace=True)
df_6.glucose.fillna(df_6.finger, inplace=True)

df_7.finger.fillna(df_7.glucose, inplace=True)
df_7.glucose.fillna(df_7.finger, inplace=True)

df_8.finger.fillna(df_8.glucose, inplace=True)
df_8.glucose.fillna(df_8.finger, inplace=True)

df_9.finger.fillna(df_9.glucose, inplace=True)
df_9.glucose.fillna(df_9.finger, inplace=True)

df_10.finger.fillna(df_10.glucose, inplace=True)
df_10.glucose.fillna(df_10.finger, inplace=True)

df_11.finger.fillna(df_11.glucose, inplace=True)
df_11.glucose.fillna(df_11.finger, inplace=True)

df_0['carbs'][0] = 0 if math.isnan(df_0['carbs'][0]) else df_0['carbs'][0]
df_1['carbs'][0] = 0 if math.isnan(df_1['carbs'][0]) else df_1['carbs'][0]
df_2['carbs'][0] = 0 if math.isnan(df_2['carbs'][0]) else df_2['carbs'][0]
df_3['carbs'][0] = 0 if math.isnan(df_3['carbs'][0]) else df_3['carbs'][0]
df_4['carbs'][0] = 0 if math.isnan(df_4['carbs'][0]) else df_4['carbs'][0]
df_5['carbs'][0] = 0 if math.isnan(df_5['carbs'][0]) else df_5['carbs'][0]
df_6['carbs'][0] = 0 if math.isnan(df_6['carbs'][0]) else df_6['carbs'][0]
df_7['carbs'][0] = 0 if math.isnan(df_7['carbs'][0]) else df_7['carbs'][0]
df_8['carbs'][0] = 0 if math.isnan(df_8['carbs'][0]) else df_8['carbs'][0]
df_9['carbs'][0] = 0 if math.isnan(df_9['carbs'][0]) else df_9['carbs'][0]
df_10['carbs'][0] = 0 if math.isnan(df_10['carbs'][0]) else df_10['carbs'][0]
df_11['carbs'][0] = 0 if math.isnan(df_11['carbs'][0]) else df_11['carbs'][0]

df_0['dose'][0] = 0 if math.isnan(df_0['dose'][0]) else df_0['dose'][0]
df_1['dose'][0] = 0 if math.isnan(df_1['dose'][0]) else df_1['dose'][0]
df_2['dose'][0] = 0 if math.isnan(df_2['dose'][0]) else df_2['dose'][0]
df_3['dose'][0] = 0 if math.isnan(df_3['dose'][0]) else df_3['dose'][0]
df_4['dose'][0] = 0 if math.isnan(df_4['dose'][0]) else df_4['dose'][0]
df_5['dose'][0] = 0 if math.isnan(df_5['dose'][0]) else df_5['dose'][0]
df_6['dose'][0] = 0 if math.isnan(df_6['dose'][0]) else df_6['dose'][0]
df_7['dose'][0] = 0 if math.isnan(df_7['dose'][0]) else df_7['dose'][0]
df_8['dose'][0] = 0 if math.isnan(df_8['dose'][0]) else df_8['dose'][0]
df_9['dose'][0] = 0 if math.isnan(df_9['dose'][0]) else df_9['dose'][0]
df_10['dose'][0] = 0 if math.isnan(df_10['dose'][0]) else df_10['dose'][0]
df_11['dose'][0] = 0 if math.isnan(df_11['dose'][0]) else df_11['dose'][0]

df_0 = df_0[features].ffill()
df_1 = df_1[features].ffill()
df_2 = df_2[features].ffill()
df_3 = df_3[features].ffill()
df_4 = df_4[features].ffill()
df_5 = df_5[features].ffill()
df_6 = df_6[features].ffill()
df_7 = df_7[features].ffill()
df_8 = df_8[features].ffill()
df_9 = df_9[features].ffill()
df_10 = df_10[features].ffill()
df_11 = df_11[features].ffill()


df_0['PatientID'] = 0
df_1['PatientID'] = 1
df_2['PatientID'] = 2
df_3['PatientID'] = 3
df_4['PatientID'] = 4
df_5['PatientID'] = 5
df_6['PatientID'] = 6
df_7['PatientID'] = 7
df_8['PatientID'] = 8
df_9['PatientID'] = 9
df_10['PatientID'] = 10
df_11['PatientID'] = 11

df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11])

extracted_features = extract_features(df, column_id='PatientID', column_sort='', impute_function=impute, default_fc_parameters=MinimalFCParameters())
print(extracted_features)

#extracted_features.dropna(axis=1, inplace=True)
#print(extracted_features)

joblib.dump(extracted_features, './feature_engineering.pkl')

##create scaled DataFrame where each variable has mean of 0 and standard dev of 1
#scaled_df = StandardScaler().fit_transform(extracted_features)
#print(scaled_df)
#joblib.dump(scaled_df, './feature_engineering.pkl')
