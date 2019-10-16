from pyforest import *
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

#read csv
daily_df = pd.read_csv('daily 2018.csv')

#change column names
daily_df.columns = daily_df.iloc[0]

#drop first row
daily_df = daily_df.drop([0])

#convert types to float
daily_df['MONTHS'] = daily_df['MONTHS'].astype(float)
daily_df['TEMPERATURE'] = daily_df['TEMPERATURE'].astype(float)
daily_df['DAYS'] = daily_df['DAYS'].astype(float)
daily_df['HUMIDITY'] = daily_df['HUMIDITY'].astype(float)
daily_df['HISTORIC LOAD (MW)'] = daily_df['HISTORIC LOAD (MW)'].astype(float)

#declare features and labels
features = daily_df[['DAYS','MONTHS','TEMPERATURE','HUMIDITY']]
label = daily_df[['HISTORIC LOAD (MW)']]

#split dataset
features_train, features_test, labels_train, labels_test = train_test_split(features, label, test_size=0.1)

print(features_train.shape)
print(features_test.shape)

model = LinearSVR()

model.fit(features_train, labels_train)

print(labels_test)


print(model.predict(features_test))