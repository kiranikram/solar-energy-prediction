
import os
import pandas as pd
import numpy as np
from joblib import dump

from sklearn.preprocessing import StandardScaler, LabelEncoder

PATH  =  os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_raw_spring.csv'), header=0)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')
df['day'] = df['DateTime'].dt.day
df['month'] = df['DateTime'].dt.month
df['hour'] = df['DateTime'].dt.hour
df['minute'] = df['DateTime'].dt.minute

hours_in_day = 24
month_in_year = 12
days_in_month = 31
mins_in_hour = 60

df['sin_day'] = np.sin(2*np.pi* df['day'].apply(lambda x: x / days_in_month))
df['cos_day'] = np.cos(2*np.pi* df['day'].apply(lambda x: x / days_in_month))

df['sin_month'] = np.sin(2*np.pi* df['month'].apply(lambda x: x / month_in_year))
df['cos_month'] = np.cos(2*np.pi* df['month'].apply(lambda x: x / month_in_year))

df['sin_hour'] = np.sin(2*np.pi* df['month'].apply(lambda x: x / hours_in_day))
df['cos_hour'] = np.cos(2*np.pi* df['month'].apply(lambda x: x / hours_in_day))

df['sin_min'] = np.sin(2*np.pi*df['month'].apply(lambda x: x / mins_in_hour))
df['cos_min'] = np.cos(2*np.pi*df['month'].apply(lambda x: x / mins_in_hour))

# Drop DateTime column
#df = df.drop(columns=['DateTime'])
df = df[['day', 'month', 'hour', 'minute', 'sin_day', 'cos_day', 'sin_month', 'cos_month', 'sin_hour', 'cos_hour', 'sin_min', 'cos_min', 'Total', 'DateTime']]

categorical_features = ["day", "month", "hour", "minute"]
label_encoders = {}
for cat_col in categorical_features:
    label_encoders[cat_col] = LabelEncoder()
    df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    dump(label_encoders[cat_col], os.path.join(PATH, '..\\' 'encoders', 'label_encoder_{}.joblib'.format(cat_col)))



cat_dims = [int(df[col].nunique()) for col in categorical_features]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

#Print embedding dimensions
print(emb_dims)

# Normalise the data
#scaler = StandardScaler()
#df['Total'] = scaler.fit_transform(df['Total'].values.reshape(-1, 1))

df['Total'] = df['Total'].apply(lambda x: x / 100)


# Split the data into train and test, validation
# Randomly get datapoints

test_val_split_size = int(len(df) * 0.2)

df_test_val = df.tail(test_val_split_size)

df.drop(df.tail(test_val_split_size).index, inplace=True)

half_split_indice = int(len(df_test_val) / 2)
df_test = df_test_val.iloc[:half_split_indice]
df_val = df_test_val.iloc[half_split_indice:]


df.to_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_clean_spring_train.csv'), index=False)
df_test.to_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_clean_spring_test.csv'), index=False)
df_val.to_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_clean_spring_val.csv'), index=False)

