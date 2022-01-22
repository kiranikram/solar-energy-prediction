
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

PATH  =  os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_raw_april.csv'), header=0)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')


df['day'] = df['DateTime'].dt.day
df['month'] = df['DateTime'].dt.month
df['hour'] = df['DateTime'].dt.hour
df['minute'] = df['DateTime'].dt.minute

# Drop DateTime column
df = df.drop(columns=['DateTime'])

df = df[['day', 'month', 'hour', 'minute', 'Total']]

categorical_features = ["day", "month", "hour", "minute"]

label_encoders = {}
for cat_col in categorical_features:
    label_encoders[cat_col] = LabelEncoder()
    print(df[cat_col] )
    df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
    print(df[cat_col] )


cat_dims = [int(df[col].nunique()) for col in categorical_features]

emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

#Print embedding dimensions
print(emb_dims)

# Normalise the data
#scaler = StandardScaler()
#df['Total'] = scaler.fit_transform(df['Total'].values.reshape(-1, 1))

df['Total'] = df['Total'].apply(lambda x: x / 100)

df.to_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_clean_april.csv'), index=False)

