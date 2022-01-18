import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("train.csv",)

categorical_features = ["day", "month", "hour", "min"]
label_encoders = {}

for cat_col in categorical_features:
    label_encoders[cat_col] = LabelEncoder()
    data[cat_col] = label_encoders[cat_col].fit_transform(data[cat_col])

