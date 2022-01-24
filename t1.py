import pandas as pd
import numpy as np
np.random.seed(10)

remove_n = 1
df = pd.DataFrame({"a":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]})
#drop_indices = np.random.choice(df.index, remove_n, replace=False)

test_val_split_size = int(len(df) * 0.5)

df_test_val = df.tail(test_val_split_size)

df.drop(df.tail(test_val_split_size).index, inplace=True)

half_split_indice = int(len(df_test_val) / 2)
df_test = df_test_val.iloc[:half_split_indice]
df_val = df_test_val.iloc[half_split_indice:]

#print(df.head())
print(df_test.head())
print(df_val.head())

x=1
#test_val_split_indicies = np.random.choice(len(df) - 1, test_val_split_size, replace=False)


#test_indices = test_val_split_indicies[0: int(len(test_val_split_indicies) / 2)]
#val_indices = test_val_split_indicies[int(len(test_val_split_indicies)/ 2): ]

#df_test = df.loc[test_indices]
#df = df.drop(test_indices)


#df_val = df.loc[val_indices]
#df = df.drop(val_indices)

#print(df.head())



#df_subset = df.drop(drop_indices)

