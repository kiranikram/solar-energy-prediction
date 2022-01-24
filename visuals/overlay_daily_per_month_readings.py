import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler

PATH  =  os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(PATH, 'data', 'sunrock_raw.csv'), header=0)
total_list = df['Total'].tolist()
dates_list = df['DateTime'].tolist()

print(len(total_list) / 96)

y = total_list[0:96]

x_label = []
for i in range(24):
    hour = str(i)
    if len(hour) == 1: hour = '0' + hour
    x_label.append(hour + ':00')
    x_label.append(hour + ':15')
    x_label.append(hour + ':30')
    x_label.append(hour + ':45')


plt.ylabel('Solar energy production (kWh).')
plt.xlabel('Time of day (15 min intervals)')
#plt.xticks(rotation=45)
for i in range(int(len(total_list) / 96)):
    plt.plot(total_list[i * 96: 96 * (i+1)], label = "Day {}".format(i))

plt.xticks(ticks=range(0,len(x_label)) ,labels=x_label, rotation = 45)
#plt.legend()
plt.suptitle('Daily Solar panel production')
plt.show()
p = 1
