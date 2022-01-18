import os
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import matplotlib.ticker as plticker
import matplotlib.ticker as ticker

PATH  =  os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_raw_april.csv'), header=0)
total_list = df['Total'].tolist()
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')

start_date = dt.datetime(2020, 4, 1, 0, 0, 0)
end_date = dt.datetime(2020, 4, 30, 23, 0, 0)

mask = (df['DateTime'] > start_date) & (df['DateTime'] <= end_date)

april_readings = df.loc[mask]

x_labels = []
for i in range(start_date.day, end_date.day + 1):
    day = str(i)
    for i in range(24):
        hour = str(i)
        if len(hour) == 1: hour = '0' + hour
        if i%24==0:
            x_labels.append(day + ' ' + hour + ':00')
        else:
            x_labels.append('')
        
        x_labels.append('')
        x_labels.append('')
        x_labels.append('')

        #x_labels.append(day + ' ' + hour + ':00')
        #x_labels.append(day + ' ' + hour + ':15')
        #x_labels.append(day + ' ' + hour + ':30')
        #x_labels.append(day + ' ' + hour + ':45')


fig, ax = plt.subplots(1,1)
ax.set_title('Solar panel production for April 2020')
ax.set_ylabel('Solar energy production (kWh).')
ax.set_xlabel('Time of day (15 min intervals)')
ax.set_xticklabels(x_labels, rotation = 45)
ax.set_xticks(ticks=range(0,len(x_labels)))
ax.plot(total_list)

#ax.set_xticks(ax.get_xticks()[::10])

plt.show()
p = 1
