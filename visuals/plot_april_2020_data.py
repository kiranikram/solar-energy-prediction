from cProfile import label
import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import seaborn as sns
from calendar import monthrange

import warnings
warnings.filterwarnings("ignore")

PATH  =  os.path.dirname(os.path.abspath(__file__))

fig, axs = plt.subplots(6, 2, gridspec_kw={'hspace': 0.5}, sharey=True)
plt.tight_layout()

def get_xlabels():
    x_labels = []

    for i in range(1, 32):
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

    return x_labels

def plot_monthly_data(idx, x0, df, year):
    # Loop over all months
    for month in range(1, 13):
        start_date = dt.datetime(year, month, 1, 0, 0, 0)
        end_date = dt.datetime(year, month, monthrange(year, month)[1], 23, 45, 0)

        mask = (df['DateTime'] > start_date) & (df['DateTime'] <= end_date)
        readings_df = df.loc[mask]
        total_list = readings_df['Total'].tolist()
   
        if(len(total_list) > 0):
            if idx % 2 == 0 and idx != 0:
                x0  += 1

            if(idx % 2 != 0):
                y0 = 1
            else:
                y0 = 0

            axs[x0, y0].plot(total_list)
            axs[x0, y0].set_title(start_date.strftime('%B') + ' ' + str(year))

            idx += 1

    return idx, x0

# Read in data
df = pd.read_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_raw.csv'), header=0)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')


idx = 0
x0 = 0
idx, x0 = plot_monthly_data(idx, x0, df, 2020)

idx, x0 = plot_monthly_data(idx, x0, df, 2021)

x_labels = get_xlabels()
fig.subplots_adjust(hspace=0.2)

for ax in axs.flat:
    ax.set(ylabel='(kWh)')

#axs.set_title('Solar panel production for April 2020')
#axs.set_ylabel('Solar energy production (kWh).')
#axs.set_xlabel('Time of day (15 min intervals)')
#axs.set_xticklabels(x_labels, rotation = 90)
#axs.set_xticks(ticks=range(0,len(x_labels)))

plt.show()

t =1
