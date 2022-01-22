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

def plot_data(df, year, axs, idx=0, x0=0):

    for month in range(1, 13):
        month_name = dt.datetime(year, month, 1, 0, 0, 0).strftime('%B')
        no_of_days = monthrange(year, month)[1]

        avg_daily_list = []

        start_month_date = dt.datetime(year, month, 1, 0, 0, 0)
        end_month_date = dt.datetime(year, month, monthrange(year, month)[1], 23, 45, 0)
        mask = (df['DateTime'] > start_month_date) & (df['DateTime'] <= end_month_date)
        readings_df = df.loc[mask]

        if(len(readings_df) > 0):

            for day in range(1, no_of_days + 1):
                start_dt = dt.datetime(year, month, day, 0, 0, 0)
                end_dt = dt.datetime(year, month, day, 23, 45, 0)
                mask = (df['DateTime'] > start_dt) & (df['DateTime'] <= end_dt)
                readings_df = df.loc[mask]
                total_list = readings_df['Total'].tolist()
                if len(total_list) > 0:
                    avg = sum(total_list) / len(total_list)
                    avg_daily_list.append(avg)

            print(len(avg_daily_list))
            if(len(avg_daily_list) > 0):
                if idx % 2 == 0 and idx != 0:
                    x0  += 1

                if(idx % 2 != 0):
                    y0 = 1
                else:
                    y0 = 0

                axs[x0, y0].grid()
                axs[x0, y0].plot(avg_daily_list)
                axs[x0, y0].set_title(month_name + ' ' + str(year))
                x_labels = [i + 1 for i in range(len(avg_daily_list)  + 1)]
                axs[x0, y0].set_xticklabels(x_labels)
                axs[x0, y0].set_xticks(ticks=range(0, len(avg_daily_list) + 1))
             
                idx += 1

    return axs, idx, x0

def main():

    fig, axs = plt.subplots(6, 2, gridspec_kw={'hspace': 0.5}, sharey=True)
    plt.tight_layout()
    plt.grid(True)

    # Read in data
    df = pd.read_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_raw.csv'), header=0)
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y %H:%M')

    axs, idx, x0 = plot_data(df, 2020, axs, idx=0, x0=0)
    plot_data(df, 2021, axs, idx, x0)

    fig.subplots_adjust(hspace=0.2)

    for ax in axs.flat:
        ax.set(ylabel='(kWh)')

    plt.grid(axis = 'y')
    plt.show()

    t =1

if __name__ == "__main__":
    main()

