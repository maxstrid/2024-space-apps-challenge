import pandas as pd
import numpy as np
import scipy
from enum import Enum
from filtering import butter_bandpass_filter

import matplotlib.pyplot as plt

data_folder = "../nasa_space_apps/demo/space_apps_2024_seismic_detection/space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/"

class DataType(Enum):
    Mars = 0
    Lunar = 1

dara_type = DataType.Lunar;

def read_catalog() -> pd.DataFrame:
    df = pd.read_csv('./apollo12_catalog_GradeA_final.csv')
    return df

def plot_catalog_event(event: pd.Series) -> None:
    time_rel = event['time_rel(sec)']
    filename = event['filename']

    event_df = pd.read_csv(f'{data_folder}/{filename}.csv')

    time = np.array(event_df['time_rel(sec)'].tolist())
    velocity = np.array(event_df['velocity(m/s)'].tolist())

    velocity_filtered = butter_bandpass_filter(velocity, 500, 1250, 5000)

    fig,ax = plt.subplots(1,1, figsize=(10,3))
    ax.plot(time, velocity_filtered)

    ax.set_xlim([min(time), max(time)])
    ax.set_ylabel('Velocity (m/s)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'{filename}')
    arrival_line = ax.axvline(x=time_rel, c='red', label='Arrival')
    ax.legend(handles=[arrival_line])

    plt.show()

def test_events():
    catalog_df = read_catalog()
    for _, row in catalog_df.iterrows():
        plot_catalog_event(row)

def main():
    test_events()

if __name__ == "__main__":
    main()
