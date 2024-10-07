import pandas as pd
import numpy as np
import scipy
from enum import Enum
import random
import obspy
from scipy.signal import butter, sosfilt

import matplotlib.pyplot as plt

from dataclasses import dataclass

from pathlib import Path
from glob import glob
import os

data_folder = "./data/data/"


class CelestialBody(Enum):
    Lunar = 0
    Mars = 1


class DataType(Enum):
    Training = 0
    Test = 1


@dataclass
class SeismicData:
    velocity: np.array
    time: np.array
    delta: float
    sampling_rate: float
    filename: str
    max_ranges: list[tuple[int, int]]
    time_of_event: None | float

    def plot(self, title, ax):
        ax.plot(self.time, self.velocity)

        ax.set_xlim([min(self.time), max(self.time)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{self.filename}: {title}')

        lines = []

        for i, (vel_start, vel_end) in enumerate(self.max_ranges):
            lines.append(
                ax.axvline(x=self.time[vel_start],
                           c='blue',
                           label=f'Range {i + 1} Start'))
            lines.append(
                ax.axvline(x=self.time[vel_end],
                           c='blue',
                           label=f'Range {i + 1} End'))

        if self.time_of_event:
            arrival_line = ax.axvline(x=self.time_of_event,
                                      c='red',
                                      label='Arrival')
            ax.legend(handles=lines + [arrival_line])
            return

        ax.legend(handles=lines)


class DataReader:

    def __init__(self,
                 celestial_body=CelestialBody.Lunar,
                 data_type=DataType.Training,
                 filter_data=True,
                 pool_data=False):
        self.data_type = data_type
        if self.data_type == DataType.Training:
            if celestial_body == CelestialBody.Lunar:
                self.catalog_df = pd.read_csv(
                    f'{data_folder}lunar/training/catalogs/apollo12_catalog_GradeA_final.csv'
                )
            else:
                self.catalog_df = pd.read_csv(
                    f'{data_folder}mars/training/catalogs/Mars_InSight_training_catalog_final.csv'
                )
        else:
            folder = "lunar"
            if celestial_body == CelestialBody.Mars:
                folder = "mars"
            self.catalog = list(glob(f'{data_folder}{folder}/test/**/*.mseed', recursive=True))

        self.celestial_body = celestial_body

    def read(self,
             index: int,
             filter_data=True,
             pool_data=False,
             n_max_subsections=10,
             n_max_sections=2):

        traces = None
        time_of_event = None
        event_filename = None

        if self.data_type == DataType.Training:
            seed_data = self.__parse_lunar_data(
                self.catalog_df.iloc[index]
            ) if self.celestial_body == CelestialBody.Lunar else self.__parse_mars_data(
                self.catalog_df.iloc[index])

            time_of_event = seed_data['time_of_event']
            event_filename = seed_data['event_filename']
            traces = seed_data['traces']
        else:
            seed_data = obspy.read(self.catalog[index])

            event_filename = os.path.basename(self.catalog[index])
            traces = seed_data.traces[0].copy()

        time = np.array(traces.times())
        velocity = np.array(traces.data)

        delta = traces.stats.delta
        sampling_rate = traces.stats.sampling_rate

        if filter_data:
            velocity = self.__butter_bandpass_filter(velocity, 0.5, 1.0, 6.0)

        if pool_data:
            data_pooled = self.__max_pool_1d(np.vstack((velocity, time)), 100)

            velocity = data_pooled[0:1, :].flatten()
            time = data_pooled[1:2, :].flatten()
            if time_of_event:

                time_index = self.__find_nearest_time_index(
                    time, time_of_event)

            delta = delta * 100

        ranges = self.__find_peak_ranges(velocity,
                                         time,
                                         n_subsections=n_max_subsections,
                                         n_sections=n_max_sections)

        return SeismicData(velocity=velocity,
                           time=time,
                           delta=delta,
                           filename=event_filename,
                           max_ranges=ranges,
                           sampling_rate=sampling_rate,
                           time_of_event=time_of_event)

    def __max_pool_1d(self, array: np.array, n: int) -> np.array:
        size = array.shape[1] // n

        result = np.zeros((2, size))

        for i in range(size):
            subarr = array[:, i * n:(i * n) + n]
            max = np.max(subarr, axis=1)
            result[0, i] = max[0]
            result[1, i] = max[1]

        return result

    def __find_nearest_time_index(self, timesteps: np.array,
                                  time: float) -> int:
        size = timesteps.shape[0]
        n = size // 2
        max = size
        min = 0
        print("Searching for", time)
        val = 0
        while True:
            val = timesteps[n]

            if max - min <= 1:
                print(n, val, timesteps[n])
                return n

            if time < val:
                max = n
                n = (min + max) // 2
            elif time > val:
                min = n
                n = (max + min) // 2

    def __butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.__butter_bandpass(lowcut, highcut, fs, order=5)
        y = sosfilt(sos, data)
        return y

    def __butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high],
                     btype='band',
                     analog=False,
                     output='sos')
        return sos

    def __parse_lunar_data(self, event: pd.Series) -> dict[str, float]:
        time_of_event = event['time_rel(sec)']
        event_filename = event['filename']

        event_seed = obspy.read(
            f'{data_folder}lunar/training/data/S12_GradeA/{event_filename}.mseed'
        )
        traces = event_seed.traces[0].copy()

        return {
            'time_of_event': time_of_event,
            'event_filename': event_filename,
            'traces': traces
        }

    def __parse_mars_data(self, event: pd.Series) -> dict[str, float]:
        time_of_event = event['time_rel(sec)']
        event_filename = Path(event['filename']).with_suffix('')

        event_seed = obspy.read(
            f'{data_folder}mars/training/data/{event_filename}.mseed')
        traces = event_seed.traces[0].copy()

        return {
            'time_of_event': time_of_event,
            'event_filename': event_filename,
            'traces': traces
        }

    # Splits the graph into 10 subsections, finds the top 3 sections, and returns their ranges.
    def __find_peak_ranges(self,
                           velocity: np.array,
                           time: np.array,
                           n_subsections=10,
                           n_sections=2) -> list[tuple[int, int]]:
        size = velocity.shape[0]
        jump_size = size // n_subsections

        max_ranges: list[tuple[int, tuple[int, int]]] = []

        for i in range(0, size, jump_size):
            vel_range = velocity[i:i + jump_size]

            max_ranges.append([np.max(vel_range), [i, i + jump_size]])

        sorted_max = sorted(max_ranges, key=lambda max_tuple: max_tuple[0])
        sorted_max.reverse()

        return [max_tuple[1] for max_tuple in sorted_max[0:n_sections]]


def main():
    reader = DataReader(celestial_body=CelestialBody.Lunar,
                        data_type=DataType.Training)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    plt.rcParams['keymap.quit'].append(' ')

    i = 0
    reader.read(i, filter_data=False).plot('Unfiltered Data', ax1)
    reader.read(i).plot('Filtered Data', ax2)
    reader.read(i, pool_data=True).plot('Filtered + Max Pooled (N = 100) Data',
                                        ax3)

    plt.show()


if __name__ == "__main__":
    main()
