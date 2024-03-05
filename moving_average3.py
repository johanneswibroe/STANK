import os
import pandas as pd
import numpy as np
from scipy import stats

# Folder path containing the CSV files
#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/mating/smooth"
#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/noise/smooth"

#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/noise"

#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/sleeping/gradient/mating/responders_only"


#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/Optogenetics/awake-mating/JO15_GTACR/lights_off"

#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/Optogenetics/awake-mating/canton_S_wildtype_control_lightlevel5"


#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/Optogenetics/awake-mating/uas-gtacr-genetic-control"

folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/new_script/canton_s_mating"

#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/mating"

#folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/mating"

# Initialize lists to store locomotion data
all_time_intervals = []
all_locomotion_averages = []

# Function to extract locomotion activity within a time range for each file
def extract_locomotion_within_range(times, locomotions, first_sound_time):
    time_interval = 2800  # 2.8 seconds on each side
    time_before = first_sound_time - time_interval
    time_after = first_sound_time + time_interval

    mask = (times >= time_before) & (times <= time_after)
    locomotions_within_range = locomotions[mask]

    return locomotions_within_range

# Process each CSV file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)

        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)

        times = df['Time (milliseconds)']
        locomotions = df['Locomotion (mm/sec)']
        first_sound_time = df['First Sound Time'].iloc[0]  # Extract the first sound time from the file

        # Extract locomotor activity for 2.8 seconds before 'First Sound Time' for each file
        locomotions_within_range = extract_locomotion_within_range(times, locomotions, first_sound_time)

        # Calculate averages within each 100-millisecond interval before 'First Sound Time'
        interval_start = first_sound_time - 2800  # Start from 2.8 seconds before First Sound Time
        while interval_start <= first_sound_time + 2800:  # Until 2.8 seconds after First Sound Time
            interval_end = interval_start + 35  #70 milliseconds
            avg_locomotion = locomotions_within_range[(times >= interval_start) & (times < interval_end)].mean()
            
            all_time_intervals.append(interval_start - first_sound_time)  # Relative to First Sound Time
            all_locomotion_averages.append(avg_locomotion if not pd.isnull(avg_locomotion) else 0)  # Handle NaN values

            interval_start += 35  # Move to the next 70 milliseconds interval

# Create a DataFrame with all time intervals and locomotion averages
result_df = pd.DataFrame({'Time_From_First_Sound': all_time_intervals, 'Average_Locomotion': all_locomotion_averages})

# Group by time intervals
grouped_data = result_df.groupby('Time_From_First_Sound')['Average_Locomotion']

# Calculate statistics for each time interval
result_stats = pd.DataFrame()   
result_stats['Time_From_First_Sound'] = grouped_data.mean().index
result_stats['Average_Locomotion'] = grouped_data.mean().values
result_stats['Standard_Deviation'] = grouped_data.std().values
result_stats['Standard_Error'] = grouped_data.sem().values

# Calculate confidence intervals (95% confidence)
confidence_intervals = grouped_data.apply(lambda x: stats.sem(x) * stats.t.ppf((1 + 0.95) / 2, len(x) - 1))
result_stats['Confidence_Interval_Low'] = grouped_data.mean() - confidence_intervals
result_stats['Confidence_Interval_High'] = grouped_data.mean() + confidence_intervals

# Get the output folder path from one of the input file
output_folder = os.path.dirname(os.path.abspath(file_path))

# Save the statistics DataFrame to a CSV file in the input folder
output_file = os.path.join(output_folder, 'canton_S_fast.csv')
result_stats.to_csv(output_file, index=False)
