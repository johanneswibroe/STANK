import os
import pandas as pd
import subprocess
from datetime import datetime

# Define the folder path
folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/new_script/C+Ds:_flies/E_UAS_RDL_RNAi_x_w1118(cs).csv/SD"

# Initialize an empty list to store the results
results = []

# Loop over all files in the folder
for filename in os.listdir(folder_path):
    # Check if the file is a CSV file
    if filename.endswith(".csv"):
        # Read in the CSV file as a pandas DataFrame
        df = pd.read_csv(os.path.join(folder_path, filename))
        
        # Initialize a dictionary to store the results for this file
        file_results = {"filename": filename}
        
        # Calculate the first sound time
        first_sound_time = df["First Sound Time"].iloc[0]
        
        # Calculate the time range for the pre-onset interval
        pre_onset_start = first_sound_time - 2800
        pre_onset_end = first_sound_time
        
        # Filter the DataFrame to only include rows within the pre-onset interval
        pre_onset_df = df[(df["Time (milliseconds)"] >= pre_onset_start) & (df["Time (milliseconds)"] < first_sound_time)]
        
        # Calculate the average of the "Locomotion (pixels/sec)" column for the pre-onset interval
        pre_onset_average = pre_onset_df["Locomotion (mm/sec)"].mean()
        
        # Add the pre-onset average to the dictionary for this file
        file_results["pre_onset_1_average"] = pre_onset_average
        
        # Calculate the time range for the post-onset interval
        post_onset_start = first_sound_time
        post_onset_end = first_sound_time + 2800
        
        # Filter the DataFrame to only include rows within the post-onset interval
        post_onset_df = df[(df["Time (milliseconds)"] > first_sound_time) & (df["Time (milliseconds)"] <= post_onset_end)]
        
        # Calculate the average of the "Locomotion (pixels/sec)" column for the post-onset interval
        post_onset_average = post_onset_df["Locomotion (mm/sec)"].mean()
        
        # Add the post-onset average to the dictionary for this file
        file_results["post_onset_1_average"] = post_onset_average
        
        # Calculate the difference between pre-onset and post-onset
        difference = post_onset_average - pre_onset_average
        
        # Add the difference to the dictionary for this file
        file_results["difference"] = difference
        
        # Get the creation time and date of the file using stat command
        try:
            stat_output = subprocess.check_output(['stat', '--format', '%y', os.path.join(folder_path, filename)])
            creation_time_str = stat_output.decode().strip().split('.')[0]  # Extracting and formatting the date/time string
            creation_time = datetime.strptime(creation_time_str, '%Y-%m-%d %H:%M:%S')  # Convert to datetime object
            
            # Calculate the difference in minutes between the creation time and 9 AM
            zt_minutes = (creation_time.hour - 9) * 60 + creation_time.minute
            file_results["ZT"] = zt_minutes
            
            # Format the creation time to European date format
            creation_time_str_eu = creation_time.strftime('%d.%m.%Y %H:%M:%S')  # Format to European date format
        except subprocess.CalledProcessError:
            creation_time_str_eu = "N/A"
        
        # Add the creation time and date to the dictionary for this file
        file_results["creation_time"] = creation_time_str_eu
        
        # Add the results for this file to the overall results list
        results.append(file_results)

# Convert the results list to a pandas DataFrame and save it to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(folder_path, "results.csv"), index=False)
