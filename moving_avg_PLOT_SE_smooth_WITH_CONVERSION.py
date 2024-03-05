import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# File path of the CSV file
#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/mating/smooth/male_smooth_intervals.csv"

file_path = "/home/joeh/Auditory_filtering_experiment/long/data/new_script/canton_s_mating/canton_S_fast.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/gaboxadol_sleeping/mating/responders_only/gbd_long_sleep.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/Optogenetics/awake-mating/JO15_GTACR/Light_level_5/results/jo15_gtacr_stats.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/Optogenetics/awake-mating/canton_S_wildtype_control_lightlevel5/sleeping_male_mating_stats_per_interval.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/Optogenetics/awake-mating/uas-gtacr-genetic-control/UAS_GTACR_control.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/Optogenetics/awake-mating/JO15_GTACR/lights_off/results/awake_jo15_gtacr_lights_off.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/R3D_TNT/TNTe/results/R3D_TNTe_stats.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/R3D_TNT/IMPT_tnt/R3D_IMP_tnt_stats.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/noise/smooth/male_noise_smooth_intervals.csv"

#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/noise/smooth/male_noise_smooth_intervals.csv"


#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/noise/male_noise_stats.csv"


#file_path = "/home/joeh/Auditory_filtering_experiment/long/data/pink_noise/female_noise_stats_per_interval.csv"

# Read the CSV file into a Pandas DataFrame
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("File not found. Please provide a valid file path.")
    exit()
except pd.errors.EmptyDataError:
    print("The file is empty.")
    exit()
except pd.errors.ParserError:
    print("Unable to parse the file. Please check the file format.")
    exit()

# Check if the required columns exist in the DataFrame
required_columns = ['Time_From_First_Sound', 'Average_Locomotion', 'Standard_Error']
if not all(col in df.columns for col in required_columns):
    print("The CSV file should have columns: 'Time_From_First_Sound', 'Average_Locomotion', and 'Standard_Error'.")
    exit()

# Convert pixels per second to millimeters per second using the new conversion factor
conversion_factor = 1 / 193  # centimeters per pixel (adjusted)

# Convert 'Average_Locomotion' from pixels per second to millimeters per second
df['Average_Locomotion_mm_per_sec'] = df['Average_Locomotion'] * conversion_factor * 10  # Convert cm to mm

# Convert 'Standard_Error' from pixels per second to millimeters per second
df['Standard_Error_mm_per_sec'] = df['Standard_Error'] * conversion_factor * 10  # Convert cm to mm

# Filter data for 1000 milliseconds before and after the sound
time_window = 2800  # milliseconds
sound_time = 0  # Assuming sound time is at 0, adjust accordingly if it's different
df_filtered = df[(df['Time_From_First_Sound'] >= sound_time - time_window) &
                 (df['Time_From_First_Sound'] <= sound_time + time_window)]

# Applying Savitzky-Golay smoothing to 'Average_Locomotion' column
window_length = 10  # Adjust this value as needed
polyorder = 1  # Adjust this value as needed 6 for the ones with individual plotting_
df_filtered['Average_Locomotion_smoothed'] = savgol_filter(df_filtered['Average_Locomotion'], window_length, polyorder)

# Applying Savitzky-Golay smoothing to 'Standard_Error' column
df_filtered['Standard_Error_smoothed'] = savgol_filter(df_filtered['Standard_Error'], window_length, polyorder)

# Plotting
plt.figure(figsize=(10, 5))

# Plot the smoothed locomotion activity line using the converted values
plt.plot(df_filtered['Time_From_First_Sound'], df_filtered['Average_Locomotion_mm_per_sec'], label='Smoothed Average Locomotion (mm/s)')

# Plot the standard error band using the converted values
plt.fill_between(df_filtered['Time_From_First_Sound'],
                 df_filtered['Average_Locomotion_mm_per_sec'] - df_filtered['Standard_Error_mm_per_sec'],
                 df_filtered['Average_Locomotion_mm_per_sec'] + df_filtered['Standard_Error_mm_per_sec'],
                 color='grey', alpha=0.3, label='Smoothed Standard Error (mm/s)')

# Plot a red vertical line at the sound time on the X-axis
plt.axvline(x=sound_time, color='red', linestyle='--', label='Sound Time')

# Set labels and title with updated y-axis label
#plt.xlabel('Mating Call')
plt.ylabel('Average Locomotion (mm/s)')  # Updated y-axis label
plt.ylim(0, 25)
plt.xlim (-2800,2800)
# Set other plot configurations as needed

# Show the plot
#plt.legend()
#plt.grid(True)
plt.show()
