import os
import pandas as pd

def calculate_response_time(csv_folder):
    response_times = []

    # Loop through all CSV files in the specified folder
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_folder, filename)
            
            # Read CSV file into a Pandas DataFrame
            df = pd.read_csv(file_path)

            # Get the time of the first sound
            first_sound_time = df.iloc[0]["First Sound Time"]

            # Filter rows after the first sound time
            df_after_first_sound = df[df["Time (milliseconds)"] >= first_sound_time]

            # Filter rows where Locomotion > 0
            nonzero_loc = df_after_first_sound[df_after_first_sound["Locomotion (pixels/sec)"] > 0]

            # Check if there are any rows with Locomotion > 0
            if not nonzero_loc.empty:
                # Get the time when Locomotion increases from zero
                response_time = nonzero_loc.iloc[0]["Time (milliseconds)"]

                # Calculate the time difference
                time_difference = response_time - first_sound_time

                # Append the result to the response_times list
                response_times.append({"Filename": filename, "Response Time (ms)": time_difference})

    # Create a DataFrame from the list of response times
    result_df = pd.DataFrame(response_times)

    # Save the result to a new CSV file
    result_df.to_csv("response_times.csv", index=False)

if __name__ == "__main__":
    folder_path = "/home/joeh/Auditory_filtering_experiment/long/data/male/gaboxadol_sleeping/mating/responders_only"
    calculate_response_time(folder_path)
