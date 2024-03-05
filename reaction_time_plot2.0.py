import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv("/home/joeh/Auditory_filtering_experiment/long/R3D_TNTE_IMP_and_CANTON_response_times_Lights_ON.csv")

# Filter values between 0 and 2800 ms
df = df[(df['Response Time (ms)'] >= 0) & (df['Response Time (ms)'] <= 2800)]

# Set the style of seaborn for better visualization
sns.set(style="whitegrid")

# Define custom colors for each condition
custom_palette = {"TNTE": "#87CEEB", "IMP": "#98FB98", "CS": "#FF6961"}  # Slightly darker red

# Define the order of conditions
condition_order = ["CS", "IMP", "TNTE"]

# Create a swarm plot with custom colors and order
plt.figure(figsize=(12, 8))

# Add swarm plot for individual data points with custom colors
ax = sns.swarmplot(x="cond", y="Response Time (ms)", data=df, palette=custom_palette, size=8, order=condition_order)

# Add error bars to the plot
ax = sns.pointplot(x="cond", y="Response Time (ms)", data=df, ci="sd", capsize=0.2, join=False, color="black", markers="_", scale=0.75, order=condition_order)

# Remove gridlines
sns.despine()

# Set plot labels and title
plt.xlabel("Condition")
plt.ylabel("Response Time (ms)")
plt.ylim(0, 2800)  # Adjusted ylim to 2800 ms
plt.title("Vertical Raincloud Plot of Response Times by Condition")

# Show the plot
plt.show()
