import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file generated by your code
df = pd.read_csv("output_resample_filter/Sub_9_0_resample.csv")

# Calculate time in seconds
frame_rate = 1 # frames per second
df['Time'] = df.index / frame_rate

# Create figure
plt.figure(figsize=(16, 6))  # Adjust width and height as needed

# Plot both AC_red and SPO2 on the same y-axis
plt.plot(df['Time'], df['Pridict_SPO2'], label='Pridict_SPO2', color='red')
plt.plot(df['Time'], df['SPO2'], label='SPO2_GT', color='blue')

# Add labels and title
plt.xlabel('Time (seconds)')
plt.ylabel('SPO2 (%)')
plt.title('Pridict_SPO2 and SPO2_GT')

# Add legend
plt.legend()

plt.show()
