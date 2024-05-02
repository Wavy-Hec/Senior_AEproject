from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Path to your TensorBoard log directory
log_dir = 'ppo_ant_tensorboard/enc27_9_1'

# Initialize an accumulator and reload it to access the scalars
ea = event_accumulator.EventAccumulator(log_dir,
                                        size_guidance={event_accumulator.SCALARS: 0})
ea.Reload()

# List all tags in the log file
print(ea.Tags()['scalars'])

# Extract and convert scalar data to pandas DataFrame
def extract_scalars(tag_name):
    events = ea.Scalars(tag_name)
    data = [(event.step, event.value) for event in events]
    df = pd.DataFrame(data, columns=['step', tag_name])
    return df

# Example tag names, replace these with actual tag names from your logs
df_accuracy = extract_scalars('rollout/ep_rew_mean')
df_loss = extract_scalars('rollout/ep_len_mean')

# Merging dataframes if you have multiple tags to compare
df = pd.merge(df_accuracy, df_loss, on='step')

# # Plotting
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
sns.lineplot(data=df, x="step", y="rollout/ep_rew_mean").set_title("Accuracy")
plt.subplot(1, 2, 2)
sns.lineplot(data=df, x="step", y="rollout/ep_len_mean").set_title("Loss")
plt.savefig("figs/test.png")
