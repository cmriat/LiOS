# %%
import json

import numpy as np
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = "iframe"  # alternatives: "notebook"

# %%
with open("actions-0.json", "r") as f:
    data = json.load(f)
# Select arm and joint to analyze
arm_choice = "left_arm"  # Can change to 'right_arm'
joint_idx = 2  # Select joint index 0-6

# Create figure
fig = go.Figure()

num_chunks = len(data)
print(f"Total number of action chunks: {num_chunks}")

# Generate colors
import colorsys


def get_color(i, total):
    hue = i / total
    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
    return f"rgb({int(rgb[0] * 255)}, {int(rgb[1] * 255)}, {int(rgb[2] * 255)})"


# Plot each action chunk
for chunk_idx, entry in enumerate(data):
    # Get timestamp and action data
    timestamp = entry["timestamp/ms"]
    delta = entry["delta/ms"]  # Time interval between actions
    actions = np.array(entry[arm_choice]["actions"])

    # Extract joint values
    joint_values = actions[:, joint_idx]
    num_actions = len(joint_values)

    # Calculate time for each action
    # Each action is delta ms apart
    times = timestamp + np.arange(num_actions) * delta

    # Convert to relative time (seconds)
    if chunk_idx == 0:
        start_time = timestamp
    times_relative = (times - start_time) / 1000.0

    # Add trace for this chunk
    color = get_color(chunk_idx, num_chunks)
    fig.add_trace(
        go.Scatter(
            x=times_relative,
            y=joint_values,
            mode="lines+markers",
            name=f"Chunk {chunk_idx + 1}",
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate="<b>Chunk %{fullData.name}</b><br>"
            + "Time: %{x:.4f}s<br>"
            + "Value: %{y:.6f}<br>"
            + "Action index: %{pointIndex}<br>"
            + "<extra></extra>",
        )
    )

# Update layout
fig.update_layout(
    title=dict(
        text=f"{arm_choice.upper()} - Joint {joint_idx} Action Chunk Time Series ({num_chunks} chunks)",
        font=dict(size=16, family="Arial Black"),
    ),
    xaxis_title="Time (s)",
    yaxis_title="Joint Value",
    hovermode="closest",
    width=1200,
    height=600,
    template="plotly_white",
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
)

# Add grid
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

fig.show()

# Print timing information
print("\n" + "=" * 60)
print("Timing Information")
print("=" * 60)
for chunk_idx, entry in enumerate(data):
    timestamp = entry["timestamp/ms"]
    delta = entry["delta/ms"]
    num_actions = len(entry[arm_choice]["actions"])
    chunk_duration = delta * (num_actions - 1)
    chunk_end = timestamp + chunk_duration

    print(f"\nChunk {chunk_idx + 1}:")
    print(f"  Start time: {timestamp:.2f} ms")
    print(f"  End time: {chunk_end:.2f} ms")
    print(f"  Action interval (delta): {delta} ms")
    print(f"  Number of actions: {num_actions}")
    print(f"  Total duration: {chunk_duration} ms")

    if chunk_idx > 0:
        prev_end = data[chunk_idx - 1]["timestamp/ms"] + data[chunk_idx - 1]["delta/ms"] * (
            len(data[chunk_idx - 1][arm_choice]["actions"]) - 1
        )
        gap = timestamp - prev_end
        print(f"  Gap from previous chunk: {gap:.2f} ms")
        if gap < 0:
            print(f"  ⚠️  Overlap: {abs(gap):.2f} ms")
# %%
# Print basic statistics
print("\n" + "=" * 60)
print(f"Joint {joint_idx} Statistics")
print("=" * 60)
all_values = []
for chunk_idx, entry in enumerate(data):
    actions = np.array(entry[arm_choice]["actions"])
    joint_values = actions[:, joint_idx]
    all_values.extend(joint_values)
    if chunk_idx < 5 or chunk_idx >= num_chunks - 2:  # Print first 5 and last 2
        print(f"Chunk {chunk_idx + 1}: start={joint_values[0]:.4f}, end={joint_values[-1]:.4f}")
    elif chunk_idx == 5:
        print("...")

all_values = np.array(all_values)
print(f"\nOverall range: [{all_values.min():.6f}, {all_values.max():.6f}]")
print(f"Overall mean: {all_values.mean():.6f}")

