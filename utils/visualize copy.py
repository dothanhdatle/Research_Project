import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

# Function to plot a single skeleton frame
def plot_skeleton(ax, joint_positions, skeleton_connections, colormap, frame_idx=0):
    """
    Draw a skeleton for a single frame.

    Args:
        ax: matplotlib axis to draw on.
        joint_positions: Tensor/Numpy array of shape (26, 3) representing joint positions.
        skeleton_connections: List of tuples defining the skeleton connections.
        colormap: Dictionary assigning colors to joints.
        frame_idx: Current frame index (optional, for labeling).
    """
    ax.clear()  # Clear the previous frame

    ax.set_title(f"Skeleton Frame: {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot the skeleton connections
    for start, end in skeleton_connections:
        x = [joint_positions[start, 0], joint_positions[end, 0]]
        y = [joint_positions[start, 1], joint_positions[end, 1]]
        z = [joint_positions[start, 2], joint_positions[end, 2]]
        ax.plot(x, y, z, c='k')  # Black lines for bones

    # Plot joints with color mapping
    for color, joints in colormap.items():
        for joint_id in joints:
            ax.scatter(joint_positions[joint_id - 1, 0],
                       joint_positions[joint_id - 1, 1],
                       joint_positions[joint_id - 1, 2],
                       c=color, s=20)  # Joint ID in colormap is 1-based

# Function to visualize a static frame
def show_static_frame(dataset, skeleton_connections, colormap, sample_idx=0, frame_idx=0):
    """
    Display a static frame from the dataset.

    Args:
        dataset: SHREC22_data dataset object.
        skeleton_connections: List of tuples defining the skeleton connections.
        colormap: Dictionary assigning colors to joints.
        sample_idx: Index of the sample to visualize.
        frame_idx: Frame index to display.
    """
    gesture_data = dataset[sample_idx]['Sequence']
    gesture_data = gesture_data.numpy()  # Convert to numpy for visualization

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    joint_positions = gesture_data[frame_idx]  # Single frame
    plot_skeleton(ax, joint_positions, skeleton_connections, colormap, frame_idx)
    plt.show()

# Function to animate the skeleton
def animate_skeleton(dataset, skeleton_connections, colormap, sample_idx=0, interval=100, anim_file_path = None):
    """
    Animate a gesture sequence.

    Args:
        dataset: SHREC22_data dataset object.
        skeleton_connections: List of tuples defining the skeleton connections.
        colormap: Dictionary assigning colors to joints.
        sample_idx: Index of the sample to visualize.
        interval: Interval (ms) between frames.
    """
    gesture_data = dataset[sample_idx]['Sequence']
    gesture_data = gesture_data.numpy()  # Convert to numpy for animation

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame_idx):
        joint_positions = gesture_data[frame_idx]
        plot_skeleton(ax, joint_positions, skeleton_connections, colormap, frame_idx)

    anim = FuncAnimation(fig, update, frames=gesture_data.shape[0], interval=interval, repeat=True)
    if anim_file_path is not None:
        anim.save(anim_file_path, writer='pillow', fps=10)
    
    plt.show()

# Function to visualize samples from the DataLoader
def visualize_sample(data_loader, skeleton_connections, colormap, batch_idx=0, sample_in_batch=0, static_frame_idx=0, animate=True):
    """
    Visualize a sample from a DataLoader.

    Args:
        data_loader: PyTorch DataLoader object for SHREC22_data.
        skeleton_connections: List of tuples defining the skeleton connections.
        colormap: Dictionary assigning colors to joints.
        batch_idx: Index of the batch to visualize.
        sample_in_batch: Index of the sample within the batch.
        static_frame_idx: Frame index for static visualization.
        animate: Whether to animate the gesture sequence.
    """
    # Fetch one batch
    for batch_idx_counter, (gesture_data_batch, _) in enumerate(data_loader):
        if batch_idx_counter == batch_idx:
            gesture_data = gesture_data_batch[sample_in_batch].numpy()  # Single sample in batch
            break

    print(f"Visualizing Batch {batch_idx}, Sample {sample_in_batch}")
    print(f"Gesture Data Shape: {gesture_data.shape}")  # Shape: (Frames, 26, 3)

    # Static frame visualization
    print("Showing a static frame...")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_skeleton(ax, gesture_data[static_frame_idx], skeleton_connections, colormap, frame_idx=static_frame_idx)
    plt.show()

    # Animation
    if animate:
        print("Animating the skeleton sequence...")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        def update(frame_idx):
            plot_skeleton(ax, gesture_data[frame_idx], skeleton_connections, colormap, frame_idx)

        anim = FuncAnimation(fig, update, frames=gesture_data.shape[0], interval=100, repeat=True)
        plt.show()
