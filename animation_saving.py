import vpython
from vpython import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_vpython_animation(scene, filename='vpython_animation.mp4', fps=120, dpi=100):
    """
    Save a VPython scene as an MP4 video file
    
    Args:
    scene (vpython.canvas): VPython scene to be recorded
    filename (str): Output MP4 filename
    fps (int): Frames per second
    dpi (int): Resolution quality
    """
    # Create a figure with the same aspect ratio as the VPython scene
    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)
    ax.set_axis_off()
    
    # Collect frames
    frames = []
    
    def update_frame(frame_num):
        """
        Capture each frame of the VPython scene
        """
        # Clear previous frame
        ax.clear()
        ax.set_axis_off()
        
        # Capture the scene as an image
        img = scene.capture()
        
        # Display the image
        ax.imshow(img)
        frames.append([ax.imshow(img)])
        
        return frames
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update_frame, 
        frames=range(len(x1)),  # Use the length of your data
        interval=1000/fps,  # Interval between frames
        blit=True
    )
    
    # Save animation
    anim.save(
        filename, 
        fps=fps, 
        extra_args=['-vcodec', 'libx264'],
        writer='ffmpeg'
    )
    
    plt.close(fig)
    print(f"Animation saved as {filename}")

# Load data
x1, y1, z1 = np.load('./data/3Dpen.npy')

# Create VPython scene
scene = canvas(width=800, height=600, background=color.white)

# Create objects
ball1 = sphere(color=color.green, radius=0.3, make_trail=True, retain=20)
rod1 = cylinder(pos=vector(0,0,0), axis=vector(0,0,0), radius=0.05)
base = box(pos=vector(0,-11,0), axis=vector(1,0,0), size=vector(10,0.5,10))

print(len(x1))
# Animation function
def animate_pendulum():
    i = 0
    while i < len(x1):
        rate(120)
        ball1.pos = vector(x1[i], z1[i], y1[i])
        rod1.axis = vector(x1[i], z1[i], y1[i])
        i += 1
    
    # Return the final scene for saving
    return scene

# Run animation and save
final_scene = animate_pendulum()
save_vpython_animation(final_scene, 'pendulum_animation.mp4')