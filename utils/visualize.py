from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt


## define skeleton joint connections
skeleton_connections = [(0, 2), (0, 6), (0, 1), (0, 16),
                       (0, 21), (2, 3), (3, 4), (4, 5),
                       (6, 7), (7, 8), (8, 9), (9, 10), 
                       (1, 11), (11, 12), (12, 13), 
                       (13, 14), (14, 15), (16, 17),
                       (17, 18), (18, 19), (19, 20), 
                       (21, 22), (22, 23), (23, 24), (24, 25)]

## associate the same color to different joints in the same finger
colormap = {'r': [4, 5, 6], 
             'b': [8, 9, 10, 11],
             'g': [13, 14, 15, 16],
             'y': [18, 19, 20, 21],
             'm': [23, 24, 25, 26],
             'c': [1, 2, 3, 7, 12, 17, 22]}
joints_colors = {}
for c, joints in colormap.items():
    for joint in joints:
        joints_colors[joint-1] = c
        
        
def draw_skeleton_data(frame, skeleton_data, skeleton_connections, plot_connections=True):
    '''
    draws skeleton data given the links connections
    '''
    plt.clf()  # Clear the previous plot

    ## Plot the skeleton connections
    for connection in skeleton_connections:
        src = skeleton_data[frame, connection[0]]
        dst = skeleton_data[frame, connection[1]]
        if plot_connections:
            plt.plot([src[0], dst[0]], [src[1], dst[1]], c=joints_colors[connection[0]], marker='o', alpha=0.6)
        plt.scatter([src[0]], [src[1]], c=joints_colors[connection[0]], marker='o', alpha=0.6)
    plt.title('gesture')
    plt.axis('off')



def show_image(tensor_image):

    if len(tensor_image.shape) == 4:  
        tensor_image = tensor_image[0]  

    image = tensor_image.detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)  

    plt.imshow(image)
    plt.axis('off')  
    plt.show()


from matplotlib.animation import FuncAnimation

def draw_skeleton_data(frame, skeleton_data, skeleton_connections, plot_connections=True):
    '''
    draws skeleton data given the links connections
    '''
    plt.clf()  # Clear the previous plot

    ## Plot the skeleton connections
    for connection in skeleton_connections:
        src = skeleton_data[frame, connection[0]]
        dst = skeleton_data[frame, connection[1]]
        if plot_connections:
            plt.plot([src[0], dst[0]], [src[1], dst[1]], c=joints_colors[connection[0]], marker='o', alpha=0.6)
        plt.scatter([src[0]], [src[1]], c=joints_colors[connection[0]], marker='o', alpha=0.6)
    plt.title('gesture: ')
    plt.axis('off')

""" fig, ax = plt.subplots()
label = 1

skeleton_sequence = train_set[10]["Sequence"]
animation = FuncAnimation(fig, draw_skeleton_data, frames=T, fargs=(skeleton_sequence, skeleton_connections, label), interval=100)

## save the animation as a GIF

animation_file_path = "Desktop/test123/anc2.gif"
animation.save(animation_file_path, writer='pillow', fps=10)
plt.show() """