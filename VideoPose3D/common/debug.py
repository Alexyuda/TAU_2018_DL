import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skeleton import Skeleton

h36m_skeleton = Skeleton(parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
       joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
       joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31])

def scatter_3d_points(keypoints,skeleton = h36m_skeleton):
    matplotlib.pyplot.switch_backend('TKAgg')

    try:
         parents = skeleton.parents()
    except:
        parents = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for n, value in enumerate(keypoints):
        [x,y,z] = value
        ax.scatter(x, y, z, c = 'b', marker='o')
        ax.text(x,y,z, str(n), size=10, zorder=1, color='k')

    for j, j_parent in enumerate(parents):
        if j_parent == -1:
            continue

        if len(parents) == keypoints.shape[0]:
            # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
            ax.plot([keypoints[j, 0], keypoints[j_parent, 0]],
                                       [keypoints[j, 1], keypoints[j_parent, 1]],
                                       [keypoints[j, 2], keypoints[j_parent, 2]], zdir='z', c='red')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.axis('equal')
    cid = plt.gcf().canvas.mpl_connect('key_press_event', quit_figure)
    plt.show()


def quit_figure(event):
    if event.key == 'q':
        plt.close(event.canvas.figure)
        matplotlib.pyplot.switch_backend('Agg')
