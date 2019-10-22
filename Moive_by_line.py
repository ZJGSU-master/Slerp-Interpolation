from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy.io as sio
import os

models = ['default', 'just_save', 'save_mat']


class plot_human(object):
    def __init__(self, data, model='default', save_path='./result_video/'):

        # data [frames,line,line_quaternion]
        self.save_path = save_path
        self.data = data
        self.files = list(data.keys())
        self.nframes = data[self.files[0]].shape[0]
        self.save = False
        self.chain = [
            np.array([0, 1, 2, 3, 4, 5]),  # 左脚 SpineBase到左脚 6
            np.array([0, 6, 7, 8, 9, 10]),  # 右脚 SpineBase到右脚 6
            np.array([0, 12, 13, 14, 15]),  # SpineBase到头  5
            np.array([13, 17, 18, 19, 22, 19, 21]),  # 右手 右手臂  7
            np.array([13, 25, 26, 27, 30, 27, 29])  # 左手 左手臂  7
        ]
        self.chains27 = [
            [0, 1, 2, 3, 4, 5],  # right leg   6
            [6, 7, 8, 9, 10, 11],  # left leg   6
            [12, 13, 14, 15, 16],  # spine    5
            [17, 18, 19, 20, 21],  # left arm   5
            [22, 23, 24, 25, 26]  # right arm   5
        ]
        self.model = model
        if self.model == 'just_save':
            self.save = True

        self.fig = plt.figure()
        self.plt = {}
        plt.title("Interpolation", fontsize=18, fontweight='bold')
        plt.axis('off')
        nrows = (len(self.files) - 1) / 3 + 1
        ncols = 1 if len(self.files) == 1 else 3
        for i in range(len(self.files)):
            self.plt[self.files[i]] = self.fig.add_subplot(nrows, ncols, i + 1, projection='3d')
            self.plt[self.files[i]].set_xlim([-700, 700])
            self.plt[self.files[i]].set_ylim([-700, 700])
            self.plt[self.files[i]].set_zlim([-1000, 700])
            # self.plt[self.files[i]].set_xlabel('x')
            # self.plt[self.files[i]].set_ylabel('y')
            # self.plt[self.files[i]].set_zlabel('z')
            self.plt[self.files[i]].scats = []
            self.plt[self.files[i]].lns = []
            self.plt[self.files[i]].axes.view_init(azim=60, elev=10)
            self.plt[self.files[i]].axes.set_xticklabels([])
            self.plt[self.files[i]].axes.set_yticklabels([])
            self.plt[self.files[i]].axes.set_zticklabels([])
            # self.plt[self.files[i]].axis('off')
            plt.title(self.files[i], fontsize=12)

        self.linecolors = ['black', '#0780ea', '#5e01a0', '#2bba00', '#e08302', '#f94e3e']

    def update(self, frame):
        for i in range(len(self.files)):
            # for scat in self.plt[self.files[i]].scats:
            #     scat.remove()
            for ln in self.plt[self.files[i]].lns:
                self.plt[self.files[i]].lines.pop(0)
            self.plt[self.files[i]].scats = []
            self.plt[self.files[i]].lns = []
            if np.shape(self.data[self.files[i]])[1] == 32:
                for j in range(len(self.chain)):
                    self.plt[self.files[i]].lns.append(
                        self.plt[self.files[i]].plot3D(self.data[self.files[i]][frame, self.chain[j], 0],
                                                       self.data[self.files[i]][frame, self.chain[j], 1],
                                                       self.data[self.files[i]][frame, self.chain[j], 2], linewidth=2.0,
                                                       color=self.linecolors[i]))
            else:
                for j in range(len(self.chain)):
                    self.plt[self.files[i]].lns.append(
                        self.plt[self.files[i]].plot3D(self.data[self.files[i]][frame, self.chains27[j], 0],
                                                       self.data[self.files[i]][frame, self.chains27[j], 1],
                                                       self.data[self.files[i]][frame, self.chains27[j], 2],
                                                       linewidth=2.0,
                                                       color=self.linecolors[i]))

    def plot(self):
        ani = FuncAnimation(self.fig, self.update, frames=self.nframes, interval=30, repeat=False)
        if self.model == models[0]:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
            file_name = ''
            for file in self.files:
                file_name = file_name + "_" + str(file)
            # ani.save('Human.mp4', writer='ffmpeg', fps=25)
            if self.save:
                ani.save(self.save_path + file_name + '.mp4', writer='ffmpeg', fps=25)
            plt.show()
        elif self.model == 'just_save':
            two_motion_name = self.files[0] + '_' + self.files[-1]
            video_name = two_motion_name
            for file in self.files[1:-1]:
                video_name = video_name + "_" + str(file)
            ani.save(self.save_path + video_name + '.mp4', writer='ffmpeg', fps=25)

            if not os.path.exists(self.save_path + two_motion_name):
                os.mkdir(self.save_path + two_motion_name)

            for file in self.files[1:-1]:
                joint_xyz = self.data[file]
                sio.savemat(self.save_path + two_motion_name + '/' + str(file) + '.mat', {'joint_xyz': joint_xyz})
            plt.close(self.fig)
        elif self.model == 'save_mat':
            two_motion_name = self.files[0] + '_' + self.files[-1]

            if not os.path.exists(self.save_path + two_motion_name):
                os.mkdir(self.save_path + two_motion_name)

            for file in self.files[1:-1]:
                joint_xyz = self.data[file]
                sio.savemat(self.save_path + two_motion_name + '/' + str(file) + '.mat', {'joint_xyz': joint_xyz})
            plt.close(self.fig)

# if __name__ == '__main__':
# # plot = plot_human(data)
# # plot.plot()
