import numpy as np
import matplotlib
import scipy.io as sio
import os
import math
from Moive_by_line import plot_human

chains32 = [
    np.array([0, 1, 2, 3, 4, 5]),  # 右脚 SpineBase到右脚
    np.array([0, 6, 7, 8, 9, 10]),  # 左脚 SpineBase到左脚
    np.array([0, 12, 13, 14, 15]),  # SpineBase到头   5
    np.array([13, 17, 18, 19, 22, 19, 21]),  # 左手 左手臂
    np.array([13, 25, 26, 27, 30, 27, 29])  # 右手 右手臂
]

frames_clip_start = 0
frames_clip_length = 375


# 正则化
def normalization(v):
    # v[a,b,c,d]
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2 + v[3] ** 2)
    return v / norm, norm


# Slerp interpolation
# reference from https://en.wikipedia.org/wiki/Slerp
def slerp(v0, v1, t_array, negative=False):
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    # if negative is True,  when dot<0, dot=-dot,v1=-v1.(find shortest path,regard v0 and v1 as two rotation)

    t_array = np.array(t_array)

    v0 = np.array(v0)
    v1 = np.array(v1)

    v0, v0_norm = normalization(v0)
    v1, v1_norm = normalization(v1)
    dot = np.sum(v0 * v1)

    if (negative is True) and (dot < 0.0):
        v1 = -v1
        dot = -dot

    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        # If the inputs are too close for comfort, linearly interpolate
        result = v0[np.newaxis, :] * v0_norm + t_array[:, np.newaxis] * (v1 * v1_norm - v0 * v0_norm)[np.newaxis, :]
        return result

    theta = np.arccos(dot)  # 取反余弦
    sin_theta = np.sin(theta)

    t_theta = theta * t_array
    sin_t_theta = np.sin(t_theta)

    s0 = np.cos(t_theta) - dot * sin_t_theta / sin_theta  # == sin(theta-t_theta)/sin(theta)
    s1 = sin_t_theta / sin_theta
    return ((s0[:, np.newaxis] * v0[np.newaxis, :] * v0_norm) + (s1[:, np.newaxis] * v1[np.newaxis, :]) * v1_norm)


# 画图
def line_to_point(quaternion, title="demo", show=False):
    # quaternion[frames_clip_length,nbones,4]
    quaternion_shape = np.shape(quaternion)
    data_plot = np.zeros(shape=[frames_clip_length, 32, 3])
    chains = chains32

    for frame in range(frames_clip_length):
        count = 0
        for chain in chains:
            for i in range(1, len(chain)):
                data_plot[frame][chain[i]] = data_plot[frame][chain[i - 1]] - quaternion[frame][count][1:]
                count = count + 1
    return data_plot


def interpolation_between_two_action32(file_name_1, file_name_2, file_path='./data/h3.6m/Train/train_xyz/',
                                       model='default'):
    data = {}
    data_quaternion = {}  # 以四元数的形式存放
    data_slerp = {}  # 存放展示的数据
    slerp_t = [0.2, 0.4, 0.6, 0.8]
    bone_number = 26
    # load data
    files = [file_name_1, file_name_2]
    item_shape = [32, 32]
    for i, file in enumerate(files):
        datafile = sio.loadmat(file_path + file)
        data[file] = datafile[list(datafile.keys())[3]]
        item_shape[i] = np.shape(data[file])[1]

    if item_shape[0] == 27 or item_shape[1] == 27:
        return item_shape[0], item_shape[1]
    # data to quaternion
    # data is dictionary
    for item in data:
        chains = chains32
        one_frame_quaternion = list()
        for i in range(frames_clip_start, frames_clip_start + frames_clip_length):
            # temp = 一帧所包含的四元数
            temp = list()
            for chain in chains:
                for j in range(len(chain) - 1):
                    temp.append((0,
                                 data[item][i][chain[j]][0] - data[item][i][chain[j + 1]][0],
                                 data[item][i][chain[j]][1] - data[item][i][chain[j + 1]][1],
                                 data[item][i][chain[j]][2] - data[item][i][chain[j + 1]][2]))
            one_frame_quaternion.append(temp)
        data_quaternion[item] = np.array(one_frame_quaternion)
    # ————————————————初始动作————————————————
    data_slerp[files[0]] = line_to_point(data_quaternion[files[0]])
    #  ———————————————————————————————————————
    # 两组动作之间插值
    temp1 = []
    for i in range(frames_clip_length):
        temp2 = []
        for j in range(bone_number):
            a = np.array(
                slerp(data_quaternion[files[0]][i][j], data_quaternion[files[1]][i][j], slerp_t, negative=False))
            temp2.append(a)
        temp1.append(temp2)

    # 将插值后的数据放入画图的数据中
    for index, one_list in enumerate(np.split(np.array(temp1), len(slerp_t), axis=2)):
        one_list = np.squeeze(one_list)
        data_quaternion[slerp_t[index]] = one_list
        data_slerp[slerp_t[index]] = line_to_point(data_quaternion[slerp_t[index]])
    # ————————————————最终动作————————————————
    data_slerp[files[1]] = line_to_point(data_quaternion[files[1]])
    # ———————————————————————————————————————
    plot = plot_human(data_slerp, model=model, save_path='./result_32/')
    plot.plot()
    return item_shape[0], item_shape[1]

# test
if __name__ == '__main__':
    files = ['S11_directions_1_xyz', 'S8_smoking_2_xyz']
    print(interpolation_between_two_action32(files[0], files[1]))
