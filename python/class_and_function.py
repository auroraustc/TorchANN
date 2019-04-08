import numpy as np
import torch as tf
import os
import sys
import termios

class Parameters():
    def __init__(self):
        cutoff_1 = 0.0
        cutoff_2 = 0.0
        cutoff_3 = 0.0
        cutoff_max = 0.0
        N_types_all_frame = 0
        type_index_all_frame = []
        SEL_A_max = 0
        Nframes_tot = 0
        sym_coord_type = 1


def press_any_key_exit(msg):
    # 获取标准输入的描述符
    fd = sys.stdin.fileno()

    # 获取标准输入(终端)的设置
    old_ttyinfo = termios.tcgetattr(fd)

    # 配置终端
    new_ttyinfo = old_ttyinfo[:]

    # 使用非规范模式(索引3是c_lflag 也就是本地模式)
    new_ttyinfo[3] &= ~termios.ICANON
    # 关闭回显(输入不会被显示)
    new_ttyinfo[3] &= ~termios.ECHO

    # 输出信息
    sys.stdout.write(msg)
    sys.stdout.flush()
    # 使设置生效
    termios.tcsetattr(fd, termios.TCSANOW, new_ttyinfo)
    # 从终端读取
    os.read(fd, 7)

    # 还原终端设置
    termios.tcsetattr(fd, termios.TCSANOW, old_ttyinfo)

def read_parameters(parameters):
    if (type(parameters).__name__ != "Parameters"):
        print("type error in read_parameters:", type(parameters).__name__," is NOT a correct Parameters class")
        return 1
    parameters.cutoff_1 = 7.7
    parameters.cutoff_2 = 8.0
    parameters.cutoff_max = 8.0
    parameters.N_types_all_frame = 2
    parameters.type_index_all_frame = [0, 1]
    parameters.SEL_A_max = 200
    parameters.Nframes_tot = 120
    parameters.sym_coord_type = 1
    return 0

def reshape_to_frame_wise(source, n_atoms, parameters, flag):
    if (type(parameters).__name__ != "Parameters"):
        print("type error in reshape:", type(parameters).__name__," is NOT a correct Parameters class")
        return 1
    if (flag == 1):#force and coord
        interval = 3
    elif (flag == 2):#DeePMD type sym_coord
        interval = parameters.SEL_A_max * 4
    start_atom = 0
    end_atom = n_atoms[0] - 1
    source_Reshape = np.array([np.append([], [source[j] for j in range(start_atom * interval, (end_atom + 1) * interval)])], dtype = source.dtype)
    for i in range(1, parameters.Nframes_tot):
        start_atom = end_atom + 1
        end_atom = end_atom + n_atoms[i]
        source_Reshape = np.append(source_Reshape,
                                  [np.append([], [source[j] for j in range(start_atom * interval, (end_atom + 1) * interval)])],
                                  axis=0)
        """if (i == 5): #For debug and compare
            print(source_Reshape)
            break"""
    return source_Reshape

###Allso slow. Not used
def read_reshape_DeepMD(n_atoms, parameters):
    if (type(parameters).__name__ != "Parameters"):
        print("type error in reshape:", type(parameters).__name__," is NOT a correct Parameters class")
        return 1
    fp = open("./SYM_COORD.BIN", "rb")
    interval_per_atom = parameters.SEL_A_max * 4 * np.dtype(np.float64).itemsize
    start_atom = 0
    end_atom = n_atoms[0] - 1
    fp.seek(interval_per_atom * start_atom, os.SEEK_SET)
    tmp = fp.read(interval_per_atom * (end_atom - start_atom + 1))
    fp_out = open("./tmp", "wb")
    fp_out.write(tmp)
    fp_out.close()
    tmp_in = np.fromfile("./tmp", dtype = np.float64)
    source_Reshape = np.array([np.append([], tmp_in)])
    #print(source_Reshape)
    for i in range(1, parameters.Nframes_tot):
        start_atom = end_atom + 1
        end_atom = end_atom + n_atoms[i]
        fp.seek(interval_per_atom * start_atom, os.SEEK_SET)
        tmp = fp.read(interval_per_atom * (end_atom - start_atom + 1))
        fp_out = open("./tmp", "wb")
        fp_out.write(tmp)
        fp_out.close()
        tmp_in = np.fromfile("./tmp", dtype=np.float64)
        source_Reshape = np.append(source_Reshape, np.array([np.append([], tmp_in)]), axis = 0)
        """if (i == 5): #For debug and compare
            print(source_Reshape)
            break"""
    fp.close()
    return source_Reshape
