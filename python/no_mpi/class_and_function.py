import numpy as np
import torch as tf
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import termios
#from torch.autograd import Variable
#from torch.autograd import Variable
import json
#import torchvision.models as models
#from graphviz import Digraph
import re

tf.set_default_dtype(tf.float64)

class Parameters():
    def __init__(self):
        cutoff_1 = 0.0
        cutoff_2 = 0.0
        cutoff_3 = 0.0
        cutoff_max = 0.0
        N_types_all_frame = 0
        type_index_all_frame = []
        N_Atoms_max = 1
        SEL_A_max = 0
        Nframes_tot = 0
        sym_coord_type = 1
        N_sym_coord = 0

        batch_size = 1
        stop_epoch = 1
        num_filter_layer = 1
        filter_neuron = []
        axis_neuron = 1
        num_fitting_layer = 1
        fitting_neuron = []
        start_lr = 0.0005
        decay_steps = 100
        decay_epoch = 1
        decay_rate = 0.95
        start_pref_e = 1
        limit_pref_e = 400
        start_pref_f = 1000
        limit_pref_f = 1

        check_step = 1000
        output_epoch = 10
        save_epoch = 10
    def __str__(self):
        """
        print(">>> cutoff_1: %.6e" % self.cutoff_1)
        print(">>> cutoff_2: %.6e" % self.cutoff_2)
        print(">>> cutoff_3: %.6e" % self.cutoff_3)
        print(">>> cutoff_max: %.6e" % self.cutoff_max)
        print(">>> N_types_all_frame: %2d" % self.N_types_all_frame)
        print(">>> type_index_all_frame: ", self.type_index_all_frame)
        print(">>> SEL_A_max: %2d" % self.SEL_A_max)
        print(">>> Nframes_tot: %2d" % self.Nframes_tot)
        print(">>> sym_coord_type: %2d" % self.sym_coord_type)
        print(">>> batch_size: %2d" % self.batch_size)
        print(">>> epoch: %2d" % self.epoch)
        print(">>> num_filter_layer: %2d" % self.num_filter_layer)
        print(">>> filter_neuron: ", self.filter_neuron)
        print(">>> axis_neuron: %2d" % self.axis_neuron)
        print(">>> num_fitting_layer: %2d" % self.num_fitting_layer)
        print(">>> fitting_neuron: ", self.fitting_neuron)
        print(">>> start_lr: %.6e" % self.start_lr)
        print(">>> (Not used)decay_steps: %2d" % self.decay_steps)
        print(">>> decay_epoch: %2d" % self.decay_epoch)
        print(">>> decay_rate: %.6e" % self.decay_rate)
        print(">>> start_pref_e: %.6e" % self.start_pref_e)
        print(">>> limit_pref_e: %.6e" % self.limit_pref_e)
        print(">>> start_pref_f: %.6e" % self.start_pref_f)
        print(">>> limit_pref_f: %.6e" % self.limit_pref_f)
        """
        str_ = []
        str_ += (">>> cutoff_1: %.6e\n" % self.cutoff_1)
        str_ += (">>> cutoff_2: %.6e\n" % self.cutoff_2)
        str_ += (">>> cutoff_3: %.6e\n" % self.cutoff_3)
        str_ += (">>> cutoff_max: %.6e\n" % self.cutoff_max)
        str_ += (">>> N_types_all_frame: %2d\n" % self.N_types_all_frame)
        str_ += (">>> type_index_all_frame: ", self.type_index_all_frame, "\n")
        str_ += (">>> N_Atoms_max: %2d\n" % self.N_Atoms_max)
        str_ += (">>> SEL_A_max: %2d\n" % self.SEL_A_max)
        str_ += (">>> Nframes_tot: %2d\n" % self.Nframes_tot)
        str_ += (">>> sym_coord_type: %2d\n" % self.sym_coord_type)
        str_ += (">>> N_sym_coord: %2d\n" % self.N_sym_coord)
        str_ += (">>> batch_size: %2d\n" % self.batch_size)
        str_ += (">>> stop_epoch: %2d\n" % self.stop_epoch)
        str_ += (">>> num_filter_layer: %2d\n" % self.num_filter_layer)
        str_ += (">>> filter_neuron: ", self.filter_neuron, "\n")
        str_ += (">>> axis_neuron: %2d\n" % self.axis_neuron)
        str_ += (">>> num_fitting_layer: %2d\n" % self.num_fitting_layer)
        str_ += (">>> fitting_neuron: ", self.fitting_neuron, "\n")
        str_ += (">>> start_lr: %.6e\n" % self.start_lr)
        str_ += (">>> (Not used)decay_steps: %2d\n" % self.decay_steps)
        str_ += (">>> decay_epoch: %2d\n" % self.decay_epoch)
        str_ += (">>> decay_rate: %.6e\n" % self.decay_rate)
        str_ += (">>> start_pref_e: %.6e\n" % self.start_pref_e)
        str_ += (">>> limit_pref_e: %.6e\n" % self.limit_pref_e)
        str_ += (">>> start_pref_f: %.6e\n" % self.start_pref_f)
        str_ += (">>> limit_pref_f: %.6e\n" % self.limit_pref_f)

        str_ += (">>> check_step: %d\n" % self.check_step)
        str_ += (">>> output_epoch: %d\n" % self.output_epoch)
        str_ += (">>> save_epoch: %d\n" % self.save_epoch)
        str_ = ''.join(str(element) for element in str_)

        return str_

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            #return self.__dict__ == other.__dict__
            structure_unchanged = (self.filter_neuron == other.filter_neuron) and (
                        self.fitting_neuron == other.fitting_neuron) and (
                                              self.num_filter_layer == other.num_filter_layer) and (
                                              self.num_fitting_layer == other.num_fitting_layer) and (
                                              self.sym_coord_type == other.sym_coord_type) and (
                                              self.SEL_A_max == other.SEL_A_max)
            return structure_unchanged
        else:
            return False



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
    INPUT_FILE = open('ALL_PARAMS.json')
    INPUT_DATA = json.load(INPUT_FILE)
    parameters.cutoff_1 = INPUT_DATA['cutoff_1']
    parameters.cutoff_2 = INPUT_DATA['cutoff_2']
    parameters.cutoff_3 = INPUT_DATA['cutoff_3']
    parameters.cutoff_max = INPUT_DATA['cutoff_max']
    parameters.N_types_all_frame = INPUT_DATA['N_types_all_frame']
    parameters.type_index_all_frame = INPUT_DATA['type_index_all_frame']
    parameters.N_Atoms_max = INPUT_DATA['N_Atoms_max']
    parameters.SEL_A_max = INPUT_DATA['SEL_A_max']
    parameters.Nframes_tot = INPUT_DATA['Nframes_tot']
    parameters.sym_coord_type = INPUT_DATA['sym_coord_type']
###New add parameters
    parameters.N_sym_coord = INPUT_DATA['N_sym_coord']
    parameters.batch_size = INPUT_DATA['batch_size']
    parameters.stop_epoch = INPUT_DATA['stop_epoch']
    parameters.num_filter_layer = INPUT_DATA['num_filter_layer']
    parameters.filter_neuron = INPUT_DATA['filter_neuron']
    parameters.axis_neuron = INPUT_DATA['axis_neuron']
    parameters.num_fitting_layer = INPUT_DATA['num_fitting_layer']
    parameters.fitting_neuron = INPUT_DATA['fitting_neuron']
    parameters.start_lr = INPUT_DATA['start_lr']
    parameters.decay_steps = INPUT_DATA['decay_steps'] #abandoned
    parameters.decay_epoch = INPUT_DATA['decay_epoch']
    parameters.decay_rate = INPUT_DATA['decay_rate']
    parameters.start_pref_e = INPUT_DATA['start_pref_e']
    parameters.limit_pref_e = INPUT_DATA['limit_pref_e']
    parameters.start_pref_f = INPUT_DATA['start_pref_f']
    parameters.limit_pref_f = INPUT_DATA['limit_pref_f']

    parameters.check_step = INPUT_DATA['check_step']
    parameters.output_epoch = INPUT_DATA['output_epoch']
    parameters.save_epoch = INPUT_DATA['save_epoch']

    INPUT_FILE.close()
    return 0

def read_and_init_bin_file_all(parameters, default_dtype):
    COORD = np.fromfile("./COORD.BIN", dtype=np.float64)
    SYM_COORD = np.fromfile("./SYM_COORD.BIN", dtype=np.float64)
    ENERGY = np.fromfile("./ENERGY.BIN", dtype=np.float64)
    FORCE = np.fromfile("./FORCE.BIN", dtype=np.float64)
    TYPE = np.fromfile("./TYPE.BIN", dtype=np.int32)
    N_ATOMS = np.fromfile("./N_ATOMS.BIN", dtype=np.int32)
    NEI_IDX = np.fromfile("./NEI_IDX.BIN", dtype=np.int32)
    NEI_COORD = np.fromfile("./NEI_COORD.BIN", dtype=np.float64)
    SYM_COORD_DX = np.fromfile("./SYM_COORD_DX.BIN", dtype=np.float64)
    SYM_COORD_DY = np.fromfile("./SYM_COORD_DY.BIN", dtype=np.float64)
    SYM_COORD_DZ = np.fromfile("./SYM_COORD_DZ.BIN", dtype=np.float64)
    N_ATOMS_ORI = np.fromfile("./N_ATOMS_ORI.BIN", dtype=np.int32)
    NEI_TYPE = np.fromfile("./NEI_TYPE.BIN", dtype=np.int32)
    # print("Number of atoms aligned: ", N_ATOMS)
    # print(np.dtype(np.float64).itemsize)
    parameters.Nframes_tot = len(N_ATOMS)
    print("Total number of frames: ", parameters.Nframes_tot)
    print("Number of atoms aligned: ", N_ATOMS[0])
    # press_any_key_exit("Read complete. Press any key to continue\n")
    """Reshape COORD, FORCE as [Nfrmaes, N_Atoms_this_frame * 3] and SYM_COORD as [Nframes, N_Atoms_this_frame * SELA_max * 4]"""
    """DO NOT use np.reshape because it cannot deal with frames with different number of atoms."""
    """Use np.reshape now and DO NOT use the damn reshape_to_frame_wise functions because I have aligned the number of atoms in all frames"""
    # COORD_Reshape = reshape_to_frame_wise(COORD, N_ATOMS, parameters, 1)
    COORD_Reshape = np.reshape(COORD, (parameters.Nframes_tot, -1))
    print("COORD_Reshape: shape = ", COORD_Reshape.shape)  # , "\n", COORD_Reshape)
    # FORCE_Reshape = reshape_to_frame_wise(FORCE, N_ATOMS, parameters, 1)
    FORCE_Reshape = np.reshape(FORCE, (parameters.Nframes_tot, -1))
    print("FORCE_Reshape: shape = ", FORCE_Reshape.shape)  # , "\n", FORCE_Reshape)
    # SYM_COORD_Reshape = reshape_to_frame_wise(SYM_COORD, N_ATOMS, parameters, 2)
    SYM_COORD_Reshape = np.reshape(SYM_COORD, (parameters.Nframes_tot, -1))
    print("SYM_COORD_Reshape: shape = ", SYM_COORD_Reshape.shape)  # , "\n", SYM_COORD_Reshape)
    TYPE_Reshape = np.reshape(TYPE, (parameters.Nframes_tot, -1))
    print("TYPE_Reshape: shape = ", TYPE_Reshape.shape)
    NEI_IDX_Reshape = np.reshape(NEI_IDX, (parameters.Nframes_tot, -1))
    print("NEI_IDX_Reshape: shape = ", NEI_IDX_Reshape.shape)
    NEI_COORD_Reshape = np.reshape(NEI_COORD, (parameters.Nframes_tot, -1))
    print("NEI_COORD_Reshape: shape = ", NEI_COORD_Reshape.shape)
    SYM_COORD_DX_Reshape = np.reshape(SYM_COORD_DX, (parameters.Nframes_tot, -1))
    SYM_COORD_DY_Reshape = np.reshape(SYM_COORD_DY, (parameters.Nframes_tot, -1))
    SYM_COORD_DZ_Reshape = np.reshape(SYM_COORD_DZ, (parameters.Nframes_tot, -1))
    print("SYM_COORD_DX_Reshape: shape = ", SYM_COORD_DX_Reshape.shape)
    print("SYM_COORD_DY_Reshape: shape = ", SYM_COORD_DY_Reshape.shape)
    print("SYM_COORD_DZ_Reshape: shape = ", SYM_COORD_DZ_Reshape.shape)
    NEI_TYPE_Reshape = np.reshape(NEI_TYPE, (parameters.Nframes_tot, -1))
    print("NEI_TYPE_Reshape: shape = ", NEI_TYPE.shape)

    COORD_Reshape_tf = tf.from_numpy(COORD_Reshape).type(default_dtype)
    SYM_COORD_Reshape_tf = tf.from_numpy(SYM_COORD_Reshape).type(default_dtype)
    ENERGY_tf = tf.from_numpy(ENERGY).type(default_dtype)
    FORCE_Reshape_tf = tf.from_numpy(FORCE_Reshape).type(default_dtype)
    N_ATOMS_tf = tf.from_numpy(N_ATOMS)
    TYPE_Reshape_tf = tf.from_numpy(TYPE_Reshape)
    NEI_IDX_Reshape_tf = tf.from_numpy(NEI_IDX_Reshape).long()
    NEI_COORD_Reshape_tf = tf.from_numpy(NEI_COORD_Reshape).type(default_dtype)
    #NEI_COORD_Reshape_tf = tf.zeros(len(SYM_COORD_Reshape_tf))
    FRAME_IDX_tf = tf.ones(len(COORD_Reshape_tf), dtype=tf.int32)
    for i in range(len(FRAME_IDX_tf)):
        FRAME_IDX_tf[i] = i
    SYM_COORD_DX_Reshape_tf = tf.from_numpy(SYM_COORD_DX_Reshape).type(default_dtype)
    SYM_COORD_DY_Reshape_tf = tf.from_numpy(SYM_COORD_DY_Reshape).type(default_dtype)
    SYM_COORD_DZ_Reshape_tf = tf.from_numpy(SYM_COORD_DZ_Reshape).type(default_dtype)
    N_ATOMS_ORI_tf = tf.from_numpy(N_ATOMS_ORI)
    NEI_TYPE_Reshape_tf = tf.from_numpy(NEI_TYPE_Reshape)

    return COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf, \
           NEI_IDX_Reshape_tf, NEI_COORD_Reshape_tf, FRAME_IDX_tf, SYM_COORD_DX_Reshape_tf, SYM_COORD_DY_Reshape_tf, \
           SYM_COORD_DZ_Reshape_tf, N_ATOMS_ORI_tf, NEI_TYPE_Reshape_tf

def read_and_init_bin_file(parameters, default_dtype):
    COORD = np.fromfile("./COORD.BIN", dtype=np.float64)
    #SYM_COORD = np.fromfile("./SYM_COORD.BIN", dtype=np.float64)
    ENERGY = np.fromfile("./ENERGY.BIN", dtype=np.float64)
    FORCE = np.fromfile("./FORCE.BIN", dtype=np.float64)
    TYPE = np.fromfile("./TYPE.BIN", dtype=np.int32)
    N_ATOMS = np.fromfile("./N_ATOMS.BIN", dtype=np.int32)
    NEI_IDX = np.fromfile("./NEI_IDX.BIN", dtype=np.int32)
    NEI_COORD = np.fromfile("./NEI_COORD.BIN", dtype=np.float64)
    #SYM_COORD_DX = np.fromfile("./SYM_COORD_DX.BIN", dtype=np.float64)
    #SYM_COORD_DY = np.fromfile("./SYM_COORD_DY.BIN", dtype=np.float64)
    #SYM_COORD_DZ = np.fromfile("./SYM_COORD_DZ.BIN", dtype=np.float64)
    N_ATOMS_ORI = np.fromfile("./N_ATOMS_ORI.BIN", dtype=np.int32)
    NEI_TYPE = np.fromfile("./NEI_TYPE.BIN", dtype=np.int32)
    # print("Number of atoms aligned: ", N_ATOMS)
    # print(np.dtype(np.float64).itemsize)
    parameters.Nframes_tot = len(N_ATOMS)
    print("Total number of frames: ", parameters.Nframes_tot)
    print("Number of atoms aligned: ", N_ATOMS[0])
    # press_any_key_exit("Read complete. Press any key to continue\n")
    """Reshape COORD, FORCE as [Nfrmaes, N_Atoms_this_frame * 3] and SYM_COORD as [Nframes, N_Atoms_this_frame * SELA_max * 4]"""
    """DO NOT use np.reshape because it cannot deal with frames with different number of atoms."""
    """Use np.reshape now and DO NOT use the damn reshape_to_frame_wise functions because I have aligned the number of atoms in all frames"""
    # COORD_Reshape = reshape_to_frame_wise(COORD, N_ATOMS, parameters, 1)
    COORD_Reshape = np.reshape(COORD, (parameters.Nframes_tot, -1))
    print("COORD_Reshape: shape = ", COORD_Reshape.shape)  # , "\n", COORD_Reshape)
    # FORCE_Reshape = reshape_to_frame_wise(FORCE, N_ATOMS, parameters, 1)
    FORCE_Reshape = np.reshape(FORCE, (parameters.Nframes_tot, -1))
    print("FORCE_Reshape: shape = ", FORCE_Reshape.shape)  # , "\n", FORCE_Reshape)
    # SYM_COORD_Reshape = reshape_to_frame_wise(SYM_COORD, N_ATOMS, parameters, 2)
    #SYM_COORD_Reshape = np.reshape(SYM_COORD, (parameters.Nframes_tot, -1))
    #print("SYM_COORD_Reshape: shape = ", SYM_COORD_Reshape.shape)  # , "\n", SYM_COORD_Reshape)
    TYPE_Reshape = np.reshape(TYPE, (parameters.Nframes_tot, -1))
    print("TYPE_Reshape: shape = ", TYPE_Reshape.shape)
    NEI_IDX_Reshape = np.reshape(NEI_IDX, (parameters.Nframes_tot, -1))
    print("NEI_IDX_Reshape: shape = ", NEI_IDX_Reshape.shape)
    NEI_COORD_Reshape = np.reshape(NEI_COORD, (parameters.Nframes_tot, -1))
    print("NEI_COORD_Reshape: shape = ", NEI_COORD_Reshape.shape)
    #SYM_COORD_DX_Reshape = np.reshape(SYM_COORD_DX, (parameters.Nframes_tot, -1))
    #SYM_COORD_DY_Reshape = np.reshape(SYM_COORD_DY, (parameters.Nframes_tot, -1))
    #SYM_COORD_DZ_Reshape = np.reshape(SYM_COORD_DZ, (parameters.Nframes_tot, -1))
    #print("SYM_COORD_DX_Reshape: shape = ", SYM_COORD_DX_Reshape.shape)
    #print("SYM_COORD_DY_Reshape: shape = ", SYM_COORD_DY_Reshape.shape)
    #print("SYM_COORD_DZ_Reshape: shape = ", SYM_COORD_DZ_Reshape.shape)
    NEI_TYPE_Reshape = np.reshape(NEI_TYPE, (parameters.Nframes_tot, -1))
    print("NEI_TYPE_Reshape: shape = ", NEI_TYPE.shape)

    COORD_Reshape_tf = tf.from_numpy(COORD_Reshape).type(default_dtype)
    SYM_COORD_Reshape_tf = tf.zeros(len(COORD_Reshape_tf))#tf.from_numpy(SYM_COORD_Reshape).type(default_dtype)
    ENERGY_tf = tf.from_numpy(ENERGY).type(default_dtype)
    FORCE_Reshape_tf = tf.from_numpy(FORCE_Reshape).type(default_dtype)
    N_ATOMS_tf = tf.from_numpy(N_ATOMS)
    TYPE_Reshape_tf = tf.from_numpy(TYPE_Reshape)
    NEI_IDX_Reshape_tf = tf.from_numpy(NEI_IDX_Reshape).long()
    NEI_COORD_Reshape_tf = tf.from_numpy(NEI_COORD_Reshape).type(default_dtype)
    #NEI_COORD_Reshape_tf = tf.zeros(len(SYM_COORD_Reshape_tf))
    FRAME_IDX_tf = tf.ones(len(COORD_Reshape_tf), dtype=tf.int32)
    for i in range(len(FRAME_IDX_tf)):
        FRAME_IDX_tf[i] = i
    SYM_COORD_DX_Reshape_tf = tf.zeros(len(COORD_Reshape_tf))#tf.from_numpy(SYM_COORD_DX_Reshape).type(default_dtype)
    SYM_COORD_DY_Reshape_tf = tf.zeros(len(COORD_Reshape_tf))#tf.from_numpy(SYM_COORD_DY_Reshape).type(default_dtype)
    SYM_COORD_DZ_Reshape_tf = tf.zeros(len(COORD_Reshape_tf))#tf.from_numpy(SYM_COORD_DZ_Reshape).type(default_dtype)
    N_ATOMS_ORI_tf = tf.from_numpy(N_ATOMS_ORI)
    NEI_TYPE_Reshape_tf = tf.from_numpy(NEI_TYPE_Reshape)

    return COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf, \
           NEI_IDX_Reshape_tf, NEI_COORD_Reshape_tf, FRAME_IDX_tf, SYM_COORD_DX_Reshape_tf, SYM_COORD_DY_Reshape_tf, \
           SYM_COORD_DZ_Reshape_tf, N_ATOMS_ORI_tf, NEI_TYPE_Reshape_tf


class one_batch_net(nn.Module):

    """def __init__(self, parameters, mean_init):
        super(one_batch_net, self).__init__()
        #self.batch_norm = nn.BatchNorm1d(1)
        self.filter_input = nn.ModuleList()
        self.filter_hidden = nn.ModuleList()
        self.fitting_input = nn.ModuleList()
        self.fitting_hidden = nn.ModuleList()
        self.fitting_out = nn.ModuleList()
        for type_idx in range(len(parameters.type_index_all_frame)):

            self.filter_input.append(nn.Linear(1, parameters.filter_neuron[0]))
            self.fitting_input.append(nn.Linear(parameters.axis_neuron * parameters.filter_neuron[len(parameters.filter_neuron) - 1],
                               parameters.fitting_neuron[0]))
            self.fitting_out.append(nn.Linear(parameters.fitting_neuron[len(parameters.fitting_neuron) - 1], 1))
            self.filter_hidden.append(nn.ModuleList())
            self.fitting_hidden.append(nn.ModuleList())

            #self.filter_input[type_idx].bias.data.normal_(mean = mean_init[type_idx], std = 1)
            #self.fitting_input[type_idx].bias.data.normal_(mean = mean_init[type_idx], std = 1)
            self.filter_input[type_idx].bias.data.normal_(mean = 0, std = 1)
            self.fitting_input[type_idx].bias.data.normal_(mean = 0, std = 1)
            self.fitting_out[type_idx].bias.data.normal_(mean = mean_init[type_idx], std = 1)


        for type_idx in range(len(parameters.type_index_all_frame)):

            for hidden_idx in range(len(parameters.filter_neuron) - 1):
                self.filter_hidden[type_idx].append(nn.Linear(parameters.filter_neuron[hidden_idx],
                                         parameters.filter_neuron[hidden_idx + 1]))

                #self.filter_hidden[type_idx][hidden_idx].bias.data.normal_(mean = mean_init[type_idx], std = 1)
                self.filter_hidden[type_idx][hidden_idx].bias.data.normal_(mean = 0, std = 1)

            for hidden_idx in range(len(parameters.fitting_neuron) - 1):
                self.fitting_hidden[type_idx].append(nn.Linear(parameters.fitting_neuron[hidden_idx],
                                         parameters.fitting_neuron[hidden_idx + 1]))

                #self.fitting_hidden[type_idx][hidden_idx].bias.data.normal_(mean = mean_init[type_idx], std = 1)
                self.fitting_hidden[type_idx][hidden_idx].bias.data.normal_(mean = 0, std = 1)"""
    ###The above uses len(parameters.type_index_all_frame) filter nets. In DeePMD,
    ###len(parameters.type_index_all_frame) * len(parameters.type_index_all_frame) filter nets are used

    def __init__(self, parameters, mean_init):
        super(one_batch_net, self).__init__()
        # self.batch_norm = nn.BatchNorm1d(1)
        self.batch_norm = nn.BatchNorm1d(parameters.N_Atoms_max)
        self.filter_input = nn.ModuleList()
        self.filter_hidden = nn.ModuleList()
        self.fitting_input = nn.ModuleList()
        self.fitting_hidden = nn.ModuleList()
        self.fitting_out = nn.ModuleList()
        #fitting net: len(parameters.type_index_all_frame)
        for type_idx in range(len(parameters.type_index_all_frame)):
            self.fitting_input.append(
                nn.Linear(parameters.axis_neuron * parameters.filter_neuron[len(parameters.filter_neuron) - 1],
                          parameters.fitting_neuron[0]))
            self.fitting_out.append(nn.Linear(parameters.fitting_neuron[len(parameters.fitting_neuron) - 1], 1))
            self.fitting_hidden.append(nn.ModuleList())

            self.fitting_input[type_idx].bias.data.normal_(mean=0, std=1)
            self.fitting_out[type_idx].bias.data.normal_(mean=mean_init[type_idx], std=1)

        for type_idx in range(len(parameters.type_index_all_frame)):
            for hidden_idx in range(len(parameters.fitting_neuron) - 1):
                self.fitting_hidden[type_idx].append(nn.Linear(parameters.fitting_neuron[hidden_idx],
                                                               parameters.fitting_neuron[hidden_idx + 1]))

                self.fitting_hidden[type_idx][hidden_idx].bias.data.normal_(mean=0, std=1)

        #filter net: len(parameters.type_index_all_frame) * len(parameters.type_index_all_frame)
        for type_idx in range(len(parameters.type_index_all_frame)):
            self.filter_input.append(nn.ModuleList())
            self.filter_hidden.append(nn.ModuleList())

        for type_idx in range(len(parameters.type_index_all_frame)):
            for nei_type_idx in range(len(parameters.type_index_all_frame)):
                self.filter_input[type_idx].append(nn.Linear(1, parameters.filter_neuron[0]))
                self.filter_hidden[type_idx].append(nn.ModuleList())

                self.filter_input[type_idx][nei_type_idx].bias.data.normal_(mean=0, std=1)
                for hidden_idx in range(len(parameters.filter_neuron) - 1):
                    self.filter_hidden[type_idx][nei_type_idx].append(nn.Linear(parameters.filter_neuron[hidden_idx],
                                                                  parameters.filter_neuron[hidden_idx + 1]))

                    self.filter_hidden[type_idx][nei_type_idx][hidden_idx].bias.data.normal_(mean=0, std=1)


    def forward(self, data_cur, parameters, std, avg, use_std_avg, device, comput_descrpt_and_deriv_module) : #cur mean current batch
        data_cur[1], data_cur[9], data_cur[10], data_cur[11] = comput_descrpt_and_deriv_module.calc_descrpt_and_deriv_DPMD(
            data_cur[0], data_cur[7], data_cur[6], len(data_cur[0]), parameters.N_Atoms_max, parameters.SEL_A_max,
            parameters.cutoff_1, parameters.cutoff_2)[0:4]

        SYM_COORD_Reshape_tf_cur = data_cur[1]
        N_ATOMS_tf_cur = data_cur[4]
        SYM_COORD_Reshape_tf_cur_Reshape = tf.reshape(data_cur[1], \
                                                      (len(SYM_COORD_Reshape_tf_cur), N_ATOMS_tf_cur[0], \
                                                       parameters.SEL_A_max, 4))
        SYM_COORD_DX_Reshape_tf_cur_Reshape = tf.reshape(data_cur[9], SYM_COORD_Reshape_tf_cur_Reshape.shape)
        SYM_COORD_DY_Reshape_tf_cur_Reshape = tf.reshape(data_cur[10], SYM_COORD_Reshape_tf_cur_Reshape.shape)
        SYM_COORD_DZ_Reshape_tf_cur_Reshape = tf.reshape(data_cur[11], SYM_COORD_Reshape_tf_cur_Reshape.shape)
        NEI_IDX_Reshape_tf_cur = tf.reshape(data_cur[6], (len(data_cur[6]), data_cur[4][0], parameters.SEL_A_max))
        NEI_TYPE_Reshape_tf_cur = tf.reshape(data_cur[13], (len(data_cur[6]), data_cur[4][0], parameters.SEL_A_max))
        #NEI_COORD_Reshape_tf_cur = tf.reshape(data_cur[7], (len(data_cur[6]), data_cur[4][0], parameters.SEL_A_max, 3))
        E_cur_batch = tf.zeros(len(SYM_COORD_Reshape_tf_cur), device = device)
        E_cur_batch_atom_wise = tf.zeros((len(data_cur[1]), data_cur[4][0]), device=device).reshape(-1, )
        ###DO NOT forget to multiply -1 for F!!!
        F_cur_batch = tf.zeros((len(SYM_COORD_Reshape_tf_cur), data_cur[4][0], 3), device=device)
        std_ = tf.zeros((1, parameters.N_types_all_frame, 4), device = device)
        avg_ = tf.zeros((1, parameters.N_types_all_frame, 4), device = device)
        for type_idx in range(parameters.N_types_all_frame):
            F_cur_batch_as_center_atom = tf.zeros((len(data_cur[1]) * data_cur[4][0], 3), device = device)
            F_cur_batch_as_nei_atom = tf.zeros((len(data_cur[1]) * parameters.SEL_A_max, 3), device=device)
            type_idx_cur_type = (data_cur[5]==parameters.type_index_all_frame[type_idx]).nonzero()
            offset_array_frame = type_idx_cur_type.narrow(1, 0, 1)
            type_idx_cur_type = (
                        type_idx_cur_type.narrow(1, 0, 1) * data_cur[4][0].long() + type_idx_cur_type.narrow(1, 1, 1)).reshape(
                -1, )
            NEI_IDX_Reshape_tf_cur_cur_type = tf.index_select(
                NEI_IDX_Reshape_tf_cur.reshape(len(data_cur[1]) * data_cur[4][0], parameters.SEL_A_max), 0,
                type_idx_cur_type)
            NEI_TYPE_Reshape_tf_cur_cur_type = tf.index_select(
                NEI_TYPE_Reshape_tf_cur.reshape(len(data_cur[1]) * data_cur[4][0], parameters.SEL_A_max), 0,
                type_idx_cur_type)
            SYM_COORD_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_Reshape_tf_cur_Reshape.reshape(len(data_cur[1]) * data_cur[4][0], parameters.SEL_A_max, 4), 0,
                type_idx_cur_type)
            if ((SYM_COORD_Reshape_tf_cur_Reshape_cur_type.shape)[0] == 0):
                continue
            SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_DX_Reshape_tf_cur_Reshape.reshape(len(data_cur[1]) * data_cur[4][0], parameters.SEL_A_max, 4),
                0,
                type_idx_cur_type)
            SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_DY_Reshape_tf_cur_Reshape.reshape(len(data_cur[1]) * data_cur[4][0], parameters.SEL_A_max, 4),
                0,
                type_idx_cur_type)
            SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_DZ_Reshape_tf_cur_Reshape.reshape(len(data_cur[1]) * data_cur[4][0], parameters.SEL_A_max, 4),
                0,
                type_idx_cur_type)
            #batch-norm
            if (use_std_avg == True):
                """SYM_COORD_Reshape_tf_cur_Reshape_cur_type = (SYM_COORD_Reshape_tf_cur_Reshape_cur_type - avg[
                    type_idx]) / std[type_idx]
                SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type / std[
                    type_idx]
                SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type / std[
                    type_idx]
                SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type / std[
                    type_idx]"""
                std_ = std
                avg_ = avg
                std_cur_type = std_[0][type_idx]
                avg_cur_type = avg_[0][type_idx]
            else:
                with tf.no_grad():
                    sji = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.narrow(2, 0, 1)
                    xyz_hat = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.narrow(2, 1, 3)
                    sji_avg = tf.mean(sji)
                    xyz_hat_avg = tf.mean(xyz_hat)
                    avg_unit = tf.cat((sji_avg.reshape(1, 1), xyz_hat_avg.expand(1, 3)), dim=1)
                    sji_avg2 = tf.mean(sji ** 2)
                    xyz_hat_avg2 = tf.mean(xyz_hat ** 2)
                    avg_unit2 = tf.cat((sji_avg2.reshape(1, 1), xyz_hat_avg2.expand(1, 3)), dim=1)
                    std_unit = tf.sqrt(avg_unit2 - avg_unit ** 2)
                    avg_cur_type = tf.cat((avg_unit[0][0].reshape(1, 1), tf.zeros((1, 3), device=device)), dim=1)
                    #avg_cur_type = avg_unit
                    std_cur_type = std_unit + 1E-16
                    std_[0][type_idx] = std_cur_type
                    avg_[0][type_idx] = avg_cur_type

            SYM_COORD_Reshape_tf_cur_Reshape_cur_type = (SYM_COORD_Reshape_tf_cur_Reshape_cur_type - avg_cur_type) / \
                                                        std_cur_type
            SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type / std_cur_type
            SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type / std_cur_type
            SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type / std_cur_type



            SYM_COORD_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.requires_grad_()

            #Calculate energy of these atoms in all frames
            SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.narrow(2, 0, 1)

            offset_array = tf.arange(0, (SYM_COORD_Reshape_tf_cur_Reshape_cur_type.shape)[0],
                                     device=device).reshape(-1, 1).expand(
                (SYM_COORD_Reshape_tf_cur_Reshape_cur_type.shape)[0], parameters.SEL_A_max).reshape(-1, ).to(device) * parameters.SEL_A_max

            #filter net
            ### len(parameters.type_index_all_frame) filter nets version
            """G_cur_type = tf.tanh(self.filter_input[type_idx](SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice))
            for filter_hidden_idx, filter_hidden_layer in enumerate(self.filter_hidden[type_idx]):
                G_cur_type = tf.tanh(filter_hidden_layer(G_cur_type))"""
            ### len(parameters.type_index_all_frame) * len(parameters.type_index_all_frame) filter nets version
            shape_input_tmp = SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice.shape
            G_cur_type = tf.zeros((shape_input_tmp[0] * shape_input_tmp[1], parameters.filter_neuron[-1]), device=device)
            for nei_type_idx in range(parameters.N_types_all_frame):
                nei_type_idx_cur_type = (NEI_TYPE_Reshape_tf_cur_cur_type.reshape(-1,) == parameters.type_index_all_frame[nei_type_idx]).nonzero().reshape(-1,)
                SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice_cur_nei_type = tf.index_select(
                    SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice.reshape(
                        SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice.shape[0] *
                        SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice.shape[1], 1), 0, nei_type_idx_cur_type)
                G_cur_type_cur_nei_type = tf.tanh(self.filter_input[type_idx][nei_type_idx](SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice_cur_nei_type))
                for filter_hidden_idx, filter_hidden_layer in enumerate(self.filter_hidden[type_idx][nei_type_idx]):
                    G_cur_type_cur_nei_type = tf.tanh(filter_hidden_layer(G_cur_type_cur_nei_type))
                nei_type_idx_cur_type_tmp = nei_type_idx_cur_type.reshape(-1, 1).expand(len(nei_type_idx_cur_type), parameters.filter_neuron[-1])
                G_cur_type.scatter_(0, nei_type_idx_cur_type_tmp, G_cur_type_cur_nei_type)
            G_cur_type = G_cur_type.reshape(shape_input_tmp[0], shape_input_tmp[1], parameters.filter_neuron[-1])



            #filter net end. Start to transformation
            RG_cur_type = tf.bmm((SYM_COORD_Reshape_tf_cur_Reshape_cur_type).transpose(1, 2), G_cur_type)
            RG_cur_type = RG_cur_type / (parameters.SEL_A_max)
            GRRG_cur_type = tf.bmm(RG_cur_type.transpose(1, 2), RG_cur_type.narrow(2, 0, parameters.axis_neuron))
            GRRG_cur_type = tf.reshape(GRRG_cur_type, ((GRRG_cur_type.shape)[0], parameters.filter_neuron[
                len(parameters.filter_neuron) - 1] * parameters.axis_neuron,))
            #fitting net
            E_cur_type = tf.tanh(self.fitting_input[type_idx](GRRG_cur_type))
            for fitting_hidden_idx, fitting_hidden_layer in enumerate(self.fitting_hidden[type_idx]):
                E_cur_type = tf.tanh(fitting_hidden_layer(E_cur_type))
            E_cur_type = (self.fitting_out[type_idx](E_cur_type))  # Final layer do not use activation function
            #fitting net end.
            E_cur_batch_atom_wise.scatter_(0, type_idx_cur_type, tf.reshape(E_cur_type, (-1,)))
            E_tot_cur_type = tf.sum(E_cur_type)
            D_E_D_SYM_cur_type = \
            tf.autograd.grad(E_tot_cur_type, SYM_COORD_Reshape_tf_cur_Reshape_cur_type, create_graph=True)[0]
            # Start to calculate force of this type. DO NOT forget to multiply -1 !!
            ##as center atom
            F_as_center_atom_curtype_x = tf.sum(
                tf.sum(D_E_D_SYM_cur_type * SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type, dim=1), dim=1)
            F_as_center_atom_curtype_y = tf.sum(
                tf.sum(D_E_D_SYM_cur_type * SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type, dim=1), dim=1)
            F_as_center_atom_curtype_z = tf.sum(
                tf.sum(D_E_D_SYM_cur_type * SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type, dim=1), dim=1)
            F_as_center_xyz_atom = tf.reshape(
                tf.cat((F_as_center_atom_curtype_x, F_as_center_atom_curtype_y, F_as_center_atom_curtype_z)),
                (3, -1)).transpose(0, 1)
            F_cur_batch_as_center_atom.scatter_(0, type_idx_cur_type.expand(3, len(type_idx_cur_type)).transpose(0, 1),
                                       F_as_center_xyz_atom)

            ##as nei atom
            ##For example, atom j is atom i's kth neighbour. Then:
            ##F_X_Atom_j += (-1) * tf.sum(D_E_D_SYM_cur_type[i][k] * SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type[i][k])
            ##F_Y_Atom_j += ......
            ##......
            ##Use index_select to select i and j; Use reshape and add offset to simulate looping over i

            offset_array = tf.arange(0, (D_E_D_SYM_cur_type.shape)[0], device=device).reshape(-1, 1).expand(
                (D_E_D_SYM_cur_type.shape)[0], parameters.SEL_A_max).reshape(-1, ).to(device) * parameters.SEL_A_max
            NEI_IDX_Reshape_tf_cur_cur_type_new = NEI_IDX_Reshape_tf_cur_cur_type.reshape(-1, ) + offset_array
            """### Here is D_E_D_SYM_cur_type[i][j] * SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type[i][k], 
            ### which is of course incorrect.
            D_E_D_SYM_cur_type_nei = tf.index_select(
                D_E_D_SYM_cur_type.reshape((D_E_D_SYM_cur_type.shape)[0] * (D_E_D_SYM_cur_type.shape)[1],
                                           (D_E_D_SYM_cur_type.shape)[2]), 0,
                NEI_IDX_Reshape_tf_cur_cur_type_new).reshape(D_E_D_SYM_cur_type.shape)"""
            D_E_D_SYM_cur_type_nei = D_E_D_SYM_cur_type
            SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type_nei = SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type.reshape(
                D_E_D_SYM_cur_type.shape)
            SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type_nei = SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type.reshape(
                D_E_D_SYM_cur_type.shape)
            SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type_nei = SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type.reshape(
                D_E_D_SYM_cur_type.shape)
            F_as_nei_atom_curtype_x = (-1.0) * tf.sum(
                D_E_D_SYM_cur_type_nei * SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type_nei, dim=2)
            F_as_nei_atom_curtype_y = (-1.0) * tf.sum(
                D_E_D_SYM_cur_type_nei * SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type_nei, dim=2)
            F_as_nei_atom_curtype_z = (-1.0) * tf.sum(
                D_E_D_SYM_cur_type_nei * SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type_nei, dim=2)
            F_as_nei_xyz_atom_cur_type = tf.cat(
                (F_as_nei_atom_curtype_x, F_as_nei_atom_curtype_y, F_as_nei_atom_curtype_z), dim=1).reshape(
                len(F_as_nei_atom_curtype_x), 3, parameters.SEL_A_max).transpose(1, 2)  # shape=(N_Atoms_cur_type, SEL_A_max, 3)
            F_cur_batch_as_nei_atom.scatter_add_(0, (
                        NEI_IDX_Reshape_tf_cur_cur_type + offset_array_frame * parameters.SEL_A_max).reshape(-1,
                                                                                                             1).expand(
                len(F_as_nei_atom_curtype_x) * parameters.SEL_A_max, 3), F_as_nei_xyz_atom_cur_type.reshape(
                len(F_as_nei_atom_curtype_x) * parameters.SEL_A_max, 3))

            F_cur_batch_as_center_atom = F_cur_batch_as_center_atom.reshape(len(data_cur[1]), data_cur[4][0], 3)
            F_cur_batch_as_nei_atom = F_cur_batch_as_nei_atom.reshape(len(data_cur[1]), parameters.SEL_A_max, 3).narrow(1, 0, data_cur[4][0])
            F_cur_batch = F_cur_batch + F_cur_batch_as_center_atom + F_cur_batch_as_nei_atom


        """
        for frame_idx in range(len(SYM_COORD_Reshape_tf_cur)):
            E_cur_frame = tf.zeros(1, device = device)
            E_cur_frame_atom_wise = tf.zeros(N_ATOMS_tf_cur[0], device = device)

            for type_idx in range(parameters.N_types_all_frame):
                F_cur_frame_dummy = tf.zeros((data_cur[4][0], 3), device = device)
                F_cur_frame_dummy_nei = tf.zeros((parameters.SEL_A_max, 3), device=device)
                type_index_cur_type = tf.reshape(
                    (data_cur[5][frame_idx] == parameters.type_index_all_frame[type_idx]).nonzero(), (-1,))
                NEI_IDX_Reshape_tf_cur_cur_type = tf.index_select(NEI_IDX_Reshape_tf_cur[frame_idx], 0, type_index_cur_type)
                SYM_COORD_Reshape_tf_cur_Reshape_cur_type = tf.index_select(SYM_COORD_Reshape_tf_cur_Reshape[frame_idx],
                                                                            0, type_index_cur_type)
                SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                    SYM_COORD_DX_Reshape_tf_cur_Reshape[frame_idx], 0, type_index_cur_type)
                SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                    SYM_COORD_DY_Reshape_tf_cur_Reshape[frame_idx], 0, type_index_cur_type)
                SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                    SYM_COORD_DZ_Reshape_tf_cur_Reshape[frame_idx], 0, type_index_cur_type)
                # batch - norm
                #if (True):
                with tf.no_grad():

                    sji = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.narrow(2,0,1)
                    xyz_hat = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.narrow(2,1,3)
                    sji_avg = tf.mean(sji)
                    xyz_hat_avg = tf.mean(xyz_hat)
                    avg_unit = tf.cat((sji_avg.reshape(1, 1), xyz_hat_avg.expand(1,3)), dim = 1)
                    sji_avg2 = tf.mean(sji ** 2)
                    xyz_hat_avg2 = tf.mean(xyz_hat ** 2)
                    avg_unit2 = tf.cat((sji_avg2.reshape(1,1), xyz_hat_avg2.expand(1,3)), dim = 1)
                    std_unit = tf.sqrt(avg_unit2 - avg_unit ** 2)
                    avg_cur_type = tf.cat((avg_unit[0][0].reshape(1, 1), tf.zeros((1,3),device = device)), dim = 1)
                    std_cur_type = std_unit + 1E-8

                SYM_COORD_Reshape_tf_cur_Reshape_cur_type = (SYM_COORD_Reshape_tf_cur_Reshape_cur_type - avg_cur_type) / std_cur_type
                SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type / std_cur_type
                SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type / std_cur_type
                SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type / std_cur_type

                SYM_COORD_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.requires_grad_()

                SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.narrow(2, 0, 1)
                G_cur_type  = tf.tanh(self.filter_input[type_idx](SYM_COORD_Reshape_tf_cur_Reshape_cur_type_slice))
                for filter_hidden_idx, filter_hidden_layer in enumerate(self.filter_hidden[type_idx]):
                    G_cur_type = tf.tanh(filter_hidden_layer(G_cur_type))
                RG_cur_type = tf.bmm((SYM_COORD_Reshape_tf_cur_Reshape_cur_type).transpose(1, 2), G_cur_type)
                RG_cur_type = RG_cur_type / ( parameters.SEL_A_max)
                GRRG_cur_type = tf.bmm( RG_cur_type.transpose(1,2), RG_cur_type.narrow(2, 0, parameters.axis_neuron) )
                GRRG_cur_type = tf.reshape(GRRG_cur_type, ((GRRG_cur_type.shape)[0], parameters.filter_neuron[len(parameters.filter_neuron) - 1] * parameters.axis_neuron,))
                E_cur_type = tf.tanh(self.fitting_input[type_idx](GRRG_cur_type))
                for fitting_hidden_idx, fitting_hidden_layer in enumerate(self.fitting_hidden[type_idx]):
                    E_cur_type = tf.tanh(fitting_hidden_layer(E_cur_type))
                E_cur_type = (self.fitting_out[type_idx](E_cur_type))  # Final layer do not use activation function
                E_cur_frame_atom_wise.scatter_(0, type_index_cur_type, tf.reshape(E_cur_type, (-1, )))
                E_tot_cur_type = tf.sum(E_cur_type)
                D_E_D_SYM_cur_type = tf.autograd.grad(E_tot_cur_type, SYM_COORD_Reshape_tf_cur_Reshape_cur_type, create_graph = True)[0] #(N_Atoms_cur_type, SEL_A_max, 4)
                #Start to calculate force of this type. DO NOT forget to multiply -1 !!
                ##as center atom
                F_as_center_atom_curtype_x = tf.sum(
                    tf.sum(D_E_D_SYM_cur_type * SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type, dim=1), dim=1)
                F_as_center_atom_curtype_y = tf.sum(
                    tf.sum(D_E_D_SYM_cur_type * SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type, dim=1), dim=1)
                F_as_center_atom_curtype_z = tf.sum(
                    tf.sum(D_E_D_SYM_cur_type * SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type, dim=1), dim=1)

                F_as_center_xyz_atom = tf.reshape(tf.cat( (F_as_center_atom_curtype_x, F_as_center_atom_curtype_y, F_as_center_atom_curtype_z) ), (3, -1)).transpose(0,1)
                F_cur_frame_dummy.scatter_(0, type_index_cur_type.expand(3, len(type_index_cur_type)).transpose(0, 1), F_as_center_xyz_atom)


                ##as nei atom

                ##For example, atom j is atom i's kth neighbour. Then:
                ##F_X_Atom_j += (-1) * tf.sum(D_E_D_SYM_cur_type[i][j] * SYM_COORD_DX_Reshape_tf_cur_Teshape_cur_type[i][k])
                ##F_Y_Atom_j += ......
                ##......
                ##Use index_select to select i and j; Use reshape and add offset to simulate looping over i

                offset_array = tf.arange(0, (D_E_D_SYM_cur_type.shape)[0], device=device).reshape(-1, 1).expand(
                    (D_E_D_SYM_cur_type.shape)[0], parameters.SEL_A_max).reshape(-1, ).to(device) * parameters.SEL_A_max
                NEI_IDX_Reshape_tf_cur_cur_type_new = NEI_IDX_Reshape_tf_cur_cur_type.reshape(-1, ) + offset_array
                D_E_D_SYM_cur_type_nei = tf.index_select(
                    D_E_D_SYM_cur_type.reshape((D_E_D_SYM_cur_type.shape)[0] * (D_E_D_SYM_cur_type.shape)[1],
                                               (D_E_D_SYM_cur_type.shape)[2]), 0,
                    NEI_IDX_Reshape_tf_cur_cur_type_new).reshape(D_E_D_SYM_cur_type.shape)
                SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type_nei = SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type.reshape(D_E_D_SYM_cur_type.shape)
                SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type_nei = SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type.reshape(D_E_D_SYM_cur_type.shape)
                SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type_nei = SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type.reshape(D_E_D_SYM_cur_type.shape)
                F_as_nei_atom_curtype_x = (-1.0) * tf.sum(
                    D_E_D_SYM_cur_type_nei * SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type_nei, dim=2)
                F_as_nei_atom_curtype_y = (-1.0) * tf.sum(
                    D_E_D_SYM_cur_type_nei * SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type_nei, dim=2)
                F_as_nei_atom_curtype_z = (-1.0) * tf.sum(
                    D_E_D_SYM_cur_type_nei * SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type_nei, dim=2)
                F_as_nei_xyz_atom_cur_type = tf.cat(
                    (F_as_nei_atom_curtype_x, F_as_nei_atom_curtype_y, F_as_nei_atom_curtype_z), dim=1).reshape(
                    len(F_as_nei_atom_curtype_x), 3, parameters.SEL_A_max).transpose(1,
                                                                                     2)  # shape=(N_Atoms_cur_type, SEL_A_max, 3)
                F_cur_frame_dummy_nei.scatter_add_(0, NEI_IDX_Reshape_tf_cur_cur_type.reshape(-1, 1).expand(
                    len(F_as_nei_atom_curtype_x) * parameters.SEL_A_max, 3), F_as_nei_xyz_atom_cur_type.reshape(
                    len(F_as_nei_atom_curtype_x) * parameters.SEL_A_max, 3))

                F_cur_frame_dummy_nei = F_cur_frame_dummy_nei.narrow(0, 0, data_cur[4][0])#data_cur[12][frame_idx])

                ##add to F_cur_batch[frame_idx]
                F_cur_batch[frame_idx] += F_cur_frame_dummy
                F_cur_batch[frame_idx] += F_cur_frame_dummy_nei

            E_cur_frame = tf.sum(E_cur_frame_atom_wise)
            #print("EATOM",E_cur_frame_atom_wise)
            #gg = tf.autograd.grad(E_cur_frame, SYM_COORD_Reshape_tf_cur_Reshape[frame_idx])
            E_cur_batch[frame_idx] = E_cur_frame
            F_cur_batch *= -1.0
            """
        F_cur_batch *= -1.0
        E_cur_batch = tf.sum(E_cur_batch_atom_wise.reshape(len(data_cur[1]), data_cur[4][0]), dim=1)
        #use_std_avg = True
        return E_cur_batch, F_cur_batch, std_, avg_

    def forward_fitting_only(self, data_cur, parameters, std, avg, use_std_avg, device):
        SYM_COORD_Reshape_tf_cur = data_cur[1]
        N_ATOMS_tf_cur = data_cur[4]
        SYM_COORD_Reshape_tf_cur_Reshape = tf.reshape(data_cur[1], \
                                                      (len(SYM_COORD_Reshape_tf_cur), N_ATOMS_tf_cur[0], \
                                                       parameters.N_sym_coord))

        SYM_COORD_DX_Reshape_tf_cur_Reshape = tf.reshape(data_cur[9], (len(data_cur[1]), N_ATOMS_tf_cur[0],
                                                                           parameters.N_sym_coord, N_ATOMS_tf_cur[0]))
        SYM_COORD_DY_Reshape_tf_cur_Reshape = tf.reshape(data_cur[10], (len(data_cur[1]), N_ATOMS_tf_cur[0],
                                                                            parameters.N_sym_coord, N_ATOMS_tf_cur[0]))
        SYM_COORD_DZ_Reshape_tf_cur_Reshape = tf.reshape(data_cur[11], (len(data_cur[1]), N_ATOMS_tf_cur[0],
                                                                            parameters.N_sym_coord, N_ATOMS_tf_cur[0]))

        NEI_IDX_Reshape_tf_cur = tf.reshape(data_cur[6], (len(data_cur[6]), data_cur[4][0], parameters.SEL_A_max))
        NEI_TYPE_Reshape_tf_cur = tf.reshape(data_cur[13], (len(data_cur[6]), data_cur[4][0], parameters.SEL_A_max))
        E_cur_batch = tf.zeros(len(SYM_COORD_Reshape_tf_cur), device=device)
        E_cur_batch_atom_wise = tf.zeros((len(data_cur[1]), data_cur[4][0]), device=device).reshape(-1, )
        F_cur_batch = tf.zeros((len(SYM_COORD_Reshape_tf_cur), data_cur[4][0], 3), device=device)
        std_ = tf.zeros((1, parameters.N_types_all_frame, 4), device=device)
        avg_ = tf.zeros((1, parameters.N_types_all_frame, 4), device=device)

        #SYM_COORD_Reshape_tf_cur_Reshape = self.batch_norm(SYM_COORD_Reshape_tf_cur_Reshape)
        for type_idx in range(parameters.N_types_all_frame):

            """Now we load the pre-calculated derivative, N_frames * N_atoms_max * N_PTSDs * N_atoms_max * 3 * 8 / 1024 / 1024 Mb"""

            type_idx_cur_type = (data_cur[5] == parameters.type_index_all_frame[type_idx]).nonzero()
            frame_idx_cur_type = type_idx_cur_type.narrow(1, 0, 1)
            type_idx_cur_type = (
                        frame_idx_cur_type * data_cur[4][0].long() + type_idx_cur_type.narrow(1, 1, 1)).reshape(
                -1, )
            SYM_COORD_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_Reshape_tf_cur_Reshape.reshape(len(data_cur[1]) * data_cur[4][0], parameters.N_sym_coord), 0,
                type_idx_cur_type)
            shape_d_tmp = SYM_COORD_DX_Reshape_tf_cur_Reshape.shape
            SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_DX_Reshape_tf_cur_Reshape.reshape(shape_d_tmp[0] * shape_d_tmp[1], shape_d_tmp[2],
                                                            shape_d_tmp[3]), 0, type_idx_cur_type)
            SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_DY_Reshape_tf_cur_Reshape.reshape(shape_d_tmp[0] * shape_d_tmp[1], shape_d_tmp[2],
                                                            shape_d_tmp[3]), 0, type_idx_cur_type)
            SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = tf.index_select(
                SYM_COORD_DZ_Reshape_tf_cur_Reshape.reshape(shape_d_tmp[0] * shape_d_tmp[1], shape_d_tmp[2],
                                                            shape_d_tmp[3]), 0, type_idx_cur_type)

            ###DeePMD-type batch-norm
            if (use_std_avg == True):
                """SYM_COORD_Reshape_tf_cur_Reshape_cur_type = (SYM_COORD_Reshape_tf_cur_Reshape_cur_type - avg[
                    type_idx]) / std[type_idx]
                SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type / std[
                    type_idx]
                SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type / std[
                    type_idx]
                SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type / std[
                    type_idx]"""
                std_ = std
                avg_ = avg
                std_cur_type = std_[0][type_idx]
                avg_cur_type = avg_[0][type_idx]
            else:
                with tf.no_grad():
                    sji = SYM_COORD_Reshape_tf_cur_Reshape_cur_type
                    xyz_hat = SYM_COORD_Reshape_tf_cur_Reshape_cur_type
                    sji_avg = tf.mean(sji)
                    xyz_hat_avg = tf.mean(xyz_hat)
                    avg_unit = tf.cat((sji_avg.reshape(1, 1), xyz_hat_avg.expand(1, 3)), dim=1)
                    sji_avg2 = tf.mean(sji ** 2)
                    xyz_hat_avg2 = tf.mean(xyz_hat ** 2)
                    avg_unit2 = tf.cat((sji_avg2.reshape(1, 1), xyz_hat_avg2.expand(1, 3)), dim=1)
                    std_unit = tf.sqrt(avg_unit2 - avg_unit ** 2)
                    avg_cur_type = tf.cat((avg_unit[0][0].reshape(1, 1), tf.zeros((1, 3), device=device)), dim=1)
                    #avg_cur_type = avg_unit
                    std_cur_type = std_unit + 1E-16
                    std_[0][type_idx] = std_cur_type
                    avg_[0][type_idx] = avg_cur_type

            SYM_COORD_Reshape_tf_cur_Reshape_cur_type = (SYM_COORD_Reshape_tf_cur_Reshape_cur_type - avg_cur_type.reshape(-1, )[0]) / std_cur_type.reshape(-1, )[0]
            SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type / \
                                                           std_cur_type.reshape(-1, )[0]
            SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type / \
                                                           std_cur_type.reshape(-1, )[0]
            SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type / \
                                                           std_cur_type.reshape(-1, )[0]
            SYM_COORD_Reshape_tf_cur_Reshape_cur_type = SYM_COORD_Reshape_tf_cur_Reshape_cur_type.requires_grad_()

            E_cur_type = tf.tanh(self.fitting_input[type_idx](SYM_COORD_Reshape_tf_cur_Reshape_cur_type))
            for fitting_hidden_idx, fitting_hidden_layer in enumerate(self.fitting_hidden[type_idx]):
                E_cur_type = tf.tanh(fitting_hidden_layer(E_cur_type))
            E_cur_type = (self.fitting_out[type_idx](E_cur_type))
            E_cur_batch_atom_wise.scatter_(0, type_idx_cur_type, tf.reshape(E_cur_type, (-1,)))

            E_tot_cur_type = tf.sum(E_cur_type)
            D_E_D_SYM_cur_type = \
                tf.autograd.grad(E_tot_cur_type, SYM_COORD_Reshape_tf_cur_Reshape_cur_type, create_graph=True)[0]
            shape_d_e_tmp = D_E_D_SYM_cur_type.shape
            D_E_D_SYM_cur_type = tf.reshape(D_E_D_SYM_cur_type, (shape_d_e_tmp[0], shape_d_e_tmp[1], 1))
            SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type = tf.reshape(SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type * \
                                                           D_E_D_SYM_cur_type, (-1, ))
            SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type = tf.reshape(SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type * \
                                                           D_E_D_SYM_cur_type, (-1, ))
            SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type = tf.reshape(SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type * \
                                                           D_E_D_SYM_cur_type, (-1, ))
            F_x_cur_type = tf.sum(
                SYM_COORD_DX_Reshape_tf_cur_Reshape_cur_type.reshape(-1, parameters.N_sym_coord, N_ATOMS_tf_cur[0]),
                dim=1)
            F_y_cur_type = tf.sum(
                SYM_COORD_DY_Reshape_tf_cur_Reshape_cur_type.reshape(-1, parameters.N_sym_coord, N_ATOMS_tf_cur[0]),
                dim=1)
            F_z_cur_type = tf.sum(
                SYM_COORD_DZ_Reshape_tf_cur_Reshape_cur_type.reshape(-1, parameters.N_sym_coord, N_ATOMS_tf_cur[0]),
                dim=1)

            F_xyz_cur_type = tf.cat((F_x_cur_type, F_y_cur_type, F_z_cur_type)).reshape(3, -1).transpose(0,1)
            len_frame_idx = len(frame_idx_cur_type)
            F_idx_xyz_cur_type_ = tf.arange(N_ATOMS_tf_cur[0], device=device).expand(len_frame_idx, N_ATOMS_tf_cur[0])
            F_idx_xyz_cur_type_offset = frame_idx_cur_type.expand(len_frame_idx, N_ATOMS_tf_cur[0]) * N_ATOMS_tf_cur[0]
            F_idx_xyz_cur_type = F_idx_xyz_cur_type_ + F_idx_xyz_cur_type_offset
            F_idx_xyz_cur_type = F_idx_xyz_cur_type.reshape(-1, )
            F_idx_xyz_cur_type_all = tf.cat((F_idx_xyz_cur_type * 3, F_idx_xyz_cur_type * 3 + 1, F_idx_xyz_cur_type * 3 + 2)).reshape(3, -1).transpose(0, 1).reshape(-1, )

            shape_f_tmp = F_cur_batch.shape
            F_cur_batch = F_cur_batch.reshape(-1, ).scatter_add_(0, F_idx_xyz_cur_type_all.long(), F_xyz_cur_type.reshape(-1, )).reshape(shape_f_tmp)

            1




        F_cur_batch *= -1.
        E_cur_batch = tf.sum(E_cur_batch_atom_wise.reshape(len(data_cur[1]), data_cur[4][0]), dim=1)
        return E_cur_batch, F_cur_batch, std_, avg_



def init_weights(m):
    '''Takes in a module and initializes all linear layers with weight
               values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if isinstance(m, nn.Linear):
        #print("m.bias:", m.bias.data)
        tf.nn.init.xavier_normal_(m.weight, gain = 0.707106781186547524400844362104849039284835937688)
        #tf.nn.init.constant_(m.weight,0.02)
        #m.bias.data.normal_(mean = 0, std = 1.0)
        #tf.nn.init.constant_(m.bias, -0.01)


def make_dot(var, params):
    #To use this method, you need to uncomment the import torchvision & graphviz lines
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if tf.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(var.grad_fn)
    return dot





