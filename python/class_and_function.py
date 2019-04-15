import numpy as np
import torch as tf
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import termios
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models
from graphviz import Digraph
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
        SEL_A_max = 0
        Nframes_tot = 0
        sym_coord_type = 1

        batch_size = 1
        epoch = 1
        filter_neuron = []
        axis_neuron = 1
        fitting_neuron = []
        start_lr = 0.0005
        decay_steps = 100
        decay_epoch = 1
        decay_rate = 0.95
        start_pref_e = 1
        limit_pref_e = 400
        start_pref_f = 1000
        limit_pref_f = 1



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
    parameters.cutoff_3 = 0.0
    parameters.cutoff_max = 8.0
    parameters.N_types_all_frame = 2
    parameters.type_index_all_frame = [0, 1]
    parameters.SEL_A_max = 200
    parameters.Nframes_tot = 2
    parameters.sym_coord_type = 1
###New add parameters
    parameters.batch_size = 16
    parameters.epoch = 1000
    parameters.filter_neuron = [32, 96, 192]
    parameters.axis_neuron = 4
    parameters.fitting_neuron = [1024, 512, 512, 256, 128]
    parameters.start_lr = 0.005
    parameters.decay_steps = 20 #abandoned
    parameters.decay_epoch = 10
    parameters.decay_rate = 0.95
    parameters.start_pref_e = 0.1
    parameters.limit_pref_e = 100.0
    parameters.start_pref_f = 1000.0
    parameters.limit_pref_f = 1.0
    return 0


class one_batch_net(nn.Module):
    def __init__(self, parameters):
        super(one_batch_net, self).__init__()
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
        for type_idx in range(len(parameters.type_index_all_frame)):
            for hidden_idx in range(len(parameters.filter_neuron) - 1):
                self.filter_hidden[type_idx].append(nn.Linear(parameters.filter_neuron[hidden_idx],
                                         parameters.filter_neuron[hidden_idx + 1]))
            for hidden_idx in range(len(parameters.fitting_neuron) - 1):
                self.fitting_hidden[type_idx].append(nn.Linear(parameters.fitting_neuron[hidden_idx],
                                         parameters.fitting_neuron[hidden_idx + 1]))

    def forward(self, data_cur, parameters, device) : #cur mean current batch
        COORD_Reshape_tf_cur = data_cur[0]
        SYM_COORD_Reshape_tf_cur = data_cur[1]
        ENERGY_tf_cur = data_cur[2]
        FORCE_Reshape_tf_cur = data_cur[3]
        N_ATOMS_tf_cur = data_cur[4]
        TYPE_Reshape_tf_cur = data_cur[5]
        SYM_COORD_Reshape_tf_cur_grad = tf.zeros((1, len(data_cur[1]), len(data_cur[1][0])))
        SYM_COORD_Reshape_tf_cur_Reshape = tf.reshape(data_cur[1], \
                                                      (len(SYM_COORD_Reshape_tf_cur), N_ATOMS_tf_cur[0], \
                                                       parameters.SEL_A_max, 4))
        print(SYM_COORD_Reshape_tf_cur.shape)
        SYM_COORD_Reshape_tf_cur_Reshape_slice = SYM_COORD_Reshape_tf_cur_Reshape.narrow(3, 0, 1)
        E_cur_batch = tf.zeros(len(SYM_COORD_Reshape_tf_cur), device = device)
        for frame_idx in range(len(SYM_COORD_Reshape_tf_cur)):
            E_cur_frame = tf.zeros(1, device = device)
            E_cur_frame_atom_wise = tf.zeros(N_ATOMS_tf_cur[0], device = device)
            for atom_idx in range(N_ATOMS_tf_cur[0]):
                type_idx_cur_atom = parameters.type_index_all_frame.index(TYPE_Reshape_tf_cur[frame_idx][atom_idx])
                G_cur_atom = tf.tanh(self.filter_input[type_idx_cur_atom](SYM_COORD_Reshape_tf_cur_Reshape_slice[frame_idx][atom_idx]))
                for filter_hidden_idx, filter_hidden_layer in enumerate(self.filter_hidden[type_idx_cur_atom]):
                    G_cur_atom = tf.tanh(filter_hidden_layer(G_cur_atom))
                RG_cur_atom = tf.mm(SYM_COORD_Reshape_tf_cur_Reshape[frame_idx][atom_idx].transpose(0, 1), G_cur_atom)
                GRRG_cur_atom = tf.mm(RG_cur_atom.transpose(0, 1), RG_cur_atom.narrow(1, 0, parameters.axis_neuron))
                GRRG_cur_atom = tf.reshape(GRRG_cur_atom, (parameters.filter_neuron[len(parameters.filter_neuron) - 1] * parameters.axis_neuron, ))
                E_cur_atom = tf.tanh(self.fitting_input[type_idx_cur_atom](GRRG_cur_atom))
                for fitting_hidden_idx, fitting_hidden_layer in enumerate(self.fitting_hidden[type_idx_cur_atom]):
                    E_cur_atom = tf.tanh(fitting_hidden_layer(E_cur_atom))
                E_cur_atom = (self.fitting_out[type_idx_cur_atom](E_cur_atom))#Final layer do not use activation function
                E_cur_frame_atom_wise[atom_idx] = E_cur_atom
                #SYM_COORD_Reshape_tf_cur_grad[0][frame_idx] = tf.autograd.grad(E_cur_atom, data_cur[1], create_graph = True)[0][frame_idx]
            E_cur_frame = tf.sum(E_cur_frame_atom_wise)
            #gg = tf.autograd.grad(E_cur_frame, SYM_COORD_Reshape_tf_cur_Reshape[frame_idx])
            E_cur_batch[frame_idx] = E_cur_frame
        return E_cur_batch


def make_dot(var, params):
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





