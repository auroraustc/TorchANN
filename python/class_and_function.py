import numpy as np
import torch as tf

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
    parameters.Nframes_tot = 1299
    parameters.sym_coord_type = 1
    return 0

def reshape_to_frame_wise(source, n_atoms, parameters, flag):
    if (type(parameters).__name__ != "Parameters"):
        print("type error in reshape:", type(parameters).__name__," is NOT a correct Parameters class")
        return 1
    if (flag == 1):#force and coord
        interval = 3
    elif (flag == 2):#sym_coord
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
    return source_Reshape

###not complete
def read_reshape(target, dtype, n_atoms, parameters, flag):
    if (type(parameters).__name__ != "Parameters"):
        print("type error in reshape:", type(parameters).__name__," is NOT a correct Parameters class")
        return 1
    fp = open("./SYM_COORD.BIN", "rb")
    start_atom = 0
    end_atom = n_atoms[0] - 1
    target = []
    if (flag == 3):
        interval = parameters.SEL_A_max * 4
    byte_offset = end_atom * np.dtype(dtype).itemsize * interval


    fp.close()
