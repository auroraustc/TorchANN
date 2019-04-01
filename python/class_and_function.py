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
    parameters.Nframes_tot = 23
    parameters.sym_coord_type = 1
    return 0
