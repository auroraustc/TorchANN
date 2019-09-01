#!/usr/bin/env python3

import os
import json
import numpy as np
import random
import math

DATA_PATH = "./data"
TORCHANN_CPP_EXE = " GENERATE_P "
TORCHANN_TRAIN = " train_noclassify_nompi.py "
TORCHANN_PREDICT = " predict_noclassify_nompi.py "
TORCHANN_PARAMS = " PARAMS.json "
LAMMPS_EXE = " lmp_intel_cpu_intelmpi "
LAMMPS_COMMAND = " -v mode file "
LAMMPS_INPUT = " in.clint.W "
LAMMPS_DATA = " data.W "
LASP_EXE = " lasp "
VASP_EXE = " vasp_gpu_544 "
BACKGROUND_SYMBOL = " & "
LEFT_ARROW = " < "
PYTHON2 = " python2 "
PYTHON3 = " python3 "
TORCHAN_LAMMPS_WRAPER = " TorchANN_wrap.py file "
WAIT = " wait "
DATASET_DIR_PREFIX = "DATASET_"
EXPLORE_DIR_PREFIX = "EXPLORE_"
DFT_DIR_PREFIX = "DFT_"
SCRIPTS_PATH = "../"



class auto_train_parameters():
    def __init__(self):
        total_loop = 1
        explore_method = 1 # 1 for lammps MD, 2 for lasp SSW
        dft_method = 1 # 1 for VASP
        explore_ratio = 0.2 # how many structures will be used to perform explore step
        dft_ratio = 0.2 # How many explored structures will be chosen to perform DFT and add to training set
        t_generate = "GENERATE_P"
        t_train = "train_noclassify_nompi.py"
        t_predict = "predict_noclassify_nompi.py"
        t_input = "PARAMS.json"
        data_path = "./"
        lammps_exe = "lmp_intel_cpu_intelmpi"
        lammps_input = "in.clint.W"
        lammps_data_name = "data.W"
        lasp_exe = "lasp"
        vasp_exe = "vasp_std"

    def __str__(self):
        str_ = []
        str_ += (">>> total_loop: %5d\n" % self.total_loop)
        str_ += (">>> explore_method: %5d\n" % self.explore_method)
        str_ += (">>> dft_method: %5d\n" % self.dft_method)
        str_ += (">>> explore_ratio: %5.2f%%\n" % (self.explore_ratio * 100.0))
        str_ += (">>> dft_ratio: %5.2f%%\n" % (self.dft_ratio * 100.0))
        str_ += (">>> t_generate: %s\n" % self.t_generate)
        str_ += (">>> t_train: %s\n" % self.t_train)
        str_ += (">>> t_predict: %s\n" % self.t_predict)
        str_ += (">>> t_input: %s\n" % self.t_input)
        str_ += (">>> t_data_pat: %s\n" % self.data_path)
        if (self.explore_method == 1):
            str_ += (">>> lammps_exe: %s\n" % self.lammps_exe)
            str_ += (">>> lammps_input: %s\n" % self.lammps_input)
            str_ += (">>> lammps_data_name: %s\n" % self.lammps_data_name)
        elif (self.explore_method == 2):
            str_ += (">>> lasp_exe: %s\n" % self.lasp_exe)
        if (self.dft_method == 1):
            str_ += (">>> vasp_exe: %s\n" % self.vasp_exe)

        str_ = ''.join(str(element) for element in str_)
        return str_

    def read_parameters(self, filename):
        INPUT_FILE = open(filename, "r")
        INPUT_DATA = json.load(INPUT_FILE)
        self.total_loop = INPUT_DATA['total_loop']
        self.explore_method = INPUT_DATA['explore_method']
        self.dft_method = INPUT_DATA['dft_method']
        self.explore_ratio = INPUT_DATA['explore_ratio']
        self.dft_ratio = INPUT_DATA['dft_ratio']
        self.t_generate = INPUT_DATA['t_generate']
        self.t_train = INPUT_DATA['t_train']
        self.t_predict = INPUT_DATA['t_predict']
        self.t_input = INPUT_DATA['t_input']
        self.data_path = INPUT_DATA['data_path']
        if (self.explore_method == 1):
            self.lammps_exe = INPUT_DATA['lammps_exe']
            self.lammps_input = INPUT_DATA['lammps_input']
            self.lammps_data_name = INPUT_DATA['lammps_data_name']
        elif (self.explore_method == 2):
            self.lasp_exe = INPUT_DATA['lasp_exe']
        else:
            print("explore_method not supported!")
            exit()
        if (self.dft_method == 1):
            self.vasp_exe = INPUT_DATA['vasp_exe']
        else:
            print("dft_method not supported!")
            exit()
        INPUT_FILE.close()

"""Command to run lammps"""
class run_lammps():
    def __init__(self):
        CMD1 = LAMMPS_EXE + LAMMPS_COMMAND + LEFT_ARROW + LAMMPS_INPUT + BACKGROUND_SYMBOL
        CMD2 = PYTHON2 + TORCHAN_LAMMPS_WRAPER
        CMD3 = WAIT
        CMD = CMD1 + ";" + CMD2 + ";" + CMD3

    def execuate(self):
        os.system(self.CMD)

    def printcmd(self):
        print(self.CMD)

class run_lasp():
    def __init__(self):
        CMD1 = " lasp "
        CMD = CMD1

    def execuate(self):
        os.system(self.CMD)

    def printcmd(self):
        print(self.CMD)

"""Copy file to dest. file and dest are strings"""
class cp_file():
    def __init__(self, file, dest):
        CMD1 = "cp -r "
        CMD2 = file + " "
        CMD3 = dest + " "
        CMD = CMD1 + CMD2 + CMD3

    def execuate(self):
        os.system(self.CMD)

    def printcmd(self):
        print(self.CMD)

def int_to_str(INT):
    STR_TMP = []
    STR_TMP += ("%d" % INT)
    STR = ''.join(str(element) for element in STR_TMP)
    return STR

def raw_to_lammps():
    box = np.loadtxt("box.raw", dtype=np.float)
    coord = np.loadtxt("coord.raw", dtype=np.float)
    type = np.loadtxt("type.raw", dtype=np.int)
    type_unique = np.array(list(set(type)))
    type_unique.sort()
    natoms = len(type)
    ntypes = len(type_unique)
    box = box.reshape(3,3)
    coord = coord.reshape(-1, 3)
    p_a = np.sqrt(np.sum(np.square(box[0])))
    p_b = np.sqrt(np.sum(np.square(box[1])))
    p_c = np.sqrt(np.sum(np.square(box[2])))
    lx = p_a
    p_xy = p_b * np.dot(box[0], box[1]) / p_a / p_b
    p_xz = p_c * np.dot(box[0], box[2]) / p_a / p_c
    ly = np.sqrt(p_b**2 - p_xy**2)
    p_yz = (p_b * p_c * np.dot(box[1], box[2]) / p_b / p_c - p_xy * p_xz) / ly
    lz = np.sqrt(p_c**2 - p_xz**2 - p_yz**2)
    lammps_data_f = open(LAMMPS_DATA, "wt")

    lammps_data_f.write("# Converted from POSCAR to lammps format\n")
    lammps_data_f.write("\n")
    lammps_data_f.write("%d atoms\n" % natoms)
    lammps_data_f.write("%d atom types\n" % ntypes)
    lammps_data_f.write("\n")
    lammps_data_f.write("0.000000  %10.6f   xlo xhi\n" % lx)
    lammps_data_f.write("0.000000  %10.6f   ylo yhi\n" % ly)
    lammps_data_f.write("0.000000  %10.6f   zlo zhi\n" % lz)
    lammps_data_f.write("\n")
    lammps_data_f.write("%10.6f  %10.6f  %10.6f   xy xz yz\n" % (p_xy, p_xz, p_yz))
    lammps_data_f.write("\n")
    lammps_data_f.write("Atoms\n")
    lammps_data_f.write("\n")

    for i in range(len(type)):
        lammps_data_f.write("%4i  %-4d   %7f  %7f  %7f\n" % (i + 1, 1 + (type[i] == type_unique).nonzero()[0], \
                                                        coord[i][0], coord[i][1], coord[i][2]))

    lammps_data_f.close()


"""Convert raw to POSCAR. The elements will arange in ascending order, so make sure the element sequence in POTCAR
   is also aranged in ascending order according to their element numbers."""
def raw_to_poscar():

    POSCAR_F = open("POSCAR", "wt")
    box = np.loadtxt("box.raw")
    POSCAR_HEADER = "Converted from raw\n"
    POSCAR_F.write(POSCAR_HEADER)
    POSCAR_F.write(" %f\n" % 1.0)
    POSCAR_F.write(" %10.6f %10.6f %10.6f\n %10.6f %10.6f %10.6f\n %10.6f %10.6f %10.6f\n" % (box[0], box[1], box[2], box[3], box[4], box[5], \
                                                                   box[6], box[7], box[8]))
    type = np.loadtxt("type.raw", dtype=np.int)
    type_unique = np.array(list(set(type)))
    type_unique.sort()

    for i in range(len(type_unique)):
        POSCAR_F.write(" %d " % len((type == type_unique[i]).nonzero()[0]))
    POSCAR_F.write("\n")
    POSCAR_F.write("Cart\n")

    coord = np.loadtxt("coord.raw", dtype=np.float)
    coord = coord.reshape((-1, 3))
    for i in range(len(type_unique)):
        coord_tmp = coord[(type == type_unique[i]).nonzero()[0]]
        for j in range(len(coord_tmp)):
            POSCAR_F.write(" %10.6f %10.6f %10.6f \n" % (coord_tmp[j][0], coord_tmp[j][1], coord_tmp[j][2]))





    POSCAR_F.close()


"""One loop: train->explore->dft"""
def one_loop(loop_idx, auto_train_parameters):

    LOG_f = open("auto_train.log", "at")
    LOG_f.write("Auto training loop %4d\n" % loop_idx)
    #LOG_f.close()

    DIR_SUFFIX = int_to_str(loop_idx)
    DATASET_DIR = DATASET_DIR_PREFIX + DIR_SUFFIX
    EXPLORE_DIR = EXPLORE_DIR_PREFIX + DIR_SUFFIX
    DFT_DIR = DFT_DIR_PREFIX + DIR_SUFFIX
    os.system("mkdir " + DATASET_DIR)
    os.system("mkdir " + EXPLORE_DIR)
    os.system("mkdir " + DFT_DIR)

    """Move necessary files into DATASET_DIR"""
    if (loop_idx == 0):
        CMD = "cp " + auto_train_parameters.data_path + "/*.raw " + DATASET_DIR
        os.system(CMD)
    else:
        for i in ["coord.raw", "type.raw", "energy.raw", "box.raw", "force.raw"]:
            CMD = "cp " + DATASET_DIR_PREFIX + int_to_str(loop_idx - 1) + "/" + i + " " + DATASET_DIR
            os.system(CMD)
            CMD = "cat " + DFT_DIR_PREFIX + int_to_str(loop_idx - 1) + "/" + i + " >> " + DATASET_DIR + "/" + i
            os.system(CMD)
        CMD = "cp " + DATASET_DIR_PREFIX + int_to_str(loop_idx - 1) + "/" + "freeze_model.pytorch" + " " + DATASET_DIR +\
            "/freeze_model.pytorch.ckpt.cont"
        os.system(CMD)
    tmp_frame = np.loadtxt(DATASET_DIR + "/energy.raw")
    tot_frame = len(tmp_frame)
    sel_frame = math.ceil(tot_frame * auto_train_parameters.explore_ratio)
    explore_systems_idx = random.sample(range(tot_frame), sel_frame)
    CMD = "cp " + auto_train_parameters.data_path + "/" + auto_train_parameters.t_input + " " + DATASET_DIR
    os.system(CMD)
    """Start train"""

    #LOG_f = open("auto_train.log", "at")
    LOG_f.write("1/3 of %4d: Training...\n" % loop_idx)
    #LOG_f.close()

    os.chdir(DATASET_DIR)
    CMD1 = TORCHANN_CPP_EXE + " 2>&1 > runlog_g "
    CMD2 = TORCHANN_TRAIN + " 2>&1 > runlog_t "
    CMD = CMD1 + ";" + CMD2
    os.system(CMD)
    os.chdir("../")#DATASET_DIR
    os.system("cp " + DATASET_DIR + "/freeze_model.pytorch" + "  " + EXPLORE_DIR)
    os.system("cp " + DATASET_DIR + "/PARAMS.json" + "  " + EXPLORE_DIR)

    """Select data"""
    for i in ["coord.raw", "type.raw", "energy.raw", "box.raw", "force.raw"]:
        CMD = "cp " + DATASET_DIR + "/" + i + " " + EXPLORE_DIR
        os.system(CMD)

    box_raw_from_data_f = open(DATASET_DIR + "/box.raw", "rt")
    coord_raw_from_data_f = open(DATASET_DIR + "/coord.raw", "rt")
    force_raw_from_data_f = open(DATASET_DIR + "/force.raw", "rt")
    energy_raw_from_data_f = open(DATASET_DIR + "/energy.raw", "rt")
    type_raw_from_data_f = open(DATASET_DIR + "/type.raw", "rt")

    box_raw_from_data = box_raw_from_data_f.readlines()
    coord_raw_from_data = coord_raw_from_data_f.readlines()
    force_raw_from_data = force_raw_from_data_f.readlines()
    energy_raw_from_data = energy_raw_from_data_f.readlines()
    type_raw_from_data = type_raw_from_data_f.readlines()

    box_raw_from_data_f.close()
    coord_raw_from_data_f.close()
    force_raw_from_data_f.close()
    energy_raw_from_data_f.close()
    type_raw_from_data_f.close()

    os.chdir(EXPLORE_DIR)
    for i in range(sel_frame):
        CMD = "mkdir " + int_to_str(i)
        os.system(CMD)
        os.system("cp freeze_model.pytorch" + " " + int_to_str(i))
        os.system("cp PARAMS.json" + " " + int_to_str(i))
        os.chdir(int_to_str(i))

        box = open("box.raw", "wt")
        box.write(box_raw_from_data[explore_systems_idx[i]])
        box.close()
        coord = open("coord.raw", "wt")
        coord.write(coord_raw_from_data[explore_systems_idx[i]])
        coord.close()
        force = open("force.raw", "wt")
        force.write(force_raw_from_data[explore_systems_idx[i]])
        force.close()
        energy = open("energy.raw", "wt")
        energy.write(energy_raw_from_data[explore_systems_idx[i]])
        energy.close()
        type = open("type.raw", "wt")
        type.write(type_raw_from_data[explore_systems_idx[i]])
        type.close()

        os.chdir("../")

    """Run explore in dir 0..sel_frame - 1"""

    #LOG_f = open("auto_train.log", "at")
    LOG_f.write("2/3 of %4d: Exploring...\n" % loop_idx)
    #LOG_f.close()

    for i in range(sel_frame):
        os.chdir(int_to_str(i))

        if (auto_train_parameters.explore_method == 1): #LAMMPS
            CMD = "cp ../../data/" + auto_train_parameters.lammps_input + "  ./"
            os.system(CMD)
            """Convert data to lammps type"""
            raw_to_lammps()
            CMD = "cp " + SCRIPTS_PATH + "/TorchANN_wrap_python2.py " + " ./"
            os.system(CMD)
            CMD1 = TORCHANN_CPP_EXE
            CMD2 = TORCHANN_PREDICT
            CMD3 = CMD1 + ";" + CMD2
            with open("torchanncmd.txt","wt") as f:
                f.write(CMD3)
                f.close()

            """Run lammps"""
            CMD1 = LAMMPS_EXE + " -v mode file < " + auto_train_parameters.lammps_input + " " + BACKGROUND_SYMBOL
            CMD2 = PYTHON2 + " TorchANN_wrap_python2.py file"
            CMD3 = WAIT
            CMD = CMD1 + " \n " + CMD2 + " \n " + CMD3
            with open("run.sh", "wt") as f:
                f.write(CMD)
            os.system("bash run.sh")

        os.chdir("../")

    """cat all frames together"""
    for i in range(sel_frame):
        os.chdir(int_to_str(i))

        os.system("cat ./coord_all.raw >> ../coord_all_exp.raw")
        os.system("cat ./force_all.raw >> ../force_all_exp.raw")
        os.system("cat ./box_all.raw >> ../box_all_exp.raw")
        os.system("cat ./type_all.raw >> ../type_all_exp.raw")
        os.system("cat ./energy_all.raw >> ../energy_all_exp.raw")

        os.chdir("../")

    os.chdir("../") #EXPLORE_DIR

    os.chdir(DFT_DIR)

    """Copy frames to DFT_dir"""
    for i in ["coord", "box", "type", "energy"]:
        CMD1 = "cp " + "../" + EXPLORE_DIR + "/" + i + "_all_exp.raw" + " ./"
        CMD = CMD1
        os.system(CMD)



    """Select data for DFT"""
    tmp_frame_DFT = np.loadtxt("energy_all_exp.raw")
    tot_frame_DFT = len(tmp_frame_DFT)
    sel_frame_DFT = math.ceil(tot_frame_DFT * auto_train_parameters.dft_ratio)
    dft_systems_idx = random.sample(range(tot_frame_DFT), sel_frame_DFT)

    box_raw_from_data_f = open("box_all_exp.raw")
    coord_raw_from_data_f = open("coord_all_exp.raw")
    type_raw_from_data_f = open("type_all_exp.raw")

    box_raw_from_data = box_raw_from_data_f.readlines()
    coord_raw_from_data = coord_raw_from_data_f.readlines()
    type_raw_from_data = type_raw_from_data_f.readlines()

    box_raw_from_data_f.close()
    coord_raw_from_data_f.close()
    type_raw_from_data_f.close()

    box = open("box.raw", "wt")
    coord = open("coord.raw", "wt")
    type = open("type.raw", "wt")
    for i in range(sel_frame_DFT):
        box.write(box_raw_from_data[dft_systems_idx[i]])
        coord.write(coord_raw_from_data[dft_systems_idx[i]])
        type.write(type_raw_from_data[dft_systems_idx[i]])
    box.close()
    coord.close()
    type.close()

    os.chdir("../")
    for i in range(sel_frame_DFT):
        CMD = "mkdir " + DFT_DIR + "/" + int_to_str(i)
        os.system(CMD)
        CMD = "cp " + auto_train_parameters.data_path + "/INCAR " + " " + DFT_DIR + "/" + int_to_str(i)
        os.system(CMD)
        CMD = ("cp " + auto_train_parameters.data_path + "/KPOINTS " + " " + DFT_DIR + "/" +int_to_str(i))
        os.system(CMD)
        CMD = ("cp " + auto_train_parameters.data_path + "/POTCAR " + " " + DFT_DIR + "/" +int_to_str(i))
        os.system(CMD)
    os.chdir(DFT_DIR)

    for i in range(sel_frame_DFT):
        os.chdir(int_to_str(i))
        box = open("box.raw", "wt")
        box.write(box_raw_from_data[dft_systems_idx[i]])
        box.close()
        coord = open("coord.raw", "wt")
        coord.write(coord_raw_from_data[dft_systems_idx[i]])
        coord.close()
        type = open("type.raw", "wt")
        type.write(type_raw_from_data[dft_systems_idx[i]])
        type.close()
        os.chdir("../")





    """Perform DFT calculation"""

    #LOG_f = open("auto_train.log", "at")
    LOG_f.write("3/3 of %4d: DFT...\n" % loop_idx)
    LOG_f.close()

    for i in range(sel_frame_DFT):
        os.chdir(int_to_str(i))
        raw_to_poscar()
        os.system(VASP_EXE)
        """Extract energy and force"""
        os.system("grep free\ \ energy\ \ \ T OUTCAR | awk '{print $5}' > energy.raw")
        type = np.loadtxt("type.raw", dtype=np.int)
        natoms = len(type) + 2
        natoms_str = str(natoms)
        CMD1 = "grep TOTAL-FORCE OUTCAR -A" + natoms_str + " "
        CMD2 = " | sed '/^\ -/d' | sed '/POS/d' | awk '{print $4,$5,$6}' >force.raw "
        CMD = CMD1 + CMD2
        os.system(CMD)
        CMD = "sed -i ':a;N;s/\\n/\ /g;ta' force.raw"
        os.system(CMD)
        os.system("cat energy.raw >> ../energy.raw")
        os.system("cat force.raw >> ../force.raw")
        os.chdir("../")





    os.chdir("../")  # DFT_DIR

    return

def main():

    global TORCHANN_CPP_EXE
    global TORCHANN_TRAIN
    global TORCHANN_PREDICT
    global LAMMPS_EXE
    global LAMMPS_INPUT
    global LAMMPS_DATA
    global LASP_EXE
    global SCRIPTS_PATH
    global VASP_EXE

    SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
    SCRIPTS_PATH += "/scripts"

    auto_train_parameters_in = auto_train_parameters()
    auto_train_parameters_in.read_parameters("PARAMS_AUTO.json")
    print("Check input parameters:\n")
    print(auto_train_parameters_in)
    if (auto_train_parameters_in.explore_method == 1):
        EXP_TYPE = "LAMMPS"
        LAMMPS_EXE = auto_train_parameters_in.lammps_exe
        LAMMPS_INPUT = "" + auto_train_parameters_in.data_path + auto_train_parameters_in.lammps_input + " "
        LAMMPS_DATA = auto_train_parameters_in.lammps_data_name
    elif (auto_train_parameters_in.explore_method == 1):
        EXP_TYPE = "LASP-SSW"
        LASP_EXE = auto_train_parameters_in.lasp_exe
    if (auto_train_parameters_in.dft_method == 1):
        DFT_TYPE = "VASP"
    print("Selected exploration type: %s\n" % EXP_TYPE)
    print("Selected DFT code: %s\n" % DFT_TYPE)

    TORCHANN_CPP_EXE = auto_train_parameters_in.t_generate
    TORCHANN_TRAIN = auto_train_parameters_in.t_train
    TORCHANN_PREDICT = auto_train_parameters_in.t_predict
    VASP_EXE = auto_train_parameters_in.vasp_exe

    for i in range(auto_train_parameters_in.total_loop):
        one_loop(i, auto_train_parameters_in)

    return

if __name__ == '__main__':
    main()






