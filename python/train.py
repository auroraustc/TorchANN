#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import math
import torch as tf
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import horovod.torch as hvd
import os
import gc
import time
from ctypes import *
from class_and_function import *


default_dtype = tf.float64
tf.set_default_dtype(default_dtype)
tf.set_printoptions(precision=10)
device = tf.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = tf.device('cpu')

hvd.init()
if (device != tf.device('cpu')):
    print("cuDNN version: ", tf.backends.cudnn.version())
    # tf.backends.cudnn.enabled = False
    #tf.backends.cudnn.benchmark = True
    tf.cuda.set_device(hvd.local_rank() % tf.cuda.device_count())
if (hvd.rank() == 0):
    f_out = open("./LOSS.OUT", "w")
    f_out.close()

print("hvd.size():", hvd.size())
if (hvd.rank() == 0):
    f_out = open("./LOSS.OUT", "a")
    print("hvd.size():", hvd.size(), file=f_out)
    f_out.close()

#device = torch.device('cpu')


"""Load coordinates, sym_coordinates, energy, force, type, n_atoms and parameters"""
parameters = Parameters()
read_parameters_flag = read_parameters(parameters)
print("All parameters:")
print(parameters)
if (hvd.rank() == 0):
    f_out = open("./LOSS.OUT", "a")
    print("All parameters:", file = f_out)
    print(parameters, file = f_out)
    f_out.close()
if (read_parameters_flag != 0):
    print("Reading parameters error with code %d\n"%read_parameters_flag)
    exit()

COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf, TYPE_Reshape_tf, NEI_IDX_Reshape_tf, \
NEI_COORD_Reshape_tf, FRAME_IDX_tf, SYM_COORD_DX_Reshape_tf, SYM_COORD_DY_Reshape_tf, SYM_COORD_DZ_Reshape_tf, \
N_ATOMS_ORI_tf= read_and_init_bin_file(parameters, default_dtype=default_dtype)

"""Now all the needed information has been stored in the COORD_Reshape, SYM_COORD_Reshape, 
   ENERGY and FORCE_Reshape array."""

print("Data pre-processing complete. Building net work.\n")

mean_init=np.zeros(parameters.N_types_all_frame)
A = tf.zeros(parameters.N_types_all_frame, parameters.Nframes_tot)
for type_idx in range(parameters.N_types_all_frame):
    A[type_idx] = tf.sum(TYPE_Reshape_tf == parameters.type_index_all_frame[type_idx], dim=1)
A = A.transpose(0,1).numpy()
B = ENERGY_tf.numpy()
mean_init = np.linalg.lstsq(A,B,rcond=-1)[0]


ONE_BATCH_NET = one_batch_net(parameters, mean_init)
###init_weights using xavier with gain = sqrt(0.5) is necessary. Now the damn adam works good with this initialization
ONE_BATCH_NET.apply(init_weights)
ONE_BATCH_NET = ONE_BATCH_NET.to(device)
TOTAL_NUM_PARAMS = sum(p.numel() for p in ONE_BATCH_NET.parameters() if p.requires_grad)
"""if (tf.cuda.device_count() > 1):
    ONE_BATCH_NET = nn.DataParallel(ONE_BATCH_NET)"""
print(ONE_BATCH_NET)
print("Number of parameters in the net: %d"%TOTAL_NUM_PARAMS)
if (hvd.rank() == 0):
    f_out = open("./LOSS.OUT", "a")
    print(ONE_BATCH_NET, file=f_out)
    print("Number of parameters in the net: %d" % TOTAL_NUM_PARAMS, file=f_out)
    f_out.close()

DATA_SET = tf.utils.data.TensorDataset(COORD_Reshape_tf, SYM_COORD_Reshape_tf, ENERGY_tf, FORCE_Reshape_tf, N_ATOMS_tf, \
                                       TYPE_Reshape_tf, NEI_IDX_Reshape_tf, NEI_COORD_Reshape_tf, FRAME_IDX_tf, \
                                       SYM_COORD_DX_Reshape_tf, SYM_COORD_DY_Reshape_tf, SYM_COORD_DZ_Reshape_tf, \
                                       N_ATOMS_ORI_tf)#0..12
TRAIN_SAMPLER = tf.utils.data.distributed.DistributedSampler(DATA_SET, num_replicas=hvd.size(), rank=hvd.rank())
TRAIN_LOADER = tf.utils.data.DataLoader(DATA_SET, batch_size = parameters.batch_size, sampler = TRAIN_SAMPLER)
#TRAIN_LOADER = tf.utils.data.DataLoader(DATA_SET, batch_size = parameters.batch_size, shuffle = True)
OPTIMIZER2 = optim.Adam(ONE_BATCH_NET.parameters(), lr = parameters.start_lr)#, amsgrad = True, weight_decay = 1e-3)
OPTIMIZER2 = hvd.DistributedOptimizer(OPTIMIZER2, named_parameters=ONE_BATCH_NET.named_parameters())

"""
###DO NOT use LBFGS. LBFGS is horrible on such kind of optimizations
###Adam works also horribly. So it is better to use Adadelta
###Now adam works OK if applied init_weights
OPTIMIZER = optim.LBFGS(ONE_BATCH_NET.parameters(), lr = parameters.start_lr)
"""


CRITERION = nn.MSELoss(reduction = "mean")
LR_SCHEDULER = tf.optim.lr_scheduler.ExponentialLR(OPTIMIZER2, parameters.decay_rate)
START_TRAIN_TIMER = time.time()
STEP_CUR = 0

hvd.broadcast_parameters(ONE_BATCH_NET.state_dict(), root_rank=0)

print("Start training using device: ", device, ", count: ", tf.cuda.device_count())
if (parameters.decay_rate > 1):
    print("-----------------------------------------------")
    print("|*******************WARNING*******************|")
    print("|YOUR SETTING FOR decay_rate IS LARGER THAN 1!|")
    print("|     HOPE YOU KNOW WHAT YOU ARE DOING!       |")
    print("-----------------------------------------------")
    if (hvd.rank() == 0):
        f_out = open("./LOSS.OUT", "a")
        print("-----------------------------------------------", file=f_out)
        print("|*******************WARNING*******************|", file=f_out)
        print("|YOUR SETTING FOR decay_rate IS LARGER THAN 1!|", file=f_out)
        print("|     HOPE YOU KNOW WHAT YOU ARE DOING!       |", file=f_out)
        print("-----------------------------------------------", file=f_out)
        f_out.close()
#with tf.autograd.profiler.profile(enabled = True, use_cuda=True) as prof:

if (True):
#with tf.autograd.profiler.profile(enabled = True, use_cuda=True) as prof:
    START_BATCH_USER_TIMER = time.time()
    for epoch in range(parameters.epoch):
        TRAIN_SAMPLER.set_epoch(epoch)
        START_EPOCH_TIMER = time.time()
        if (parameters.epoch != 1 ):
            pref_e = (parameters.limit_pref_e - parameters.start_pref_e) * 1.0 / (parameters.epoch - 1.0) * epoch + parameters.start_pref_e
            pref_f = (parameters.limit_pref_f - parameters.start_pref_f) * 1.0 / (parameters.epoch - 1.0) * epoch + parameters.start_pref_f
        else:
            pref_e = parameters.start_pref_e
            pref_f = parameters.start_pref_f
        if ((epoch % parameters.decay_epoch == 0)):  # and (STEP_CUR > 0)):
            LR_SCHEDULER.step()
            print("LR update: lr = %lf" % OPTIMIZER2.param_groups[0].get("lr"))
            if (hvd.rank() == 0):
                f_out = open("./LOSS.OUT", "a")
                print("LR update: lr = %lf" % OPTIMIZER2.param_groups[0].get("lr"), file=f_out)
                f_out.close()

        for batch_idx, data_cur in enumerate(TRAIN_LOADER):
            for i in range(len(data_cur)):
                data_cur[i] = data_cur[i].to(device)
            START_BATCH_TIMER = time.time()

            PROF_FLAG = (STEP_CUR == -1)
            with tf.autograd.profiler.profile(enabled=PROF_FLAG, use_cuda=True) as prof:
            #if (True):
                #data_cur[1].requires_grad = True
                NEI_IDX_Reshape_tf_cur = data_cur[6]
                NEI_IDX_Reshape_tf_cur = tf.reshape(NEI_IDX_Reshape_tf_cur,
                                                    (len(NEI_IDX_Reshape_tf_cur), data_cur[4][0], parameters.SEL_A_max))
                FORCE_Reshape_tf_cur = data_cur[3]
                #FORCE_Reshape_tf_cur_Reshape = tf.reshape(FORCE_Reshape_tf_cur,
                #                                          (len(FORCE_Reshape_tf_cur), data_cur[4][0], 3))
                #FORCE_net_tf_cur = tf.zeros((len(FORCE_Reshape_tf_cur), data_cur[4][0] * 3), device = device)
                NEI_COORD_Reshape_tf_cur = data_cur[7]

                ###Adam
                # correct
                if (data_cur[1].grad):
                    data_cur[1].grad.data.zero_()
                E_cur_batch, F_cur_batch = ONE_BATCH_NET(data_cur, parameters, device)
                # Energy loss part
                loss_E_cur_batch = CRITERION(E_cur_batch, data_cur[2])
                # Force
                F_cur_batch = tf.reshape(F_cur_batch, (len(data_cur[6]), data_cur[4][0] * 3))
                loss_F_cur_batch = tf.zeros(1,device = device)

                if ((STEP_CUR % 1000 < hvd.size()) and hvd.rank() == 0):
                    print("Force check:\n", F_cur_batch[0].data)
                    f_out = open("./LOSS.OUT", "a")
                    print("Force check:\n", F_cur_batch.data[0], file=f_out)
                    f_out.close()
                loss_F_cur_batch = CRITERION(F_cur_batch, data_cur[3])

                loss_cur_batch = pref_e * loss_E_cur_batch + pref_f * loss_F_cur_batch
                OPTIMIZER2.zero_grad()
                loss_cur_batch.backward()
                OPTIMIZER2.step()
                # correct end
                ###Adam end

                """
                ###LBFGS
                #correct
                #correct end
                ###LBFGS end
                """

                END_BATCH_TIMER = time.time()

                ###Adam
                if (batch_idx  == 0 and epoch % 10 == 0):
                    #print(data_cur[8])
                    END_BATCH_USER_TIMER = time.time()
                    #print("Rank ", hvd.rank(), "Select frame:", data_cur[8])
                    print("Rank ", hvd.rank(), "Epoch: %-10d, Batch: %-10d, lossE: %10.6f eV/atom, lossF: %10.6f eV/A, time: %10.3f s" % (
                        epoch, batch_idx, tf.sqrt(loss_E_cur_batch) / data_cur[4][0], tf.sqrt(loss_F_cur_batch),
                    END_BATCH_USER_TIMER - START_BATCH_USER_TIMER))
                    #print("Rank ", hvd.rank(), "Select frame:", data_cur[8], file=f_out)
                    if (hvd.rank() == 0):
                        f_out = open("./LOSS.OUT", "a")
                        print("Rank ", hvd.rank(), "Epoch: %-10d, Batch: %-10d, lossE: %10.6f eV/atom, lossF: %10.6f eV/A, time: %10.3f s" % ( \
                           epoch, batch_idx, tf.sqrt(loss_E_cur_batch) / data_cur[4][0], tf.sqrt(loss_F_cur_batch),
                        END_BATCH_USER_TIMER - START_BATCH_USER_TIMER), \
                        file = f_out)
                        f_out.close()
                    START_BATCH_USER_TIMER = time.time()
                ###Adam end

                """
                ###LBFGS
                f_out = open("./LOSS.OUT", "a")
                print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.6f eV/A, time: %10.3f s" % (
                epoch, batch_idx, closure() / N_ATOMS[0], closure() / N_ATOMS[0], END_BATCH_TIMER - START_BATCH_TIMER))
                print("Epoch: %-10d, Batch: %-10d, lossE: %10.3f eV/atom, lossF: %10.3f eV/A, time: %10.3f s" % (\
                    epoch, batch_idx, closure() / N_ATOMS[0], closure() / N_ATOMS[0], END_BATCH_TIMER - START_BATCH_TIMER),\
                      file = f_out)
                f_out.close()
                ###LBFGS end
                """

                # print(COORD_Reshape_tf_cur, SYM_COORD_Reshape_tf_cur, ENERGY_tf_cur, FORCE_Reshape_tf_cur, N_ATOMS_tf_cur)

                STEP_CUR += 1

                """if (STEP_CUR >= 2):
                    break"""
            if (PROF_FLAG and hvd.rank() == 0):
                f_prof = open("./PROF.OUT", "w")
                print("profiling info saved in ./PROF.OUT")
                print(prof.table(sort_by="cpu_time"), file = f_prof)
                f_prof.close()
        END_EPOCH_TIMER = time.time()


        if (False):
            break

if (hvd.rank() == 0):
    torch.save(ONE_BATCH_NET.state_dict(), "./freeze_model.pytorch")
    print("Rank 0: Model saved to ./freeze_model.pytorch")
    f_out = open("./LOSS.OUT", "a")
    print("Rank 0: Model saved to ./freeze_model.pytorch", file = f_out)
    f_out.close()

END_TRAIN_TIMER = time.time()
ELAPSED_TRAIN = END_TRAIN_TIMER - START_TRAIN_TIMER
print("Training complete. Time elapsed: %10.3f s\n"%ELAPSED_TRAIN)
