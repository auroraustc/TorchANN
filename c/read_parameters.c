/*
2019.03.28 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Read in parameters from file.

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_PARAM

#ifdef DEBUG_PARAM
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int read_parameters(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    FILE * fp_param = NULL;
    int i, j, k;
    int N_Atoms_max;
    int NUM_FILTER_LAYER;
    int NUM_FITTING_LAYER;


    parameters_info->cutoff_1 = 7.7;
    parameters_info->cutoff_2 = 8.0;
    parameters_info->cutoff_3 = 0.0;
    parameters_info->cutoff_max = 8.0;

    N_Atoms_max = 0;
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        N_Atoms_max = (N_Atoms_max <= frame_info[i].N_Atoms ? frame_info[i].N_Atoms : N_Atoms_max);
        printf_d("N_Atoms: %d\n", frame_info[i].N_Atoms);
    }
    parameters_info->N_Atoms_max = N_Atoms_max;

    parameters_info->batch_size = 1;
    parameters_info->stop_epoch = 1;
    parameters_info->num_filter_layer = 3;
    parameters_info->filter_neuron = (int *)calloc(parameters_info->num_filter_layer, sizeof(int));
    for (i = 0; i <= parameters_info->num_filter_layer - 1; i++)
    {
        parameters_info->filter_neuron[i] = 16;
    }
    parameters_info->axis_neuron = 4;
    parameters_info->num_fitting_layer = 5;
    parameters_info->fitting_neuron = (int *)calloc(parameters_info->num_fitting_layer, sizeof(int));
    for (i = 0; i <= parameters_info->num_fitting_layer - 1; i++)
    {
        parameters_info->fitting_neuron[i] = 128;
    }
    parameters_info->start_lr = 0.0005;
    parameters_info->decay_steps = 1;
    parameters_info->decay_epoch = 1;
    parameters_info->decay_rate = 0.95;
    parameters_info->start_pref_e = 0.1;
    parameters_info->limit_pref_e = 1.0;
    parameters_info->start_pref_f = 10000.0;
    parameters_info->limit_pref_f = 1.0;
    parameters_info->check_step = 1000;
    parameters_info->check_batch = -1;
    parameters_info->check_epoch = -1;
    parameters_info->output_step = -1;
    parameters_info->output_batch = -1;
    parameters_info->output_epoch = 10;
    parameters_info->save_step = -1;
    parameters_info->save_batch = -1;
    parameters_info->save_epoch = 10;


    return 0;
}