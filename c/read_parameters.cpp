/*
2019.03.28 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Read in parameters from file.

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include "struct.h"
#include "template_func.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_PARAM

#ifdef DEBUG_PARAM
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

/*Read the array values in a json file and returns a std::vector<T>*/

int read_parameters(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    

    FILE * fp_param = NULL;
    int i, j, k;
    int N_Atoms_max;
    int NUM_FILTER_LAYER;
    int NUM_FITTING_LAYER;
    FILE * tmp;
    boost::property_tree::ptree PARAMS;
    boost::property_tree::ptree PARAMS_ITEMS;
    tmp = fopen("./PARAMS.json", "r");
    if (tmp == NULL)
    {
        printf("Please provide PARAMS.json!\n");
        return 21;
    }
    fclose(tmp);
    boost::property_tree::read_json<boost::property_tree::ptree>("./PARAMS.json", PARAMS);

    parameters_info->cutoff_1 = PARAMS.get<double>("cutoff_1");
    parameters_info->cutoff_2 = PARAMS.get<double>("cutoff_2");
    parameters_info->cutoff_3 = PARAMS.get<double>("cutoff_3");
    parameters_info->cutoff_max = PARAMS.get<double>("cutoff_max");
    parameters_info->sym_coord_type = PARAMS.get<int>("sym_coord_type");
    if ((parameters_info->sym_coord_type != 1) && (parameters_info->sym_coord_type != 2))
    {
        printf("sym_coord_type %d not supported!\n", parameters_info->sym_coord_type);
        return 22;
    }

    N_Atoms_max = 0;
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        N_Atoms_max = (N_Atoms_max <= frame_info[i].N_Atoms ? frame_info[i].N_Atoms : N_Atoms_max);
        printf_d("N_Atoms: %d\n", frame_info[i].N_Atoms);
    }
    parameters_info->N_Atoms_max = N_Atoms_max;

    parameters_info->batch_size = PARAMS.get<int>("batch_size");
    parameters_info->stop_epoch = PARAMS.get<int>("stop_epoch");
    parameters_info->num_filter_layer = PARAMS.get<int>("num_filter_layer");
    std::vector<int> FILTER_NEURON = as_vector<int>(PARAMS, "filter_neuron");
    if (FILTER_NEURON.size() != parameters_info->num_filter_layer)
    {
        parameters_info->num_filter_layer = FILTER_NEURON.size();
    }
    parameters_info->filter_neuron = (int *)calloc(parameters_info->num_filter_layer, sizeof(int));
    for (i = 0; i <= parameters_info->num_filter_layer - 1; i++)
    {
        parameters_info->filter_neuron[i] = FILTER_NEURON[i];
    }
    parameters_info->axis_neuron = PARAMS.get<int>("axis_neuron");
    parameters_info->num_fitting_layer = PARAMS.get<int>("num_fitting_layer");
    std::vector<int> FITTING_NEURON = as_vector<int>(PARAMS, "fitting_neuron");
    if (FITTING_NEURON.size() != parameters_info->num_fitting_layer)
    {
        parameters_info->num_fitting_layer = FITTING_NEURON.size();
    }
    parameters_info->fitting_neuron = (int *)calloc(parameters_info->num_fitting_layer, sizeof(int));
    for (i = 0; i <= parameters_info->num_fitting_layer - 1; i++)
    {
        parameters_info->fitting_neuron[i] = FITTING_NEURON[i];
    }
    parameters_info->start_lr = PARAMS.get<double>("start_lr");
    parameters_info->decay_steps = -1;
    parameters_info->decay_epoch = PARAMS.get<int>("decay_epoch");
    parameters_info->decay_rate = PARAMS.get<double>("decay_rate");
    parameters_info->start_pref_e = PARAMS.get<double>("start_pref_e");
    parameters_info->limit_pref_e = PARAMS.get<double>("limit_pref_e");
    parameters_info->start_pref_f = PARAMS.get<double>("start_pref_f");
    parameters_info->limit_pref_f = PARAMS.get<double>("limit_pref_f");
    parameters_info->check_step = PARAMS.get<int>("check_step");
    parameters_info->check_batch = -1;
    parameters_info->check_epoch = -1;
    parameters_info->output_step = -1;
    parameters_info->output_batch = -1;
    parameters_info->output_epoch = PARAMS.get<int>("output_epoch");
    parameters_info->save_step = -1;
    parameters_info->save_batch = -1;
    parameters_info->save_epoch = PARAMS.get<int>("save_epoch");

    // fp_param = fopen("./INPUT.raw","r");
    // if (fp_param != NULL)
    // {
    //     int sym_coord_type;
    //     fscanf(fp_param, "%d", &sym_coord_type);
    //     printf_d("sym_coord_type read from file: %d\n", sym_coord_type);
    //     parameters_info->sym_coord_type = (((sym_coord_type == 1) || (sym_coord_type == 2)) ? sym_coord_type : parameters_info->sym_coord_type);
    //     fclose(fp_param);
    // }
    // else
    // {
    //     parameters_info->sym_coord_type = 1;
    // }
    
    

    return 0;
}