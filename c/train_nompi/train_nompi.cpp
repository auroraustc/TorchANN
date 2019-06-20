/*
2019.06.18 by Aurora. Contact:fanyi@mail.ustc.edu.cn

cpp version of training using libtorch.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <torch/torch.h>
#include "struct_train_nompi.h"
#include "../struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_MAIN

#ifdef DEBUG_MAIN
#define printf_d printf
#else
#define printf_d //
#endif


/***************MACRO FOR DEBUG END***************/

int main()
{
    int test();

    int read_parameters(frame_info_struct * frame_info, parameters_info_struct * parameters_info, char * filename);
    int read_bin_files(input_bin_files_struct * input_bin_files, parameters_info_struct * parameters_info);

    int read_parameters_flag = 0;
    int read_bin_files_flag = 0;

    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));
    input_bin_files_struct input_bin_files;


    read_parameters_flag = read_parameters(NULL, parameters_info, "./ALL_PARAMSS.json");
    if (read_parameters_flag != 0)
    {
        printf("Error when reading input parameters: read_parameters_info_flag = %d\n", read_parameters_flag);
        return read_parameters_flag;
    }
    printf("No error when reading parameters.\n");

    read_bin_files_flag = read_bin_files(&input_bin_files, parameters_info);
    if (read_bin_files_flag != 0)
    {
        printf("Error when reading binary files: read_bin_file_flag = %d\n", read_bin_files_flag);
        return read_bin_files_flag;
    }
    printf("No error when reading binary files.\n");
#ifdef DEBUG_MAIN
    printf("Check tensors:\n");
    std::cout << input_bin_files.COORD.size(0) << std::endl;
    std::cout << input_bin_files.SYM_COORD.size(0) << std::endl;
    std::cout << input_bin_files.ENERGY.size(0) << std::endl;
    std::cout << input_bin_files.FORCE.size(0) << std::endl;
    std::cout << input_bin_files.N_ATOMS.size(0) << std::endl;
    std::cout << input_bin_files.TYPE.size(0) << std::endl;
    std::cout << input_bin_files.NEI_IDX.size(0) << std::endl;
    std::cout << input_bin_files.NEI_COORD.size(0) << std::endl;
    std::cout << input_bin_files.FRAME_IDX.size(0) << std::endl;
    std::cout << input_bin_files.SYM_COORD_DX.size(0) << std::endl;
    std::cout << input_bin_files.SYM_COORD_DY.size(0) << std::endl;
    std::cout << input_bin_files.SYM_COORD_DZ.size(0) << std::endl;
    std::cout << input_bin_files.N_ATOMS_ORI.size(0) << std::endl;
    std::cout << input_bin_files.NEI_TYPE.size(0) << std::endl;
#endif
    std::cout << parameters_info->N_Atoms_max << std::endl;
    std::cout << torch::eq(torch::reshape(input_bin_files.TYPE, {1, parameters_info->N_Atoms_max}), 1).nonzero() << std::endl;

    //test();
    /*free all the data*/
        /*parameters*/
    free(parameters_info->filter_neuron);
    free(parameters_info->fitting_neuron);
    free(parameters_info->type_index_all_frame);
    free(parameters_info);
    return 0;
}

