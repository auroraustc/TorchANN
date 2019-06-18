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

int main()
{
    int read_parameters(frame_info_struct * frame_info, parameters_info_struct * parameters_info, char * filename);
    int save_to_file_parameters(parameters_info_struct * parameters_info);

    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));

    int error_code = 0;
    int read_parameters_info_flag;
    int save_to_file_parameters_flag;

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available())
    {
        printf("GPU available! Training on GPU.\n");
        device = torch::Device(torch::kCUDA);
    }

    read_parameters_info_flag = read_parameters(NULL, parameters_info, "./ALL_PARAMSS.json");
    if (read_parameters_info_flag != 0)
    {
        printf("Error when reading input parameters: read_parameters_info_flag = %d\n", read_parameters_info_flag);
        return read_parameters_info_flag;
    }
    printf("No error when reading parameters.\n");

    save_to_file_parameters_flag = save_to_file_parameters(parameters_info);
    if (save_to_file_parameters_flag != 0)
    {
        printf("Error when saving parameters to file: save_to_file_parameters_flag = %d\n", save_to_file_parameters_flag);
        return save_to_file_parameters_flag;
    }
    printf("No error when saving parameters to file.\n");

    std::vector<torch::Tensor> tmp(2);
    tmp[0] = torch::ones({4, 5});
    tmp[1] = torch::randn({4, 5});
    auto a = torch::data::datasets::TensorDataset(tmp);
    std::cout << a.get(0).data << std::endl << a.get(1).data <<std::endl;
    tmp[0] = torch::ones({4,6});
    auto b = torch::data::datasets::ANN_TensorDataset(tmp);

    std::cout << b.size() << std::endl;
    std::cout << b.get(0).target << std::endl;// << b.get(1) << std::endl;

     

    /*free all the data*/
        /*parameters*/
    free(parameters_info->filter_neuron);
    free(parameters_info->fitting_neuron);
    free(parameters_info->type_index_all_frame);
    free(parameters_info); 

    return 0;
}