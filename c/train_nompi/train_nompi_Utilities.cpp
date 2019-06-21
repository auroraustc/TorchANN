/*
2019.06.18 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Utilities functions for training.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tuple>
#include <torch/torch.h>
#include "struct_train_nompi.h"


int read_bin_files(input_bin_files_struct * input_bin_files, parameters_info_struct * parameters_info)
{
    FILE * fp_tmp = NULL;
    
    fp_tmp = fopen("./COORD.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("COORD.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->COORD = torch::from_file("./COORD.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * 3, torch::TensorOptions().dtype(torch::kFloat64));
    }

    fp_tmp = fopen("./SYM_COORD.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("SYM_COORD.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->SYM_COORD = torch::from_file("./SYM_COORD.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * parameters_info->N_sym_coord, torch::TensorOptions().dtype(torch::kFloat64));
    }

    fp_tmp = fopen("./ENERGY.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("ENERGY.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->ENERGY = torch::from_file("./ENERGY.BIN", true, parameters_info->Nframes_tot * 1, torch::TensorOptions().dtype(torch::kFloat64));
    }

    fp_tmp = fopen("./FORCE.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("FORCE.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->FORCE = torch::from_file("./FORCE.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * 3, torch::TensorOptions().dtype(torch::kFloat64));
    }

    fp_tmp = fopen("./N_ATOMS.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("N_ATOMS.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->N_ATOMS = torch::from_file("./N_ATOMS.BIN", true, parameters_info->Nframes_tot * 1, torch::TensorOptions().dtype(torch::kInt));
    }

    fp_tmp = fopen("./TYPE.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("TYPE.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->TYPE = torch::from_file("./TYPE.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max, torch::TensorOptions().dtype(torch::kInt));
    }

    fp_tmp = fopen("./NEI_IDX.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("NEI_IDX.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->NEI_IDX = torch::from_file("./NEI_IDX.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * parameters_info->SEL_A_max, torch::TensorOptions().dtype(torch::kInt));
    }

    fp_tmp = fopen("./NEI_COORD.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("NEI_COORD.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->NEI_COORD = torch::from_file("./NEI_COORD.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * parameters_info->SEL_A_max * 3, torch::TensorOptions().dtype(torch::kFloat64));
    }

    // fp_tmp = fopen("./FRAME_IDX.BIN", "rb");
    // if (fp_tmp == NULL)
    // {
    //     printf("FRAME_IDX.BIN does not exist!\n");
    //     return 21; 
    // }
    // else
    // {
    //     fclose(fp_tmp);
    //     input_bin_files->FRAME_IDX = torch::from_file("./FRAME_IDX.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * 3, torch::TensorOptions().dtype(torch::kFloat64));
    // }
    input_bin_files->FRAME_IDX = torch::arange(parameters_info->Nframes_tot, torch::TensorOptions().dtype(torch::kInt));

    fp_tmp = fopen("./SYM_COORD_DX.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("SYM_COORD_DX.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->SYM_COORD_DX = torch::from_file("./SYM_COORD_DX.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * parameters_info->N_sym_coord, torch::TensorOptions().dtype(torch::kFloat64));
    }

    fp_tmp = fopen("./SYM_COORD_DY.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("SYM_COORD_DY.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->SYM_COORD_DY = torch::from_file("./SYM_COORD_DY.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * parameters_info->N_sym_coord, torch::TensorOptions().dtype(torch::kFloat64));
    }

    fp_tmp = fopen("./SYM_COORD_DZ.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("SYM_COORD_DZ.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->SYM_COORD_DZ = torch::from_file("./SYM_COORD_DZ.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * parameters_info->N_sym_coord, torch::TensorOptions().dtype(torch::kFloat64));
    }

    fp_tmp = fopen("./N_ATOMS_ORI.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("N_ATOMS_ORI.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->N_ATOMS_ORI = torch::from_file("./N_ATOMS_ORI.BIN", true, parameters_info->Nframes_tot * 1, torch::TensorOptions().dtype(torch::kInt));
    }

    fp_tmp = fopen("./NEI_TYPE.BIN", "rb");
    if (fp_tmp == NULL)
    {
        printf("NEI_TYPE.BIN does not exist!\n");
        return 21; 
    }
    else
    {
        fclose(fp_tmp);
        input_bin_files->NEI_TYPE = torch::from_file("./NEI_TYPE.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max * parameters_info->SEL_A_max, torch::TensorOptions().dtype(torch::kInt));
    }

    
    return 0;
}

int test()
{
    int read_parameters(frame_info_struct * frame_info, parameters_info_struct * parameters_info, char * filename);
    int save_to_file_parameters(parameters_info_struct * parameters_info);

    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));

    int error_code = 0;
    int read_parameters_info_flag;
    int save_to_file_parameters_flag;
    const int HIDDEN = 128;
    FILE * fp_tmp_file = NULL;
    int file_flag = 0;

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
    tmp[0] = torch::randn({4,6});
    auto b = torch::data::datasets::ANN_TensorDataset(tmp).map(torch::data::transforms::Stack<>());
    std::cout << b.size() << std::endl;
    std::cout << ' ' << std::endl;// << b.get(1) << std::endl;

    torch::nn::Sequential Net
    (
      torch::nn::Linear(torch::nn::LinearOptions(6, HIDDEN).with_bias(true)),
      torch::nn::Functional(torch::tanh),
      torch::nn::Linear(torch::nn::LinearOptions(HIDDEN, 5).with_bias(true))
    );

    auto b_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(b), torch::data::DataLoaderOptions(4));

    torch::optim::Adam b_optimizer(Net->parameters(), torch::optim::AdamOptions(1E-4));

    for (int64_t epoch = 0; epoch <= 1000; epoch++)
    {
        int64_t batch_idx = 0;
        for (auto & batch : *b_loader)
        {
            //std::cout << batch.data << std::endl << batch.target << std::endl;
            //std::cout << batch.data << std::endl;
            torch::Tensor result = Net->forward(batch.data);
            //std::cout << result << std::endl;
            torch::Tensor loss = torch::mse_loss(result, batch.target);
            b_optimizer.zero_grad();
            loss.backward();
            b_optimizer.step();
            std::cout << "loss:" << loss.item<float>() << std::endl;
        }
    }

    file_flag = 1;
    fp_tmp_file = fopen("COORD.BIN", "rb");
    if (fp_tmp_file == NULL)
    {
        file_flag = 0;
    }
    else
    {
        fclose(fp_tmp_file);
    }
    fp_tmp_file = fopen("ENERGY.BIN", "rb");
    if (fp_tmp_file == NULL)
    {
        file_flag = 0;
    }
    else
    {
        fclose(fp_tmp_file);
    }
    if (file_flag == 1)
    {
        torch::Tensor COORD = torch::from_file("COORD.BIN", true, parameters_info->Nframes_tot * parameters_info->N_Atoms_max, torch::kFloat64);
        torch::Tensor ENERGY = torch::from_file("ENERGY.BIN", true, parameters_info->Nframes_tot, torch::kFloat64);
        std::cout << "coord:" << COORD.sizes() << std::endl << "energy:" << ENERGY.sizes() << std::endl;
    }

    std::cout << sizeof(torch::nn::Linear) << std::endl;
    one_batch_net model(parameters_info, NULL);
    torch::Tensor aa = torch::randn({128,128});
    for (const auto& pair : model.named_parameters()) 
    {
        std::cout << pair.key() << ": " << pair.value().sizes() << std::endl;
    }  
    std::cout << model.forward(aa) << std::endl;
     

    /*free all the data*/
        /*parameters*/
    free(parameters_info->filter_neuron);
    free(parameters_info->fitting_neuron);
    free(parameters_info->type_index_all_frame);
    free(parameters_info); 

    return 0;
}

int test_2()
{
    std::cout << sizeof(torch::nn::Linear) << std::endl;
    //one_batch_net model = nullptr;
}


