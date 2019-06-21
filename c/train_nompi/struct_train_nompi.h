/*
2019.06.18 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Define all the struct used in the c_train code.
*/

#include <torch/torch.h>
#include <torch/types.h>
#include "../struct.h"

//template<typename Data = torch::Tensor, typename Target = torch::Tensor>
typedef struct input_bin_files_struct_
{
    
    torch::Tensor COORD;//Nframes_tot * N_Atoms_max * 3 (double)
    torch::Tensor SYM_COORD;//Nframes_tot * N_Atoms_max * N_sym_coord (double)
    torch::Tensor ENERGY;//Nframes_tot * 1 (double) (Target)
    torch::Tensor FORCE;//Nframes_tot * N_Atoms_max * 3 (double) (Target)
    torch::Tensor N_ATOMS;//Nframes_tot * 1(int)
    torch::Tensor TYPE;//Nframes_tot * N_Atoms_max (int)
    torch::Tensor NEI_IDX;//Nframes_tot * N_Atoms_max * SEL_A_max (int)
    torch::Tensor NEI_COORD;//Nframes_tot * N_Atoms_max * SEL_A_max * 3 (double)
    torch::Tensor FRAME_IDX;//Nframes_tot * 1 (int)
    torch::Tensor SYM_COORD_DX;//Nframes_tot * N_Atoms_max * N_sym_coord (double)
    torch::Tensor SYM_COORD_DY;//Nframes_tot * N_Atoms_max * N_sym_coord (double)
    torch::Tensor SYM_COORD_DZ;//Nframes_tot * N_Atoms_max * N_sym_coord (double)
    torch::Tensor N_ATOMS_ORI;//Nframes_tot * 1 (int)
    torch::Tensor NEI_TYPE;//Nframes_tot * N_Atoms_max * SEL_A_max (int)
}input_bin_files_struct;




namespace torch {
namespace data {


    template <typename Data = Tensor, typename Target1 = Tensor, typename Target2 = Tensor>
    struct ANN_Example 
    {
    // using DataType = Data;
    // using TargetType = Target;

    ANN_Example() = default;
    ANN_Example(Data coord, Target1 energy, Target2 force)
        : COORD(std::move(coord)), ENERGY(std::move(energy)), FORCE(std::move(force)) {}

    Data COORD;
    Target1 ENERGY;
    Target2 FORCE;
    };


namespace datasets {
///Adapted from torch::data::datasets::TensorDataset
/// A dataset of tensors.
/// Stores a single tensor internally, which is then indexed inside `get()`.
    struct ANN_TensorDataset : public Dataset<ANN_TensorDataset> 
    {
        private:
            torch::Tensor tensor1, tensor2;
            
        public:
            /// Creates a `TensorDataset` from a vector of tensors.
            explicit ANN_TensorDataset(std::vector<torch::Tensor> tensors)//(const torch::Tensor tensors1, const torch::Tensor tensors2)
                : tensor1(tensors[0]), tensor2(tensors[1]) {}//tensor1(tensors1), tensor2(tensors2) {}
    
            //explicit ANN_TensorDataset(torch::Tensor tensor) : tensor(std::move(tensor)) {}

            /// Returns a single `TensorExample`.
            torch::data::Example<> get(size_t index) override {
                return {tensor1[index], tensor2[index] };
            }

            /// Returns the number of tensors in the dataset.
            optional<size_t> size() const override {
                return tensor1.sizes()[0];
            }

    //Tensor tensor;
    };
//std::vector<torch::data::datasets::SingleExample>
    struct ANN_TensorDataset_2 : public Dataset<ANN_TensorDataset> 
    {
        private:
            torch::Tensor tensor1, tensor2, tensor3;
            
        public:
            /// Creates a `TensorDataset` from a vector of tensors.
            explicit ANN_TensorDataset_2(input_bin_files_struct input_bin)//(const torch::Tensor tensors1, const torch::Tensor tensors2)
                : tensor1(input_bin.COORD), tensor2(input_bin.ENERGY), tensor3(input_bin.FORCE) {}//tensor1(tensors1), tensor2(tensors2) {}
    
            //explicit ANN_TensorDataset(torch::Tensor tensor) : tensor(std::move(tensor)) {}

            /// Returns a single `TensorExample`.
            torch::data::Example<> get(size_t index) override ;

            /// Returns the number of tensors in the dataset.
            optional<size_t> size() const override {
                return tensor1.sizes()[0];
            }
    //Tensor tensor;
    };


}
}
}

// struct one_batch_net_1 : torch::nn::Module
// {
//     // one_batch_net(parameters_info_struct * parameters_info, double * mean_init):
//     // {
//     //     1;
//     // };
//     one_batch_net_1(): 
//         conv1(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)),
//         conv2(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)),
//         fc1(320, 50),
//         fc2(50, 10) {
//     register_module("conv1", conv1);
//     register_module("conv2", conv2);
//     register_module("conv2_drop", conv2_drop);
//     register_module("fc1", fc1);
//     register_module("fc2", fc2);}
//     torch::nn::Conv2d conv1;
//     torch::nn::Conv2d conv2;
//     torch::nn::FeatureDropout conv2_drop;
//     torch::nn::Linear fc1;
//     torch::nn::Linear fc2;
// };

struct one_batch_net : torch::nn::Module
{
    /*Init*/
    one_batch_net(parameters_info_struct * parameters_info, double * mean_init)
    {
        int num_of_types = parameters_info->N_types_all_frame;
        int num_of_filter_layers = parameters_info->num_filter_layer;
        int num_of_fitting_layers = parameters_info->num_fitting_layer;
        int i = 0, j = 0, k = 0;
        char name_filter_input[] = "filter_input_0_0\0";
        char name_filter_hidden[] = "filter_hidden_0_0_0\0";
        char name_fitting_input[] = "fitting_input_0\0";
        char name_fitting_hidden[] = "fitting_hidden_0_0\0";
        char name_fitting_out[] = "fitting_out_0\0";
        /*allocate size for each layer*/
        filter_input.resize(num_of_types);
        filter_hidden.resize(num_of_types);
        fitting_input.assign(num_of_types, nullptr);
        //fitting_input.resize(num_of_types);
        fitting_hidden.resize(num_of_types);
        //fitting_out.resize(num_of_types);
        fitting_out.assign(num_of_types, nullptr);
        for (i = 0; i <= num_of_types - 1; i++)
        {
            //filter_input[i].resize(num_of_types);
            filter_input[i].assign(num_of_types, nullptr);
            filter_hidden[i].resize(num_of_types);
            //fitting_hidden[i].resize(num_of_fitting_layers - 1);//Note: DO NOT -2 ! The last layer outputing N*1 is not defined in the json file.
            fitting_hidden[i].assign(num_of_fitting_layers - 1, nullptr);
        }
        for (i = 0; i <= num_of_types - 1; i++)
        {
            for (j = 0; j <= num_of_types - 1; j++)
            {
                //filter_hidden[i][j].resize(num_of_filter_layers - 1);
                filter_hidden[i][j].assign(num_of_filter_layers - 1, nullptr);
            }
        }
        /*filter net*/
        //TORCH_MODULE(Linear);
        //torch::nn::ModuleHolder;
        for (i = 0; i <= num_of_types - 1; i++)
        {
            for (j = 0; j <= num_of_types - 1; j++)
            {
                name_filter_input[13] = '0' + i;
                name_filter_input[15] = '0' + j;
                std::cout << name_filter_input << std::endl;
                filter_input[i][j] = register_module(name_filter_input, torch::nn::Linear(1, parameters_info->filter_neuron[0]));
                for (k = 0; k <= num_of_filter_layers - 2; k++)
                {
                    name_filter_hidden[14] = '0' + i;
                    name_filter_hidden[16] = '0' + j;
                    name_filter_hidden[18] = '0' + k;
                    std::cout << name_filter_hidden << std::endl;
                    filter_hidden[i][j][k] = register_module(name_filter_hidden, torch::nn::Linear(parameters_info->filter_neuron[k], parameters_info->filter_neuron[k + 1]));
                }
            }
        }
        /*fitting net*/
        for (i = 0; i <= num_of_types - 1; i++)
        {
            name_fitting_input[14] = '0' + i;
            name_fitting_out[12] = '0' + i;
            std::cout << name_fitting_input << std::endl;
            std::cout << name_fitting_out << std::endl;
            fitting_input[i] = register_module(name_fitting_input, torch::nn::Linear(parameters_info->axis_neuron * parameters_info->filter_neuron[num_of_filter_layers - 1], parameters_info->fitting_neuron[0]));
            fitting_out[i] = register_module(name_fitting_out, torch::nn::Linear(parameters_info->fitting_neuron[num_of_fitting_layers - 1], 1));
            for (j = 0; j <= num_of_fitting_layers - 2; j++)
            {
                name_fitting_hidden[15] = '0' + i;
                name_fitting_hidden[17] = '0' + j;
                std::cout << name_fitting_hidden << std::endl;
                fitting_hidden[i][j] = register_module(name_fitting_hidden, torch::nn::Linear(parameters_info->fitting_neuron[j], parameters_info->fitting_neuron[j + 1]));
            }
        }


    }

//Incomplete
    /*forward*/
    torch::Tensor forward(torch::Tensor input)
    {
        return fitting_hidden[0][1](torch::tanh(fitting_hidden[0][0](input)));
    }
    /*Why 2-D for fitting_hidden: dim 0 = type_idx, dim 1 = hidden_idx for hidden layers*/
    /*Why 3-D for filter_hidden: dim 0 = type_idx, dim 1 = nei_type_idx, dim 2 = hidden_idx for hidden layers*/
    std::vector<std::vector<torch::nn::Linear>> filter_input;//type_idx, nei_type_idx
    std::vector<std::vector<std::vector<torch::nn::Linear>>> filter_hidden;//type_idx, nei_type_idx, hidden_idx
    std::vector<torch::nn::Linear> fitting_input = {nullptr};//type_idx
    std::vector<std::vector<torch::nn::Linear>> fitting_hidden;//type_idx, hidden_idx
    std::vector<torch::nn::Linear> fitting_out;//type_idx
};


