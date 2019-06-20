/*
2019.06.18 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Define all the struct used in the c_train code.
*/

#include <torch/torch.h>
#include <torch/types.h>

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
