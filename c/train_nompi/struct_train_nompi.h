/*
2019.06.18 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Define all the struct used in the c_train code.
*/

#include <torch/torch.h>

typedef struct input_bin_files_struct_
{
    torch::Tensor COORD;
    torch::Tensor SYM_COORD;
    torch::Tensor ENERGY;
    torch::Tensor FORCE;
    torch::Tensor N_ATOMS;
    torch::Tensor TYPE;
    torch::Tensor NEI_IDX;
    torch::Tensor NEI_COORD;
    torch::Tensor FRAME_IDX;
    torch::Tensor SYM_COORD_DX;
    torch::Tensor SYM_COORD_DY;
    torch::Tensor SYM_COORD_DZ;
    torch::Tensor N_ATOMS_ORI;
    torch::Tensor NEI_TYPE;
}input_bin_files_struct;


namespace torch {
namespace data {
namespace datasets {
///Adapted from torch::data::datasets::TensorDataset
/// A dataset of tensors.
/// Stores a single tensor internally, which is then indexed inside `get()`.
struct ANN_TensorDataset : public Dataset<ANN_TensorDataset> {
    private:
        torch::Tensor tensor1, tensor2;
        
    public:
        /// Creates a `TensorDataset` from a vector of tensors.
        explicit ANN_TensorDataset(const std::vector<Tensor>& tensors)
            : tensor1(tensors[0]), tensor2(tensors[1]) {}

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
}
}
}
