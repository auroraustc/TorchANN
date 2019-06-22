/*
2019.06.22 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Compute descriptors and derivatives.(GPU version)
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>
#include "../../c/struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_CONV_EXT

#ifdef DEBUG_CONV_EXT
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

#define warpSize 32
#define PI 3.141592653589793238462643383279

__device__ double s_r(double r_ij, double r_c, double r_cs)
{
    double result;
    double rc = r_c;
    double rcs = r_cs;
    result = (r_ij >= rc) ? 0 : ((r_ij >= rcs) ? 1 / r_ij * (0.5 * cos((r_ij - rcs) / (rc - rcs) * PI) + 0.5) : 1 / r_ij);
    return result;
}

__device__ double fastpow2(double number, int dummy)
{
    return number * number;
}

__global__ void calc_descrpt_and_deriv_DPMD_kernel(int Nframes_tot, int N_Atoms_max, int SEL_A_max, double rc, double rcs, torch::PackedTensorAccessor<double, 3> COORD_RESHAPE_ACCESSOR, torch::PackedTensorAccessor<double, 4> NEI_COORD_RESHAPE_ACCESSOR, torch::PackedTensorAccessor<double, 3> SYM_COORD_DPMD_RESHAPE_ACCESSOR, torch::PackedTensorAccessor<double, 3> SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR, torch::PackedTensorAccessor<double, 3> SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR, torch::PackedTensorAccessor<double, 3> SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR)
{
    int frame_loop = blockIdx.x;
    int atom_loop = threadIdx.x;
    int k;
    int i = frame_loop, j = atom_loop;
    int l;
    double r_c = rc;
    double r_cs = rcs;

    if ((frame_loop >= Nframes_tot) || (atom_loop >= N_Atoms_max))
    {
        return;
    }

    for (k = 0; k <= SEL_A_max - 1; k++)
    {
        double four_coord[4];
        double r_ij;
        double atom_coord[3] = {COORD_RESHAPE_ACCESSOR[i][j][0], COORD_RESHAPE_ACCESSOR[i][j][1], COORD_RESHAPE_ACCESSOR[i][j][2]};
        double nei_coord[3] = {NEI_COORD_RESHAPE_ACCESSOR[i][j][k][0], NEI_COORD_RESHAPE_ACCESSOR[i][j][k][1], NEI_COORD_RESHAPE_ACCESSOR[i][j][k][2]};
        double r_ji_coord[3] = {nei_coord[0] - atom_coord[0], nei_coord[1] - atom_coord[1], nei_coord[2] - atom_coord[2]};

        r_ij = sqrt(fastpow2(atom_coord[0] - nei_coord[0], 2) + fastpow2(atom_coord[1] - nei_coord[1], 2) + fastpow2(atom_coord[2] - nei_coord[2], 2));
        four_coord[0] = s_r(r_ij, r_c, r_cs);
        four_coord[1] = four_coord[0] * r_ji_coord[0] / r_ij; four_coord[2] = four_coord[0] * r_ji_coord[1] / r_ij; four_coord[3] = four_coord[0] * r_ji_coord[2] / r_ij;
        
        for (l = 0; l <= 3; l++)
        {
            int idx_sym = k * 4 + l;
            SYM_COORD_DPMD_RESHAPE_ACCESSOR[i][j][idx_sym] = four_coord[l];
        }
        if (r_ij >= rc)
        {
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 0] = 0;
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 0;
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 0;
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 0;
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 0] = 0;
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 0;
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 0;
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 0;
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 0] = 0;
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 0;
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 0;
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 0;
        }
        else if (r_ij > rcs)
        {
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 0] = r_ji_coord[0] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij);
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[0] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) - (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[0] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 2.0 * r_ji_coord[0] * r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 2.0 * r_ji_coord[0] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 0] = r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij) + PI * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) - (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij) + PI * r_ji_coord[1] * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 2.0 * r_ji_coord[1] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[1] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 0] = r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij) + PI * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[1] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 2.0 * r_ji_coord[2] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) - (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij) + PI * r_ji_coord[2] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
        }
        else
        {
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 0] = r_ji_coord[0] / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[0] / (r_ij * r_ij * r_ij * r_ij) - 1.0 / (r_ij * r_ij);
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 2.0 * r_ji_coord[0] * r_ji_coord[1] / (r_ij * r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 2.0 * r_ji_coord[0] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 0] = r_ji_coord[1] / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[1] / (r_ij * r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[1] / (r_ij * r_ij * r_ij * r_ij) - 1.0 / (r_ij * r_ij);
            SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 2.0 * r_ji_coord[1] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 0] = r_ji_coord[2] / (r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
            SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR[i][j][4 * k + 3] = 2.0 * r_ji_coord[2] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij) - 1.0 / (r_ij * r_ij);
        }
    }
}

std::vector<torch::Tensor> calc_descrpt_and_deriv_DPMD(torch::Tensor COORD, torch::Tensor NEI_COORD, torch::Tensor NEI_DIST, int Nframes_tot, int N_Atoms_max, int SEL_A_max, double rcs, double rc)
{

    torch::Tensor SYM_COORD_DPMD_RESHAPE = torch::zeros({Nframes_tot, N_Atoms_max, SEL_A_max * 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    torch::Tensor SYM_COORD_DPMD_DX_RESHAPE = torch::zeros({Nframes_tot, N_Atoms_max, SEL_A_max * 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    torch::Tensor SYM_COORD_DPMD_DY_RESHAPE = torch::zeros({Nframes_tot, N_Atoms_max, SEL_A_max * 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    torch::Tensor SYM_COORD_DPMD_DZ_RESHAPE = torch::zeros({Nframes_tot, N_Atoms_max, SEL_A_max * 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

    torch::Tensor SYM_COORD_DPMD;
    torch::Tensor SYM_COORD_DPMD_DX;
    torch::Tensor SYM_COORD_DPMD_DY;
    torch::Tensor SYM_COORD_DPMD_DZ;

    torch::Tensor COORD_RESHAPE = torch::reshape(COORD, {Nframes_tot, N_Atoms_max, 3});
    torch::Tensor NEI_COORD_RESHAPE = torch::reshape(NEI_COORD, {Nframes_tot, N_Atoms_max, SEL_A_max, 3});
    auto COORD_RESHAPE_ACCESSOR = COORD_RESHAPE.packed_accessor<double, 3>();
    auto NEI_COORD_RESHAPE_ACCESSOR = NEI_COORD_RESHAPE.packed_accessor<double, 4>();
    auto SYM_COORD_DPMD_RESHAPE_ACCESSOR = SYM_COORD_DPMD_RESHAPE.packed_accessor<double, 3>();
    auto SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR = SYM_COORD_DPMD_DX_RESHAPE.packed_accessor<double, 3>();
    auto SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR = SYM_COORD_DPMD_DY_RESHAPE.packed_accessor<double, 3>();
    auto SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR = SYM_COORD_DPMD_DZ_RESHAPE.packed_accessor<double, 3>();

    int i = 0, j = 0, k = 0, l = 0;
    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));
    parameters_info->cutoff_1 = rcs;
    parameters_info->cutoff_2 = rc;

    calc_descrpt_and_deriv_DPMD_kernel<<<Nframes_tot, (N_Atoms_max / 32 + 1) * 32>>>(Nframes_tot, N_Atoms_max, SEL_A_max, rc, rcs, COORD_RESHAPE_ACCESSOR, NEI_COORD_RESHAPE_ACCESSOR, SYM_COORD_DPMD_RESHAPE_ACCESSOR, SYM_COORD_DPMD_DX_RESHAPE_ACCESSOR, SYM_COORD_DPMD_DY_RESHAPE_ACCESSOR, SYM_COORD_DPMD_DZ_RESHAPE_ACCESSOR);

    free(parameters_info);
    SYM_COORD_DPMD = torch::reshape(SYM_COORD_DPMD_RESHAPE, {Nframes_tot, -1});
    SYM_COORD_DPMD_DX = torch::reshape(SYM_COORD_DPMD_DX_RESHAPE, {Nframes_tot, -1});
    SYM_COORD_DPMD_DY = torch::reshape(SYM_COORD_DPMD_DY_RESHAPE, {Nframes_tot, -1});
    SYM_COORD_DPMD_DZ = torch::reshape(SYM_COORD_DPMD_DZ_RESHAPE, {Nframes_tot, -1});

    return {SYM_COORD_DPMD, SYM_COORD_DPMD_DX, SYM_COORD_DPMD_DY, SYM_COORD_DPMD_DZ};

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("calc_descrpt_and_deriv_DPMD", &calc_descrpt_and_deriv_DPMD, "TEST calc_descrpt_and_deriv_DPMD");
}