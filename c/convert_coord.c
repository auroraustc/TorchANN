/*
2019.03.29 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Convert and generate symmetry coordinates.
coord_type:
    1: DeePMD-type coordinate

Return code:
    0: No errors.
    1: sym_coord_type error.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_CONV

#ifdef DEBUG_CONV
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int convert_coord(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, int sym_coord_type, void ** sym_coord)
{
    int convert_coord_DeePMD(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, void ** sym_coord);

    int i, j, k;
    int error_code;

    switch (sym_coord_type)
    {
        case 1:
        {
            error_code = convert_coord_DeePMD(frame_info, Nframes_tot, parameters_info, sym_coord);
            break;
        }
        default:
        {
            printf_d("Symmetry coordinate type not supported!\n");
            return 1;
        }
    }



    return error_code;
}

int convert_coord_DeePMD(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, void ** sym_coord)
{
    double s_r(double r_ij, parameters_info_struct * parameters_info);

    int i, j, k, l;
    sym_coord_DeePMD_struct * sym_coord_DeePMD;

    sym_coord_DeePMD = (sym_coord_DeePMD_struct *)calloc(Nframes_tot, sizeof(sym_coord_DeePMD_struct));
    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        sym_coord_DeePMD[i].N_Atoms = frame_info[i].N_Atoms;
        sym_coord_DeePMD[i].SEL_A = parameters_info->SEL_A_max;
        sym_coord_DeePMD[i].type = frame_info[i].type;
        sym_coord_DeePMD[i].coord_converted = (double **)calloc(sym_coord_DeePMD[i].N_Atoms, sizeof(double *));
        for (j = 0; j <= sym_coord_DeePMD[i].N_Atoms - 1; j++)
        {
            sym_coord_DeePMD[i].coord_converted[j] = (double *)calloc(4 * sym_coord_DeePMD[i].SEL_A, sizeof(double));
        }
    }
    #pragma omp parallel for private(j, k, l)
    for (i = 0; i <= Nframes_tot - 1; i++)//loop over each frame
    {
        for (j = 0; j <= sym_coord_DeePMD[i].N_Atoms - 1; j++)//loop over each atom in one frame
        {
            for (k = 0; k <= sym_coord_DeePMD[i].SEL_A - 1; k++)//k and l loop = SEL_A * 4 coordinates for each atom in one frame. k is also the loop of neighbour list of this atom
            {
                double four_coord[4];
                double r_ij;
                double atom_coord[3];
                double nei_coord[3];
                double r_ji_coord[3];
                atom_coord[0] = frame_info[i].coord[j][0]; atom_coord[1] = frame_info[i].coord[j][1]; atom_coord[2] = frame_info[i].coord[j][2];
                nei_coord[0] = frame_info[i].neighbour_list[j].coord_neighbours[k][0]; nei_coord[1] = frame_info[i].neighbour_list[j].coord_neighbours[k][1]; nei_coord[2] = frame_info[i].neighbour_list[j].coord_neighbours[k][2];
                r_ji_coord[0] = nei_coord[0] - atom_coord[0]; r_ji_coord[1] = nei_coord[1] - atom_coord[1]; r_ji_coord[2] = nei_coord[2] - atom_coord[2];
                r_ij = sqrt(pow(atom_coord[0] - nei_coord[0], 2) + pow(atom_coord[1] - nei_coord[1], 2) + pow(atom_coord[2] - nei_coord[2], 2));
                four_coord[0] = s_r(r_ij, parameters_info);
                four_coord[1] = four_coord[0] * r_ji_coord[0] / r_ij; four_coord[2] = four_coord[0] * r_ji_coord[1] / r_ij; four_coord[3] = four_coord[0] * r_ji_coord[2] / r_ij; 
                for (l = 0; l <= 4-1; l++)
                {
                    int idx_sym = k * 4 + l;
                    sym_coord_DeePMD[i].coord_converted[j][idx_sym] = four_coord[l];
                }
            }
        }
    }

    *((sym_coord_DeePMD_struct **)sym_coord) = sym_coord_DeePMD;
    return 0;
}