/*
2019.03.29 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Convert and generate symmetry coordinates.
coord_type:
    1: DeePMD-type coordinate
    2: LASP-type coordinate

Return code:
    0: No errors.
    1: sym_coord_type error.
    31: Open LASP.raw error.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_CONV

#ifdef DEBUG_CONV
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

#define PI 3.141592653589793238462643383279

int convert_coord(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, int sym_coord_type, void ** sym_coord)
{
    int convert_coord_DeePMD(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, void ** sym_coord);
    int convert_coord_LASP(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, void ** sym_coord);

    int i, j, k;
    int error_code;

    switch (sym_coord_type)
    {
        case 1:
        {
            error_code = convert_coord_DeePMD(frame_info, Nframes_tot, parameters_info, sym_coord);
            printf_d("Check d: %lf\n", ((sym_coord_DeePMD_struct **)sym_coord)[0][0].d_to_center_x[0][0]);
            break;
        }
        case 2:
        {
            error_code = convert_coord_LASP(frame_info, Nframes_tot, parameters_info, sym_coord);
            //printf_d("Check d: %lf\n", ((sym_coord_LASP_struct **)sym_coord)[0][0].d_to_center_x[0][0]);
            break;
        }
        default:
        {
            printf_d("Symmetry coordinate type not supported!\n");
            return 1;
        }
    }


    printf_d("From convert_coord: error_code = %d\n", error_code);
    return error_code;
}

int convert_coord_DeePMD(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, void ** sym_coord)
{
    double s_r(double r_ij, parameters_info_struct * parameters_info);
    double fastpow2(double number, int dummy);

    int i, j, k, l;
    sym_coord_DeePMD_struct * sym_coord_DeePMD;

    sym_coord_DeePMD = (sym_coord_DeePMD_struct *)calloc(Nframes_tot, sizeof(sym_coord_DeePMD_struct));
    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        sym_coord_DeePMD[i].N_Atoms = frame_info[i].N_Atoms;
        sym_coord_DeePMD[i].SEL_A = parameters_info->SEL_A_max;
        sym_coord_DeePMD[i].type = frame_info[i].type;
        sym_coord_DeePMD[i].coord_converted = (double **)calloc(sym_coord_DeePMD[i].N_Atoms, sizeof(double *));
        sym_coord_DeePMD[i].d_to_center_x = (double **)calloc(sym_coord_DeePMD[i].N_Atoms, sizeof(double *));
        sym_coord_DeePMD[i].d_to_center_y = (double **)calloc(sym_coord_DeePMD[i].N_Atoms, sizeof(double *));
        sym_coord_DeePMD[i].d_to_center_z = (double **)calloc(sym_coord_DeePMD[i].N_Atoms, sizeof(double *));
        for (j = 0; j <= sym_coord_DeePMD[i].N_Atoms - 1; j++)
        {
            sym_coord_DeePMD[i].coord_converted[j] = (double *)calloc(4 * sym_coord_DeePMD[i].SEL_A, sizeof(double));
            sym_coord_DeePMD[i].d_to_center_x[j] = (double *)calloc(4 * sym_coord_DeePMD[i].SEL_A, sizeof(double));
            sym_coord_DeePMD[i].d_to_center_y[j] = (double *)calloc(4 * sym_coord_DeePMD[i].SEL_A, sizeof(double));
            sym_coord_DeePMD[i].d_to_center_z[j] = (double *)calloc(4 * sym_coord_DeePMD[i].SEL_A, sizeof(double));
        }
    }
    int zero_count = 0;
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
                r_ij = sqrt(fastpow2(atom_coord[0] - nei_coord[0], 2) + fastpow2(atom_coord[1] - nei_coord[1], 2) + fastpow2(atom_coord[2] - nei_coord[2], 2));
                four_coord[0] = s_r(r_ij, parameters_info);
                four_coord[1] = four_coord[0] * r_ji_coord[0] / r_ij; four_coord[2] = four_coord[0] * r_ji_coord[1] / r_ij; four_coord[3] = four_coord[0] * r_ji_coord[2] / r_ij; 
                for (l = 0; l <= 4-1; l++)
                {
                    int idx_sym = k * 4 + l;
                    sym_coord_DeePMD[i].coord_converted[j][idx_sym] = four_coord[l];
                }
                /*Calculate d sym_coord / d center_atom*/
                double rcs = parameters_info->cutoff_1;
                double rc = parameters_info->cutoff_2;
                
                if (r_ij >= rc)
                {
                    zero_count++;
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 0] = 0;
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 1] = 0;
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 2] = 0;
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 3] = 0;
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 0] = 0;
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 1] = 0;
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 2] = 0;
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 3] = 0;
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 0] = 0;
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 1] = 0;
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 2] = 0;
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 3] = 0;
                }
                else if (r_ij >= rcs)
                {
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 0] = r_ji_coord[0] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[0] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) - (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[0] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 2] = 2.0 * r_ji_coord[0] * r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 3] = 2.0 * r_ji_coord[0] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 0] = r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij) + PI * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[1] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) - (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij) + PI * r_ji_coord[1] * r_ji_coord[1] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 3] = 2.0 * r_ji_coord[1] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[1] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 0] = r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij) + PI * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[0] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) + PI * r_ji_coord[1] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 3] = 2.0 * r_ji_coord[2] * r_ji_coord[2] * (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij * r_ij * r_ij) - (0.5 + 0.5 * cos(PI * (r_ij - rcs) / (rc - rcs))) / (r_ij * r_ij) + PI * r_ji_coord[2] * r_ji_coord[2] * sin(PI * (r_ij - rcs) / (rc - rcs)) / 2.0 / (rc - rcs) / (r_ij * r_ij * r_ij);
                }
                else
                {
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 0] = r_ji_coord[0] / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[0] / (r_ij * r_ij * r_ij * r_ij) - 1.0 / (r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 2] = 2.0 * r_ji_coord[0] * r_ji_coord[1] / (r_ij * r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_x[j][4 * k + 3] = 2.0 * r_ji_coord[0] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 0] = r_ji_coord[1] / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[1] / (r_ij * r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[1] / (r_ij * r_ij * r_ij * r_ij) - 1.0 / (r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_y[j][4 * k + 3] = 2.0 * r_ji_coord[1] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 0] = r_ji_coord[2] / (r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 1] = 2.0 * r_ji_coord[0] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 2] = 2.0 * r_ji_coord[1] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij);
                    sym_coord_DeePMD[i].d_to_center_z[j][4 * k + 3] = 2.0 * r_ji_coord[2] * r_ji_coord[2] / (r_ij * r_ij * r_ij * r_ij) - 1.0 / (r_ij * r_ij);
                }
                printf_d("sym_coord_DeePMD[%d].d_to_center_x[%d][0] = %lf\n", i, j, sym_coord_DeePMD[i].d_to_center_x[j][0]);
            }
        }
    }
    printf_d("zero_count d: %d\n", zero_count);

    *((sym_coord_DeePMD_struct **)sym_coord) = sym_coord_DeePMD;
    printf_d("Check d0: %lf\n", (*((sym_coord_DeePMD_struct **)sym_coord))[0].d_to_center_x[0][0]);
    printf_d("Check d1: %lf\n", sym_coord_DeePMD[0].d_to_center_x[0][0]);
    return 0;
}

int convert_coord_LASP(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, void ** sym_coord)
{
    int read_LASP_parameters(parameters_PTSDs_info_struct * parameters_PTSDs_info, parameters_info_struct * parameters_info);
    int find_index_int(int target, int * array, int array_length);

    sym_coord_LASP_struct * sym_coord_LASP;
    int i, j, k, l, x, y, z, t, M;
    int N_PTSD_tot = 0;
    int read_LASP_parameters_flag = 0;
    parameters_PTSDs_info_struct * parameters_PTSDs_info = (parameters_PTSDs_info_struct *)calloc(1, sizeof(parameters_PTSDs_info_struct));//Remeber to free parameters_PTSDs_info before the return of this function.

    read_LASP_parameters_flag = read_LASP_parameters(parameters_PTSDs_info, parameters_info);
    if (read_LASP_parameters_flag != 0)
    {
        return read_LASP_parameters_flag;
    }

    printf_d("Check from convert_coord: the read-in parameters.\n");
    for (i = 0; i <= parameters_info->N_types_all_frame - 1; i++)
    {
        int N_PTSD_tot_each_type = 0;
        for (j = 0; j <= parameters_PTSDs_info->N_PTSD_types - 1; j++)
        {
            printf_d("center type: %d, PTSD type: %d (%d body PTSD). There are %d parameters in this type of PTSD.\n", i, j + 1, parameters_PTSDs_info->PTSD_N_body_type[j], parameters_PTSDs_info->PTSD_N_params[j]);
            printf_d("Read in %d sets of parameters for this type of PTSD.\n", parameters_PTSDs_info->N_cutoff_radius[i][j]);
            for (k = 0; k <= parameters_PTSDs_info->N_cutoff_radius[i][j] - 1; k++)
            {
                printf_d("cutoff: %.2lf ", parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j][k].cutoff_radius);
                for (l = 0; l <= parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j][k].PTSD_N_body_type - 1 - 1; l++)
                {
                    printf_d("nei_type: %3d, ", parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j][k].neigh_type_array[l]);
                }
                for (l = 0; l <= parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j][k].N_params + 1; l++)
                {
                    printf_d("param%-2d: %+.2lf ", l + 1, parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j][k].params_array[l]);
                }
                printf_d("\n");
                N_PTSD_tot_each_type++;
            }
        }
        N_PTSD_tot = (N_PTSD_tot_each_type > N_PTSD_tot ? N_PTSD_tot_each_type : N_PTSD_tot);
    }
    printf_d("Total number of PTSDs: %d\n", N_PTSD_tot);
    parameters_info->N_sym_coord = N_PTSD_tot;
    
    sym_coord_LASP = (sym_coord_LASP_struct *)calloc(parameters_info->Nframes_tot, sizeof(sym_coord_LASP_struct));
    sym_coord_LASP->N_PTSDs = N_PTSD_tot;
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        sym_coord_LASP[i].N_Atoms = parameters_info->N_Atoms_max;
        sym_coord_LASP[i].SEL_A = N_PTSD_tot;
        sym_coord_LASP[i].N_PTSDs = N_PTSD_tot;
        sym_coord_LASP[i].coord_converted = (double **)calloc(parameters_info->N_Atoms_max, sizeof(double *));
        for (j = 0; j <= parameters_info->N_Atoms_max - 1; j++)
        {
            sym_coord_LASP[i].coord_converted[j] = (double *)calloc(N_PTSD_tot, sizeof(double));
            //printf_d("i, j: %d %d\n", i, j);
        }
    }
    //#pragma omp parallel for private(j, k, l)
    for (i = 0; i <=parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= parameters_info->N_Atoms_max - 1; j++)
        {
            int type_cur_atom_cur_frame = (frame_info[i].type[j] == -1 ? parameters_info->type_index_all_frame[0] : frame_info[i].type[j]);
            int ii = find_index_int(type_cur_atom_cur_frame, parameters_info->type_index_all_frame, parameters_info->N_types_all_frame);
            int N_PTSD_count_idx = 0;//N_PTSD_count_idx should be equal to SEL_A - 1
            for (k = 0; k <= parameters_PTSDs_info->N_PTSD_types - 1; k++)
            {
                /*PTSD parameters information stored in idx[ii][k][0..NN-1], NN=N_cutoff_radius[ii][k]*/
                /*Pay attention to the sequence of parameters!!*/
                for (l = 0; l <= parameters_PTSDs_info->N_cutoff_radius[ii][k] - 1; l++)
                {
                    int N_params_ii_k_l = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].N_params;
                    /*Number of neighbour atoms needed by each type of PTSD:
                    S1: 1
                    S2: 1
                    S3: 2
                    S4: 2
                    S5: 2
                    S6: 3
                    */
                    /*In addition to r_c, the parameters stored in params_array follows this order:
                    S1: (int)n
                    S2: (int)L, (int)n
                    S3: (int)n, (int)m, (int)zeta, (double)lambda
                    S4: (int)n, (int)m, (int)p, (int)zeta, (double)lambda
                    S5: (int)L, (int)n, (int)m, (int)p
                    S6: (int)n, (int)m, (int)p, (int)zeta, (double)lambda
                    */
                    //printf_d("i, j, k, l, ii, N_params: %d %d %d %d %d %d\n", i, j, k, l, ii, N_params_ii_k_l);
                    switch (k)
                    {
                        case 0:
                        {
                            int nb1;
                            break;
                        }
                        case 1:
                        {
                            break;
                        }
                        case 2:
                        {
                            break;
                        }
                        case 3:
                        {
                            break;
                        }
                        case 4:
                        {
                            break;
                        }
                        case 5:
                        {
                            break;
                        }
                        default:
                        {
                            break;
                        }
                    }
                }
            }
        }
    }

    *(sym_coord_LASP_struct **)sym_coord = sym_coord_LASP;
    //return 0;
    return (printf("Not completed, return 999.\n"), 999);//incomplete

}

int read_LASP_parameters(parameters_PTSDs_info_struct * parameters_PTSDs_info, parameters_info_struct * parameters_info)
{
    int calc_N_neigh_inter(int K, int N);
    int find_index_int(int target, int * array, int array_length);
    /*double **** calloc_params_LASP(int dim1, int dim2, int ** dim3_, int ** dim4_);*/

    const int MAX_NUM_ELEMENTS = 172;//the number of elements will not exceed 172 at 2019
    const int MAX_N_CUTOFF_RADIUS = (int)1000;// No more than 1000 data.
    const int N_TYPES_ALL_FRAME = parameters_info->N_types_all_frame;
    char * tmp_line = (char *) calloc(100000, sizeof(char)); 
    char * tmp_token = NULL;
    int center_type, PTSD_type;//PTSD_type range from 1 to 6. Remeber to -1 when use PTSD_type.
    FILE * fp;
    int N_body_type[6] = {2, 2, 3, 3, 3, 4};
    int N_params[6] = {1, 2, 4, 5, 4, 5};
    int i, j, k, l;
    

    parameters_PTSDs_info->N_PTSD_types = 6;
    parameters_PTSDs_info->N_types_all_frame = N_TYPES_ALL_FRAME;
    parameters_PTSDs_info->PTSD_N_body_type = N_body_type;
    parameters_PTSDs_info->PTSD_N_params = N_params;
    parameters_PTSDs_info->cutoff_max = parameters_info->cutoff_max;

    parameters_PTSDs_info->N_cutoff_radius = (int **)calloc(N_TYPES_ALL_FRAME, sizeof(int *));
    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)
    {
        parameters_PTSDs_info->N_cutoff_radius[i] = (int *)calloc(parameters_PTSDs_info->N_PTSD_types, sizeof(int)); 
    }
    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)
    {
        for (j = 0; j <= parameters_PTSDs_info->N_PTSD_types - 1; j++)
        {
            parameters_PTSDs_info->N_cutoff_radius[i][j] = MAX_N_CUTOFF_RADIUS;
        }
    }

    parameters_PTSDs_info->N_neigh_inter = (int **)calloc(N_TYPES_ALL_FRAME, sizeof(int *));
    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)
    {
        parameters_PTSDs_info->N_neigh_inter[i] = (int *)calloc(parameters_PTSDs_info->N_PTSD_types, sizeof(int)); // No more than (cutoff_max / 0.1) data.
    }
    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)
    {
        for (j = 0; j <= parameters_PTSDs_info->N_PTSD_types - 1; j++)
        {
            parameters_PTSDs_info->N_neigh_inter[i][j] = calc_N_neigh_inter(parameters_PTSDs_info->PTSD_N_body_type[j], parameters_PTSDs_info->N_types_all_frame);
            printf_d("%d body %d type: N_neigh_inter %d %d = %d\n", parameters_PTSDs_info->PTSD_N_body_type[j], N_TYPES_ALL_FRAME, i, j, parameters_PTSDs_info->N_neigh_inter[i][j]);
        }
    }
    
    /*parameters_PTSDs_info->cutoff_radius = (double ***)calloc(N_TYPES_ALL_FRAME, sizeof(double **));
    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)
    {
        parameters_PTSDs_info->cutoff_radius[i] = (double **)calloc(parameters_PTSDs_info->N_PTSD_types, sizeof(double *));
    }
    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)
    {
        for (j = 0; j <= parameters_PTSDs_info->N_PTSD_types - 1; j++)
        {
            parameters_PTSDs_info->cutoff_radius[i][j] = (double *)calloc(parameters_PTSDs_info->N_cutoff_radius[i][j], sizeof(double));
        }
    }*/

    /*parameters_PTSDs_info->n = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->m = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->p = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->L = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->r_c = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->zeta = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->lambda = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->Gmin = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);
    parameters_PTSDs_info->Gmax = calloc_params_LASP(N_TYPES_ALL_FRAME, parameters_PTSDs_info->N_PTSD_types, parameters_PTSDs_info->N_cutoff_radius, parameters_PTSDs_info->N_neigh_inter);*/

    parameters_PTSDs_info->parameters_PTSDs_info_one_line = (parameters_PTSDs_info_one_line_struct ***)calloc(N_TYPES_ALL_FRAME, sizeof(parameters_PTSDs_info_one_line_struct **));
    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)
    {
        parameters_PTSDs_info->parameters_PTSDs_info_one_line[i] = (parameters_PTSDs_info_one_line_struct **)calloc(parameters_PTSDs_info->N_PTSD_types, sizeof(parameters_PTSDs_info_one_line_struct *));
        for (j = 0; j <= parameters_PTSDs_info->N_PTSD_types - 1; j++)
        {
            parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j] = (parameters_PTSDs_info_one_line_struct *)calloc(MAX_N_CUTOFF_RADIUS, sizeof(parameters_PTSDs_info_one_line_struct));
        }
    }

    fp = fopen("./LASP.raw", "r");
    if (fp == NULL)
    {
        printf("Open LASP.raw failed!\n");
        return 31;
    }

    for (i = 0; i <= N_TYPES_ALL_FRAME - 1; i++)//loop over all N_types
    {
        for (j = 0; j <= parameters_PTSDs_info->N_PTSD_types - 1; j++)//loop over all types of PTSDs
        {
            /*Read information between %block_start and %block_end*/
            if (fscanf(fp, "%s%d%d\n", tmp_line, &center_type, &PTSD_type) < 3)
            {
                printf("LASP.raw file format is not correct. Unexpected space lines or incomplete data. \nReading stops at center atom type %d PTSD type %d\n", center_type, PTSD_type);
                return 32;
            }
            /*No need. Add fscanf(fp, " ") to skip all the spaces at the beginning of a line
            if (tmp_line[0] != '%')
            {
                printf("No spaces before %%block_start and %%block_end are allowed!\n");
                return 33;
            }*/
            printf_d("#%s %d %d\n", tmp_line, center_type, PTSD_type);
            int ii = find_index_int(center_type, parameters_info->type_index_all_frame, parameters_info->N_types_all_frame);
            int jj = PTSD_type - 1;//index of PTSD_type. Remember -1. In the input file, PTSD type ranges from 1 to 6, not 0 to 5.
            printf_d("center_type index = %d\n", ii);
            fgets(tmp_line, 100000, fp);//Read the comment line
            printf_d("#%s", tmp_line);
            fscanf(fp, " "); fgets(tmp_line, 100000, fp);//The first line of data
            printf_d("#%s", tmp_line);
            int cutoff_radius_pointer = 0;//Also counts for the actual value of N_cutoff_radius[i][j], and is also the number of PTSDs.


            while (tmp_line[7] != 'e')//loop over a block and read in data
            {
                int N_body_this_type_PTSD = parameters_PTSDs_info->PTSD_N_body_type[jj];
                int N_neighb_atom = N_body_this_type_PTSD - 1;
                int N_params_this_type = parameters_PTSDs_info->PTSD_N_params[jj];
                double cutoff_this_line;
                int * neighb_atom_array = (int *)calloc(N_neighb_atom, sizeof(int));
                double * params_array = (double *)calloc(N_params_this_type + 2, sizeof(double));//The last two elements are Gmin and Gmax
                parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].PTSD_type = PTSD_type;
                parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].PTSD_N_body_type = N_body_this_type_PTSD;
                parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].N_params = N_params_this_type;
                parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].neigh_type_array = (int *)calloc(N_neighb_atom, sizeof(int));
                parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].params_array = (double *)calloc(N_params_this_type + 2, sizeof(double));
                /*The data in one line should be arranged as:*/
                /*{at least 0 spaces}[cutoff]{spaces}{N_neighb_atom integers}{spaces}{N_params_this_type parameters}{spaces}{Gmin and Gmax}{at least 1 char}[\n]*/

                tmp_token = strtok(tmp_line, " ");
                if (sscanf(tmp_token, "%lf", &cutoff_this_line) != 1 )
                {
                    printf("Format within one block is incorrect. Make sure there are no comment or empty lines mixed with data lines!\nReading stops at center atom type %d PTSD type %d\n", center_type, PTSD_type);
                    return 33;
                }
                //printf_d("$%7.2lf", cutoff_this_line);
                parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].cutoff_radius = cutoff_this_line;
                for (k = 0; k <= N_neighb_atom - 1; k++)//read in the neighbour atom type
                {
                    tmp_token = strtok(NULL, " ");
                    sscanf(tmp_token, "%d", &(neighb_atom_array[k]));
                    //printf_d("   %7d", neighb_atom_array[k]);
                    parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].neigh_type_array[k] = neighb_atom_array[k];
                }
                for (k = 0; k <= N_params_this_type - 1; k++)//read in all the parameters of this type of PTSD
                {
                    tmp_token = strtok(NULL, " ");
                    sscanf(tmp_token, "%lf", &(params_array[k]));
                    //printf_d("     %6d  ", (int)params_array[k]);
                    parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].params_array[k] = params_array[k];
                }
                for (k = N_params_this_type; k <= N_params_this_type + 1; k++)//read in Gmin and Gmax
                {
                    tmp_token = strtok(NULL, " ");
                    sscanf(tmp_token, "%lf", &(params_array[k]));
                    //printf_d("      %21.15E", params_array[k]);
                    parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][jj][cutoff_radius_pointer].params_array[k] = params_array[k];
                }
                //printf_d("\n");
                
                fscanf(fp, " "); fgets(tmp_line, 100000, fp);
                printf_d("#%s", tmp_line);
                /*dim0: center type; dim1: PTSD type; dim2: cutoff radius*/
                cutoff_radius_pointer ++;
            }
            parameters_PTSDs_info->N_cutoff_radius[ii][jj] = cutoff_radius_pointer;
        }
    }

    fclose(fp);
    return 0;

}

