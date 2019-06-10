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
#include <complex>
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
            printf("Symmetry coordinate type not supported!\n");
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

    parameters_info->N_sym_coord = parameters_info->SEL_A_max * 4;
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
    double fastpow2(double number, int dummy);
    double fastpown(double number, int power);
    double R_sup_n(double r_ij, double n, double r_c);
    int compare_Nei_type(int N_neighb_atom, int * current_type, int * params_type);
    std::complex<double> Y_LM(double * coord_ij, int L, int m);
    std::complex<double> d_Y_LM_d_theta(double * coord_ij, int L, int m);
    std::complex<double> d_Y_LM_d_phi(double * coord_ij, int L, int m);
    double cos_bond_angle(double * coord_i, double * coord_j, double * coord_k);
    double cos_dihedral_angle(double * coord_i, double * coord_j, double * coord_k, double * coord_l);
    double d_R_sup_n_d_r(double r_ij, double n, double r_c);
    int d_cos_bond_angle_d_coord(double * coord_i, double * coord_j, double * coord_k, double * result);
    int d_cos_dihedral_angle_d_coord(double * coord_i, double * coord_j, double * coord_k, double * coord_l, double * result);

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
            printf_d("center type: %d, PTSD type: %d (%d body PTSD). There are %d parameters in this type of PTSD.\n", parameters_info->type_index_all_frame[i], j + 1, parameters_PTSDs_info->PTSD_N_body_type[j], parameters_PTSDs_info->PTSD_N_params[j]);
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
        sym_coord_LASP[i].N_Atoms = frame_info->N_Atoms;
        sym_coord_LASP[i].SEL_A = N_PTSD_tot;
        sym_coord_LASP[i].N_PTSDs = N_PTSD_tot;
        sym_coord_LASP[i].coord_converted = (double **)calloc(parameters_info->N_Atoms_max, sizeof(double *));
        sym_coord_LASP[i].idx_nei = (int ***)calloc(parameters_info->N_Atoms_max, sizeof(int **));
        sym_coord_LASP[i].d_x = (double ***)calloc(parameters_info->N_Atoms_max, sizeof(double **));
        sym_coord_LASP[i].d_y = (double ***)calloc(parameters_info->N_Atoms_max, sizeof(double **));
        sym_coord_LASP[i].d_z = (double ***)calloc(parameters_info->N_Atoms_max, sizeof(double **));
        for (j = 0; j <= parameters_info->N_Atoms_max - 1; j++)
        {
            sym_coord_LASP[i].coord_converted[j] = (double *)calloc(N_PTSD_tot, sizeof(double));
            sym_coord_LASP[i].idx_nei[j] = (int **)calloc(parameters_info->N_sym_coord, sizeof(int *));
            sym_coord_LASP[i].d_x[j] = (double **)calloc(parameters_info->N_sym_coord, sizeof(double *));
            sym_coord_LASP[i].d_y[j] = (double **)calloc(parameters_info->N_sym_coord, sizeof(double *));
            sym_coord_LASP[i].d_z[j] = (double **)calloc(parameters_info->N_sym_coord, sizeof(double *));
            for (k = 0; k <= parameters_info->N_sym_coord - 1; k++)
            {
                sym_coord_LASP[i].d_x[j][k] = (double *)calloc(parameters_info->N_Atoms_max, sizeof(double));
                sym_coord_LASP[i].d_y[j][k] = (double *)calloc(parameters_info->N_Atoms_max, sizeof(double));
                sym_coord_LASP[i].d_z[j][k] = (double *)calloc(parameters_info->N_Atoms_max, sizeof(double));
            }
            //printf_d("i, j: %d %d\n", i, j);
        }
    }

    #pragma omp parallel for private(j, k, l)
    for (i = 0; i <=parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= parameters_info->N_Atoms_max - 1; j++)
        {
            int type_cur_atom_cur_frame = (frame_info[i].type[j] == -1 ? parameters_info->type_index_all_frame[0] : frame_info[i].type[j]);
            int ii = find_index_int(type_cur_atom_cur_frame, parameters_info->type_index_all_frame, parameters_info->N_types_all_frame);
            int N_PTSD_count_idx = 0;
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
                    Note: the S3/S4 are S4/S3 on the paper Chem. Sci. 2018. 9. 8644-8655 
                    */
                    //printf_d("i, j, k, l, ii, N_params: %d %d %d %d %d %d\n", i, j, k, l, ii, N_params_ii_k_l);
                    switch (k)
                    {
                        case 0:
                        {
                            int nb1;
                            double * coord_i = frame_info[i].coord[j];
                            double result = 0;
                            double r_c = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].cutoff_radius;
                            double n = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[0];
                            int * params_type = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].neigh_type_array;
                            int N_body = parameters_PTSDs_info->PTSD_N_body_type[k];
                            int N_nei = N_body - 1;
                            int idx_i = j;
                            for (nb1 = 0; nb1 <= parameters_info->SEL_A_max - 1; nb1++)
                            {
                                int idx_j = frame_info[i].neighbour_list[j].index_neighbours[nb1];//idx_i = j
                                int current_type[1] = {frame_info[i].neighbour_list[j].type[nb1]};
                                //if (idx_j >= 21) printf_d("idx_j:%d\n", idx_j);
                                if (compare_Nei_type(N_nei, current_type, params_type) == 0)
                                {
                                    continue;
                                }
                                double * coord_j = frame_info[i].neighbour_list[j].coord_neighbours[nb1];
                                double r_ij = sqrt(fastpow2(coord_i[0] - coord_j[0], 2) + fastpow2(coord_i[1] - coord_j[1], 2) + fastpow2(coord_i[2] - coord_j[2], 2));
                                if (r_ij > r_c)
                                {
                                    break;
                                }
                                //printf_d("r_ij = %.2lf, ", r_ij);
                                result += R_sup_n(r_ij, n, r_c);
                                /*d_S1/d_xk = d_S1/d_rij * d_rij/d_xk, k = i or j*/
                                sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_i] += d_R_sup_n_d_r(r_ij, n, r_c) * (coord_i[0] - coord_j[0]) / r_ij;
                                sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_j] += d_R_sup_n_d_r(r_ij, n, r_c) * (coord_i[0] - coord_j[0]) / r_ij * (-1.0);
                                sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_i] += d_R_sup_n_d_r(r_ij, n, r_c) * (coord_i[1] - coord_j[1]) / r_ij;
                                sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_j] += d_R_sup_n_d_r(r_ij, n, r_c) * (coord_i[1] - coord_j[1]) / r_ij * (-1.0);
                                sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_i] += d_R_sup_n_d_r(r_ij, n, r_c) * (coord_i[2] - coord_j[2]) / r_ij;
                                sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_j] += d_R_sup_n_d_r(r_ij, n, r_c) * (coord_i[2] - coord_j[2]) / r_ij * (-1.0);
                            }
                            sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx] = result;
                            //printf_d("r_c: %.2lf, S1: %lf\n", r_c, result);
                            N_PTSD_count_idx++;                            
                            break;
                        }
                        case 1:
                        {
                            int nb1;
                            int M;
                            double * coord_i = frame_info[i].coord[j];
                            double result = 0;
                            double r_c = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].cutoff_radius;
                            int L = (int)(parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[0]);
                            double n = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[1];
                            int * params_type = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].neigh_type_array;
                            int N_body = parameters_PTSDs_info->PTSD_N_body_type[k];
                            int N_nei = N_body - 1;
                            double derivative_prefactor = 0;
                            int idx_i = j;
                            for (M = -L; M <= L; M++)
                            {
                                std::complex<double> result_inner = (0, 0);
                                int d_idx;
                                std::complex<double> * derivative_tmp = (std::complex<double> *)calloc(3 * parameters_info->N_Atoms_max, sizeof(std::complex<double>));//dx,dy,dz
                                for (nb1 = 0; nb1 <= parameters_info->SEL_A_max - 1; nb1++)
                                {
                                    int idx_j = frame_info[i].neighbour_list[j].index_neighbours[nb1];
                                    int current_type[1]= {frame_info[i].neighbour_list[j].type[nb1]};
                                    std::complex<double> R_Y;
                                    std::complex<double> YLM;
                                    double R;
                                    if (compare_Nei_type(N_nei, current_type, params_type) == 0)
                                    {
                                        continue;
                                    }
                                    double * coord_j = frame_info[i].neighbour_list[j].coord_neighbours[nb1];
                                    double r_ij = sqrt(fastpow2(coord_i[0] - coord_j[0], 2) + fastpow2(coord_i[1] - coord_j[1], 2) + fastpow2(coord_i[2] - coord_j[2], 2));
                                    double coord_ij[3] = {coord_i[0] - coord_j[0], coord_i[1] - coord_j[1], coord_i[2] - coord_j[2]};
                                    if (r_ij > r_c)
                                    {
                                        break;
                                    }
                                    YLM = Y_LM(coord_ij, L, M);
                                    R = R_sup_n(r_ij, n, r_c);
                                    R_Y = YLM * R;
                                    result_inner += R_Y;
                                    /*Calculate \partial RYLM / \partial x,y,z*/
                                    /*+= dR/dr * dr/dxi * Y_LM + R * (dY_LM/dtheta * dtheta/dxi + dY_LM/dphi * dphi/dxi)*/
                                    /*First, I need to calculate all damn d_angle/d_x,y,z*/
                                    double d_r_d_x_i = (coord_i[0] - coord_j[0]) / r_ij;
                                    double d_r_d_x_j = (coord_j[0] - coord_i[0]) / r_ij;
                                    double d_r_d_y_i = (coord_i[1] - coord_j[1]) / r_ij;
                                    double d_r_d_y_j = (coord_j[1] - coord_i[1]) / r_ij;
                                    double d_r_d_z_i = (coord_i[2] - coord_j[2]) / r_ij;
                                    double d_r_d_z_j = (coord_j[2] - coord_i[2]) / r_ij;
                                    double d_theta_d_x_i = (coord_ij[0]) * (coord_ij[2]) / (fastpown(r_ij, 3) * sqrt(1.0 - fastpow2(coord_ij[2], 2) / (fastpow2(r_ij, 2))));
                                    double d_theta_d_x_j = 0.0 - d_theta_d_x_i;
                                    double d_theta_d_y_i = (coord_ij[1]) * (coord_ij[2]) / (fastpown(r_ij, 3) * sqrt(1.0 - fastpow2(coord_ij[2], 2) / (fastpow2(r_ij, 2))));
                                    double d_theta_d_y_j = 0.0 - d_theta_d_y_i;
                                    double d_theta_d_z_i = 0.0 - (1.0 / r_ij - fastpow2(coord_ij[2], 2) / fastpown(r_ij, 3)) / (1.0 - fastpow2(coord_ij[2], 2) / fastpow2(r_ij, 2));
                                    double d_theta_d_z_j = 0.0 - d_theta_d_z_i;
                                    double d_phi_d_x_i = 0.0 - (coord_ij[1]) / ((fastpow2(coord_ij[0], 2)) * (1 + fastpow2(coord_ij[1] / coord_ij[0], 2)));
                                    double d_phi_d_x_j = 0.0 - d_phi_d_x_i;
                                    double d_phi_d_y_i = 1.0 / (coord_ij[0] * (1 + fastpow2(coord_ij[1] / coord_ij[0], 2)));
                                    double d_phi_d_y_j = 0.0 - d_phi_d_y_i;
                                    double d_phi_d_z_i = 0.0;
                                    double d_phi_d_z_j = 0.0;
                                    std::complex<double> D_YLM_D_THETA = d_Y_LM_d_theta(coord_ij, L, M);
                                    std::complex<double> D_YLM_D_PHI = d_Y_LM_d_phi(coord_ij, L, M);
                                    double D_R_D_r = d_R_sup_n_d_r(r_ij, n, r_c);
                                    //d to x
                                    derivative_tmp[idx_i] += D_R_D_r * d_r_d_x_i * YLM + R * (D_YLM_D_THETA * d_theta_d_x_i + D_YLM_D_PHI * d_phi_d_x_i);
                                    derivative_tmp[idx_j] += D_R_D_r * d_r_d_x_j * YLM + R * (D_YLM_D_THETA * d_theta_d_x_j + D_YLM_D_PHI * d_phi_d_x_j);
                                    //d to y
                                    derivative_tmp[parameters_info->N_Atoms_max - 1 + idx_i] += D_R_D_r * d_r_d_y_i * YLM + R * (D_YLM_D_THETA * d_theta_d_y_i + D_YLM_D_PHI * d_phi_d_y_i);
                                    derivative_tmp[parameters_info->N_Atoms_max - 1 + idx_j] += D_R_D_r * d_r_d_y_j * YLM + R * (D_YLM_D_THETA * d_theta_d_y_j + D_YLM_D_PHI * d_phi_d_y_j);;
                                    //d to z
                                    derivative_tmp[2 * parameters_info->N_Atoms_max - 1 + idx_i] += D_R_D_r * d_r_d_z_i * YLM + R * (D_YLM_D_THETA * d_theta_d_z_i + D_YLM_D_PHI * d_phi_d_z_i);
                                    derivative_tmp[2 * parameters_info->N_Atoms_max - 1 + idx_j] += D_R_D_r * d_r_d_z_j * YLM + R * (D_YLM_D_THETA * d_theta_d_z_j + D_YLM_D_PHI * d_phi_d_z_j);
                                }
                                result += std::norm(result_inner);
                                for (d_idx = 0; d_idx <= 3 * parameters_info->N_Atoms_max - 1; d_idx++)
                                {
                                    derivative_tmp[d_idx] *= (2.0 * result_inner);
                                }
                                for (d_idx = 0; d_idx <= parameters_info->N_Atoms_max - 1; d_idx++)
                                {
                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][d_idx] += derivative_tmp[d_idx].real() + derivative_tmp[d_idx].imag();
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][d_idx] += derivative_tmp[parameters_info->N_Atoms_max - 1 + d_idx].real() + derivative_tmp[parameters_info->N_Atoms_max - 1 + d_idx].imag();
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][d_idx] += derivative_tmp[2 * parameters_info->N_Atoms_max  - 1 + d_idx].real() + derivative_tmp[2 * parameters_info->N_Atoms_max  - 1 + d_idx].imag();
                                }
                                free(derivative_tmp);
                            }
                            sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx] = sqrt(result);
                            if (result <= 1E-8 )
                            {
                                derivative_prefactor = 0;
                            }
                            else
                            {
                                derivative_prefactor = 0.5 / sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx];
                            }
                            
                            int d_idx = 0;
                            /*Derivative = prefactor * \sum^L_(M=-L) 2 * result_inner * \sum \partial RYLM/ \partial x,y,z*/
                            for (d_idx = 0; d_idx <= parameters_info->N_Atoms_max - 1; d_idx++)
                            {
                                sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][d_idx] *= derivative_prefactor;
                                sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][d_idx] *= derivative_prefactor;
                                sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][d_idx] *= derivative_prefactor;
                            }
                            N_PTSD_count_idx++;
                            break;
                        }
                        case 2:
                        {
                            int nb1, nb2;
                            double n, m, zeta, lambda;
                            double * coord_i = frame_info[i].coord[j];
                            double result = 0;
                            double r_c = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].cutoff_radius;
                            int * params_type = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].neigh_type_array;
                            int N_body = parameters_PTSDs_info->PTSD_N_body_type[k];
                            int N_nei = N_body - 1;
                            n = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[0];
                            m = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[1];
                            zeta = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[2];
                            lambda = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[3];
                            int idx_i = j;
                            double prefac_2 = fastpown(2, (int)(1 - zeta));
                            for (nb1 = 0; nb1 <= parameters_info->SEL_A_max - 1; nb1++)
                            {
                                double * coord_j = frame_info[i].neighbour_list[j].coord_neighbours[nb1];
                                double r_ij = sqrt(fastpow2(coord_i[0] - coord_j[0], 2) + fastpow2(coord_i[1] - coord_j[1], 2) + fastpow2(coord_i[2] - coord_j[2], 2));
                                if (r_ij > r_c)
                                {
                                    break;
                                }
                                for (nb2 = nb1 + 1; nb2 <= parameters_info->SEL_A_max - 1; nb2++)
                                {
                                    int current_type[2] = {frame_info[i].neighbour_list[j].type[nb1], frame_info[i].neighbour_list[j].type[nb2]};
                                    double * coord_k = frame_info[i].neighbour_list[j].coord_neighbours[nb2];
                                    int idx_j = frame_info[i].neighbour_list[j].index_neighbours[nb1];
                                    int idx_k = frame_info[i].neighbour_list[j].index_neighbours[nb2];
                                    if (compare_Nei_type(N_nei, current_type, params_type) == 0)
                                    {
                                        continue;
                                    }
                                    double r_ik = sqrt(fastpow2(coord_i[0] - coord_k[0], 2) + fastpow2(coord_i[1] - coord_k[1], 2) + fastpow2(coord_i[2] - coord_k[2], 2));
                                    if (r_ik > r_c)
                                    {
                                        break;
                                    }
                                    double cos_theta = cos_bond_angle(coord_i, coord_j, coord_k);
                                    double d_r_ij_d_x_i = (coord_i[0] - coord_j[0]) / r_ij;
                                    double d_r_ij_d_x_j = (coord_j[0] - coord_i[0]) / r_ij;
                                    double d_r_ik_d_x_i = (coord_i[0] - coord_k[0]) / r_ik;
                                    double d_r_ik_d_x_k = (coord_k[0] - coord_i[0]) / r_ik;

                                    double d_r_ij_d_y_i = (coord_i[1] - coord_j[1]) / r_ij;
                                    double d_r_ij_d_y_j = (coord_j[1] - coord_i[1]) / r_ij;
                                    double d_r_ik_d_y_i = (coord_i[1] - coord_k[1]) / r_ik;
                                    double d_r_ik_d_y_k = (coord_k[1] - coord_i[1]) / r_ik;
                                    
                                    double d_r_ij_d_z_i = (coord_i[2] - coord_j[2]) / r_ij;
                                    double d_r_ij_d_z_j = (coord_j[2] - coord_i[2]) / r_ij;
                                    double d_r_ik_d_z_i = (coord_i[2] - coord_k[2]) / r_ik;
                                    double d_r_ik_d_z_k = (coord_k[2] - coord_i[2]) / r_ik;
                                    
                                    double d_zeta_prefac = zeta * fastpown(1 + lambda * cos_theta, (int)(zeta - 1));
                                    double zeta_prefac = fastpown((1 + lambda * cos_theta), (int)zeta);
                                    double d_cos_theta_d_coord[9];
                                    double R_n_r_ij = R_sup_n(r_ij, n, r_c);
                                    double R_m_r_ik = R_sup_n(r_ik, m, r_c);
                                    double d_R_d_r_ij = d_R_sup_n_d_r(r_ij, n, r_c);
                                    double d_R_d_r_ik = d_R_sup_n_d_r(r_ik, m, r_c);

                                    d_cos_bond_angle_d_coord(coord_i, coord_j, coord_k, d_cos_theta_d_coord);

                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_i] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[0] * R_n_r_ij * R_m_r_ik + zeta_prefac * d_R_d_r_ij * d_r_ij_d_x_i * R_m_r_ik + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_x_i);
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_i] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[1] * R_n_r_ij * R_m_r_ik + zeta_prefac * d_R_d_r_ij * d_r_ij_d_y_i * R_m_r_ik + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_y_i);
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_i] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[2] * R_n_r_ij * R_m_r_ik + zeta_prefac * d_R_d_r_ij * d_r_ij_d_z_i * R_m_r_ik + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_z_i);
                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_j] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[3] * R_n_r_ij * R_m_r_ik + zeta_prefac * d_R_d_r_ij * d_r_ij_d_x_j * R_m_r_ik);
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_j] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[4] * R_n_r_ij * R_m_r_ik + zeta_prefac * d_R_d_r_ij * d_r_ij_d_y_j * R_m_r_ik);
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_j] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[5] * R_n_r_ij * R_m_r_ik + zeta_prefac * d_R_d_r_ij * d_r_ij_d_z_j * R_m_r_ik);
                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_k] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[6] * R_n_r_ij * R_m_r_ik + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_x_k);
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_k] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[7] * R_n_r_ij * R_m_r_ik + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_y_k);
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_k] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[8] * R_n_r_ij * R_m_r_ik + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_z_k);

                                    result += (zeta_prefac * R_n_r_ij * R_m_r_ik);
                                }
                            }

                            sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx] = result * prefac_2;
                            N_PTSD_count_idx++;
                            break;
                        }
                        case 3:
                        {
                            int nb1, nb2;
                            double n, m, p, zeta, lambda;
                            double * coord_i = frame_info[i].coord[j];
                            double result = 0;
                            double r_c = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].cutoff_radius;
                            int * params_type = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].neigh_type_array;
                            int N_body = parameters_PTSDs_info->PTSD_N_body_type[k];
                            int N_nei = N_body - 1;
                            n = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[0];
                            m = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[1];
                            p = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[2];
                            zeta = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[3];
                            lambda = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[4];
                            int idx_i = j;
                            double prefac_2 = fastpown(2, (int)(1 - zeta));
                            for (nb1 = 0; nb1 <= parameters_info->SEL_A_max - 1; nb1++)
                            {
                                double * coord_j = frame_info[i].neighbour_list[j].coord_neighbours[nb1];
                                double r_ij = sqrt(fastpow2(coord_i[0] - coord_j[0], 2) + fastpow2(coord_i[1] - coord_j[1], 2) + fastpow2(coord_i[2] - coord_j[2], 2));
                                if (r_ij > r_c)
                                {
                                    break;
                                }
                                for (nb2 = nb1 + 1; nb2 <= parameters_info->SEL_A_max - 1; nb2++)
                                {
                                    int current_type[2] = {frame_info[i].neighbour_list[j].type[nb1], frame_info[i].neighbour_list[j].type[nb2]};
                                    double * coord_k = frame_info[i].neighbour_list[j].coord_neighbours[nb2];
                                    int idx_j = frame_info[i].neighbour_list[j].index_neighbours[nb1];
                                    int idx_k = frame_info[i].neighbour_list[j].index_neighbours[nb2];
                                    if (compare_Nei_type(N_nei, current_type, params_type) == 0)
                                    {
                                        continue;
                                    }
                                    double r_ik = sqrt(fastpow2(coord_i[0] - coord_k[0], 2) + fastpow2(coord_i[1] - coord_k[1], 2) + fastpow2(coord_i[2] - coord_k[2], 2));
                                    if (r_ik > r_c)
                                    {
                                        break;
                                    }
                                    double r_jk = sqrt(fastpow2(coord_j[0] - coord_k[0], 2) + fastpow2(coord_j[1] - coord_k[1], 2) + fastpow2(coord_j[2] - coord_k[2], 2));
                                    double cos_theta = cos_bond_angle(coord_i, coord_j, coord_k);
                                    double d_r_ij_d_x_i = (coord_i[0] - coord_j[0]) / r_ij;
                                    double d_r_ij_d_x_j = (coord_j[0] - coord_i[0]) / r_ij;
                                    double d_r_ik_d_x_i = (coord_i[0] - coord_k[0]) / r_ik;
                                    double d_r_ik_d_x_k = (coord_k[0] - coord_i[0]) / r_ik;
                                    double d_r_jk_d_x_j = (coord_j[0] - coord_k[0]) / r_jk;
                                    double d_r_jk_d_x_k = (coord_k[0] - coord_j[0]) / r_jk;

                                    double d_r_ij_d_y_i = (coord_i[1] - coord_j[1]) / r_ij;
                                    double d_r_ij_d_y_j = (coord_j[1] - coord_i[1]) / r_ij;
                                    double d_r_ik_d_y_i = (coord_i[1] - coord_k[1]) / r_ik;
                                    double d_r_ik_d_y_k = (coord_k[1] - coord_i[1]) / r_ik;
                                    double d_r_jk_d_y_j = (coord_j[1] - coord_k[1]) / r_jk;
                                    double d_r_jk_d_y_k = (coord_k[1] - coord_j[1]) / r_jk;
                                    
                                    double d_r_ij_d_z_i = (coord_i[2] - coord_j[2]) / r_ij;
                                    double d_r_ij_d_z_j = (coord_j[2] - coord_i[2]) / r_ij;
                                    double d_r_ik_d_z_i = (coord_i[2] - coord_k[2]) / r_ik;
                                    double d_r_ik_d_z_k = (coord_k[2] - coord_i[2]) / r_ik;
                                    double d_r_jk_d_z_j = (coord_j[2] - coord_k[2]) / r_jk;
                                    double d_r_jk_d_z_k = (coord_k[2] - coord_j[2]) / r_jk;
                                    
                                    double d_zeta_prefac = zeta * fastpown(1 + lambda * cos_theta, (int)(zeta - 1));
                                    double zeta_prefac = fastpown((1 + lambda * cos_theta), (int)zeta);
                                    double d_cos_theta_d_coord[9];
                                    double R_n_r_ij = R_sup_n(r_ij, n, r_c);
                                    double R_m_r_ik = R_sup_n(r_ik, m, r_c);
                                    double R_p_r_jk = R_sup_n(r_jk, p, r_c);
                                    double d_R_d_r_ij = d_R_sup_n_d_r(r_ij, n, r_c);
                                    double d_R_d_r_ik = d_R_sup_n_d_r(r_ik, m, r_c);
                                    double d_R_d_r_jk = d_R_sup_n_d_r(r_jk, p, r_c);

                                    d_cos_bond_angle_d_coord(coord_i, coord_j, coord_k, d_cos_theta_d_coord);

                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_i] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[0] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * d_R_d_r_ij * d_r_ij_d_x_i * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_x_i * R_p_r_jk);
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_i] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[1] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * d_R_d_r_ij * d_r_ij_d_y_i * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_y_i * R_p_r_jk);
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_i] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[2] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * d_R_d_r_ij * d_r_ij_d_z_i * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_z_i * R_p_r_jk);
                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_j] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[3] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * d_R_d_r_ij * d_r_ij_d_x_j * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_x_j);
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_j] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[4] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * d_R_d_r_ij * d_r_ij_d_y_j * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_y_j);
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_j] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[5] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * d_R_d_r_ij * d_r_ij_d_z_j * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_z_j);
                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_k] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[6] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_x_k * R_p_r_jk + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_x_k);
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_k] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[7] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_y_k * R_p_r_jk + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_y_k);
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_k] += prefac_2 * (d_zeta_prefac * lambda * d_cos_theta_d_coord[8] * R_n_r_ij * R_m_r_ik * R_p_r_jk + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_z_k * R_p_r_jk + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_z_k);

                                    result += (zeta_prefac * R_n_r_ij * R_m_r_ik * R_p_r_jk);
                                }
                            }

                            sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx] = result * fastpown(2, (int)(1 - zeta));
                            N_PTSD_count_idx++;
                            break;
                        }
                        case 4:
                        {
                            int nb1, nb2;
                            int L, M;
                            double n, m, p;
                            double * coord_i = frame_info[i].coord[j];
                            double result = 0;
                            double r_c = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].cutoff_radius;
                            int * params_type = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].neigh_type_array;
                            int N_body = parameters_PTSDs_info->PTSD_N_body_type[k];
                            int N_nei = N_body - 1;
                            L = (int)(parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[0]);
                            n = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[1];
                            m = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[2];
                            p = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[3];
                            int idx_i = j;
                            double derivative_prefactor = 0; 
                            for (M = - L; M <= L; M++)
                            {
                                std::complex<double> result_inner = (0, 0);
                                int d_idx;
                                std::complex<double> * derivative_tmp = (std::complex<double> *)calloc(3 * parameters_info->N_Atoms_max, sizeof(std::complex<double>));
                                for (nb1 = 0; nb1 <= parameters_info->SEL_A_max - 1; nb1++)
                                {
                                    double * coord_j = frame_info[i].neighbour_list[j].coord_neighbours[nb1];
                                    double r_ij = sqrt(fastpow2(coord_i[0] - coord_j[0], 2) + fastpow2(coord_i[1] - coord_j[1], 2) + fastpow2(coord_i[2] - coord_j[2], 2));
                                    if (r_ij > r_c)
                                    {
                                        break;
                                    }
                                    for (nb2 = nb1 + 1; nb2 <= parameters_info->SEL_A_max - 1; nb2++)
                                    {
                                        int idx_j = frame_info[i].neighbour_list[j].index_neighbours[nb1];
                                        int idx_k = frame_info[i].neighbour_list[j].index_neighbours[nb2];
                                        int current_type[2] = {frame_info[i].neighbour_list[j].type[nb1], frame_info[i].neighbour_list[j].type[nb2]};
                                        double * coord_k = frame_info[i].neighbour_list[j].coord_neighbours[nb2];
                                        if (compare_Nei_type(N_nei, current_type, params_type) == 0)
                                        {
                                            continue;
                                        }
                                        double r_ik = sqrt(fastpow2(coord_i[0] - coord_k[0], 2) + fastpow2(coord_i[1] - coord_k[1], 2) + fastpow2(coord_i[2] - coord_k[2], 2));
                                        if (r_ik > r_c)
                                        {
                                            break;
                                        }
                                        double r_jk = sqrt(fastpow2(coord_j[0] - coord_k[0], 2) + fastpow2(coord_j[1] - coord_k[1], 2) + fastpow2(coord_j[2] - coord_k[2], 2));
                                        double coord_ij[3] = {coord_i[0] - coord_j[0], coord_i[1] - coord_j[1], coord_i[2] - coord_j[2]};
                                        double coord_ik[3] = {coord_i[0] - coord_k[0], coord_i[1] - coord_k[1], coord_i[2] - coord_k[2]};
                                        std::complex<double> R_Y;
                                        std::complex<double> Y_LM_IJ = Y_LM(coord_ij, L, M);
                                        std::complex<double> Y_LM_IK = Y_LM(coord_ik, L, M);
                                        double R_n_r_ij = R_sup_n(r_ij, n, r_c);
                                        double R_m_r_ik = R_sup_n(r_ik, m, r_c);
                                        double R_p_r_jk = R_sup_n(r_jk, p, r_c);
                                        R_Y = Y_LM_IJ + Y_LM_IK;
                                        R_Y = R_Y * R_n_r_ij * R_m_r_ik * R_p_r_jk;
                                        result_inner += R_Y;
                                        /*For derivative*/
                                        double d_r_ij_d_x_i = (coord_i[0] - coord_j[0]) / r_ij;
                                        double d_r_ij_d_x_j = (coord_j[0] - coord_i[0]) / r_ij;
                                        double d_r_ik_d_x_i = (coord_i[0] - coord_k[0]) / r_ik;
                                        double d_r_ik_d_x_k = (coord_k[0] - coord_i[0]) / r_ik;
                                        double d_r_jk_d_x_j = (coord_j[0] - coord_k[0]) / r_jk;
                                        double d_r_jk_d_x_k = (coord_k[0] - coord_j[0]) / r_jk;

                                        double d_r_ij_d_y_i = (coord_i[1] - coord_j[1]) / r_ij;
                                        double d_r_ij_d_y_j = (coord_j[1] - coord_i[1]) / r_ij;
                                        double d_r_ik_d_y_i = (coord_i[1] - coord_k[1]) / r_ik;
                                        double d_r_ik_d_y_k = (coord_k[1] - coord_i[1]) / r_ik;
                                        double d_r_jk_d_y_j = (coord_j[1] - coord_k[1]) / r_jk;
                                        double d_r_jk_d_y_k = (coord_k[1] - coord_j[1]) / r_jk;
                                    
                                        double d_r_ij_d_z_i = (coord_i[2] - coord_j[2]) / r_ij;
                                        double d_r_ij_d_z_j = (coord_j[2] - coord_i[2]) / r_ij;
                                        double d_r_ik_d_z_i = (coord_i[2] - coord_k[2]) / r_ik;
                                        double d_r_ik_d_z_k = (coord_k[2] - coord_i[2]) / r_ik;
                                        double d_r_jk_d_z_j = (coord_j[2] - coord_k[2]) / r_jk;
                                        double d_r_jk_d_z_k = (coord_k[2] - coord_j[2]) / r_jk;

                                        double d_R_d_r_ij = d_R_sup_n_d_r(r_ij, n, r_c);
                                        double d_R_d_r_ik = d_R_sup_n_d_r(r_ik, m, r_c);
                                        double d_R_d_r_jk = d_R_sup_n_d_r(r_jk, p, r_c);

                                        double d_theta_ij_d_x_i = (coord_ij[0]) * (coord_ij[2]) / (fastpown(r_ij, 3) * sqrt(1.0 - fastpow2(coord_ij[2], 2) / (fastpow2(r_ij, 2))));
                                        double d_theta_ij_d_x_j = 0.0 - d_theta_ij_d_x_i;
                                        double d_theta_ij_d_y_i = (coord_ij[1]) * (coord_ij[2]) / (fastpown(r_ij, 3) * sqrt(1.0 - fastpow2(coord_ij[2], 2) / (fastpow2(r_ij, 2))));
                                        double d_theta_ij_d_y_j = 0.0 - d_theta_ij_d_y_i;
                                        double d_theta_ij_d_z_i = 0.0 - (1.0 / r_ij - fastpow2(coord_ij[2], 2) / fastpown(r_ij, 3)) / (1.0 - fastpow2(coord_ij[2], 2) / fastpow2(r_ij, 2));
                                        double d_theta_ij_d_z_j = 0.0 - d_theta_ij_d_z_i;
                                        double d_phi_ij_d_x_i = 0.0 - (coord_ij[1]) / ((fastpow2(coord_ij[0], 2)) * (1 + fastpow2(coord_ij[1] / coord_ij[0], 2)));
                                        double d_phi_ij_d_x_j = 0.0 - d_phi_ij_d_x_i;
                                        double d_phi_ij_d_y_i = 1.0 / (coord_ij[0] * (1 + fastpow2(coord_ij[1] / coord_ij[0], 2)));
                                        double d_phi_ij_d_y_j = 0.0 - d_phi_ij_d_y_i;
                                        double d_phi_ij_d_z_i = 0.0;
                                        double d_phi_ij_d_z_j = 0.0;
                                        std::complex<double> D_YLM_IJ_D_THETA = d_Y_LM_d_theta(coord_ij, L, M);
                                        std::complex<double> D_YLM_IJ_D_PHI = d_Y_LM_d_phi(coord_ij, L, M);

                                        double d_theta_ik_d_x_i = (coord_ik[0]) * (coord_ik[2]) / (fastpown(r_ik, 3) * sqrt(1.0 - fastpow2(coord_ik[2], 2) / (fastpow2(r_ik, 2))));
                                        double d_theta_ik_d_x_k = 0.0 - d_theta_ik_d_x_i;
                                        double d_theta_ik_d_y_i = (coord_ik[1]) * (coord_ik[2]) / (fastpown(r_ik, 3) * sqrt(1.0 - fastpow2(coord_ik[2], 2) / (fastpow2(r_ik, 2))));
                                        double d_theta_ik_d_y_k = 0.0 - d_theta_ik_d_y_i;
                                        double d_theta_ik_d_z_i = 0.0 - (1.0 / r_ik - fastpow2(coord_ik[2], 2) / fastpown(r_ik, 3)) / (1.0 - fastpow2(coord_ik[2], 2) / fastpow2(r_ik, 2));
                                        double d_theta_ik_d_z_k = 0.0 - d_theta_ik_d_z_i;
                                        double d_phi_ik_d_x_i = 0.0 - (coord_ik[1]) / ((fastpow2(coord_ik[0], 2)) * (1 + fastpow2(coord_ik[1] / coord_ik[0], 2)));
                                        double d_phi_ik_d_x_k = 0.0 - d_phi_ik_d_x_i;
                                        double d_phi_ik_d_y_i = 1.0 / (coord_ik[0] * (1 + fastpow2(coord_ik[1] / coord_ik[0], 2)));
                                        double d_phi_ik_d_y_k = 0.0 - d_phi_ik_d_y_i;
                                        double d_phi_ik_d_z_i = 0.0;
                                        double d_phi_ik_d_z_k = 0.0;
                                        std::complex<double> D_YLM_IK_D_THETA = d_Y_LM_d_theta(coord_ik, L, M);
                                        std::complex<double> D_YLM_IK_D_PHI = d_Y_LM_d_phi(coord_ik, L, M);

                                        //d to x
                                        derivative_tmp[idx_i] += d_R_d_r_ij * d_r_ij_d_x_i * R_m_r_ik * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * d_R_d_r_ik * d_r_ij_d_x_i * R_p_r_jk * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (D_YLM_IJ_D_THETA * d_theta_ij_d_x_i + D_YLM_IJ_D_PHI * d_phi_ij_d_x_i + D_YLM_IK_D_THETA * d_phi_ik_d_x_i + D_YLM_IK_D_PHI * d_phi_ik_d_x_i);
                                        derivative_tmp[idx_j] += d_R_d_r_ij * d_r_ij_d_x_j * R_m_r_ik * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_x_j * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (D_YLM_IJ_D_THETA * d_theta_ij_d_x_j + D_YLM_IJ_D_PHI * d_phi_ij_d_x_j + Y_LM_IK);
                                        derivative_tmp[idx_k] += R_n_r_ij * d_R_d_r_ik * d_r_ik_d_x_k * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_x_k * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (Y_LM_IJ + D_YLM_IK_D_THETA * d_phi_ik_d_x_k + D_YLM_IK_D_PHI * d_phi_ik_d_x_k);
                                        //d to y
                                        derivative_tmp[parameters_info->N_Atoms_max - 1 + idx_i] += d_R_d_r_ij * d_r_ij_d_y_i * R_m_r_ik * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * d_R_d_r_ik * d_r_ij_d_y_i * R_p_r_jk * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (D_YLM_IJ_D_THETA * d_theta_ij_d_y_i + D_YLM_IJ_D_PHI * d_phi_ij_d_y_i + D_YLM_IK_D_THETA * d_phi_ik_d_y_i + D_YLM_IK_D_PHI * d_phi_ik_d_y_i);
                                        derivative_tmp[parameters_info->N_Atoms_max - 1 + idx_j] += d_R_d_r_ij * d_r_ij_d_y_j * R_m_r_ik * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_y_j * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (D_YLM_IJ_D_THETA * d_theta_ij_d_y_j + D_YLM_IJ_D_PHI * d_phi_ij_d_y_j + Y_LM_IK);
                                        derivative_tmp[parameters_info->N_Atoms_max - 1 + idx_k] += R_n_r_ij * d_R_d_r_ik * d_r_ik_d_y_k * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_y_k * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (Y_LM_IJ + D_YLM_IK_D_THETA * d_phi_ik_d_y_k + D_YLM_IK_D_PHI * d_phi_ik_d_y_k);
                                        //d to z
                                        derivative_tmp[2 * parameters_info->N_Atoms_max - 1 + idx_i] += d_R_d_r_ij * d_r_ij_d_z_i * R_m_r_ik * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * d_R_d_r_ik * d_r_ij_d_z_i * R_p_r_jk * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (D_YLM_IJ_D_THETA * d_theta_ij_d_z_i + D_YLM_IJ_D_PHI * d_phi_ij_d_z_i + D_YLM_IK_D_THETA * d_phi_ik_d_z_i + D_YLM_IK_D_PHI * d_phi_ik_d_z_i);
                                        derivative_tmp[2 * parameters_info->N_Atoms_max - 1 + idx_j] += d_R_d_r_ij * d_r_ij_d_z_j * R_m_r_ik * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_z_j * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (D_YLM_IJ_D_THETA * d_theta_ij_d_z_j + D_YLM_IJ_D_PHI * d_phi_ij_d_z_j + Y_LM_IK);
                                        derivative_tmp[2 * parameters_info->N_Atoms_max - 1 + idx_k] += R_n_r_ij * d_R_d_r_ik * d_r_ik_d_z_k * R_p_r_jk *(Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * d_R_d_r_jk * d_r_jk_d_z_k * (Y_LM_IJ + Y_LM_IK) + R_n_r_ij * R_m_r_ik * R_p_r_jk * (Y_LM_IJ + D_YLM_IK_D_THETA * d_phi_ik_d_z_k + D_YLM_IK_D_PHI * d_phi_ik_d_z_k);
                                    }
                                }
                                result += std::norm(result_inner);
                                for (d_idx = 0; d_idx <= 3 * parameters_info->N_Atoms_max - 1; d_idx++)
                                {
                                    derivative_tmp[d_idx] *= (2.0 * result_inner);
                                }
                                for (d_idx = 0; d_idx <= parameters_info->N_Atoms_max - 1; d_idx++)
                                {
                                    sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][d_idx] += derivative_tmp[d_idx].real() + derivative_tmp[d_idx].imag();
                                    sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][d_idx] += derivative_tmp[parameters_info->N_Atoms_max - 1 + d_idx].real() + derivative_tmp[parameters_info->N_Atoms_max - 1 + d_idx].imag();
                                    sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][d_idx] += derivative_tmp[2 * parameters_info->N_Atoms_max  - 1 + d_idx].real() + derivative_tmp[2 * parameters_info->N_Atoms_max  - 1 + d_idx].imag();
                                }

                                free(derivative_tmp);
                            }

                            
                            sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx] = sqrt(result);
                            if (result <= 1E-8 )
                            {
                                derivative_prefactor = 0;
                            }
                            else
                            {
                                derivative_prefactor = 0.5 / sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx];
                            }

                            int d_idx = 0;
                            /*Derivative = prefactor * \sum^L_(M=-L) 2 * result_inner * \sum \partial RYLM/ \partial x,y,z*/
                            for (d_idx = 0; d_idx <= parameters_info->N_Atoms_max - 1; d_idx++)
                            {
                                sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][d_idx] *= derivative_prefactor;
                                sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][d_idx] *= derivative_prefactor;
                                sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][d_idx] *= derivative_prefactor;
                            }

                            N_PTSD_count_idx++;
                            break;
                        }
                        case 5:
                        {
                            int nb1, nb2, nb3;
                            double n, m, p, zeta, lambda;
                            double * coord_i = frame_info[i].coord[j];
                            double result = 0;
                            double r_c = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].cutoff_radius;
                            n = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[0];
                            m = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[1];
                            p = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[2];
                            zeta = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[3];
                            lambda = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].params_array[4];
                            int * params_type = parameters_PTSDs_info->parameters_PTSDs_info_one_line[ii][k][l].neigh_type_array;
                            int N_body = parameters_PTSDs_info->PTSD_N_body_type[k];
                            int N_nei = N_body - 1;
                            int idx_i = j;
                            double prefac_2 = fastpown(2, (int)(1 - zeta));
                            for (nb1 = 0; nb1 <= parameters_info->SEL_A_max - 1; nb1++)
                            {
                                double * coord_j = frame_info[i].neighbour_list[j].coord_neighbours[nb1];
                                double r_ij = sqrt(fastpow2(coord_i[0] - coord_j[0], 2) + fastpow2(coord_i[1] - coord_j[1], 2) + fastpow2(coord_i[2] - coord_j[2], 2));
                                if (r_ij > r_c)
                                {
                                    break;
                                }
                                for (nb2 = nb1 + 1; nb2 <= parameters_info->SEL_A_max - 1; nb2++)
                                {
                                    double * coord_k = frame_info[i].neighbour_list[j].coord_neighbours[nb2];
                                    double r_ik = sqrt(fastpow2(coord_i[0] - coord_k[0], 2) + fastpow2(coord_i[1] - coord_k[1], 2) + fastpow2(coord_i[2] - coord_k[2], 2));
                                    if (r_ik > r_c)
                                    {
                                        break;
                                    }
                                    for (nb3 = nb2 + 1; nb3 <= parameters_info->SEL_A_max - 1; nb3++)
                                    {
                                        double * coord_l = frame_info[i].neighbour_list[j].coord_neighbours[nb3];
                                        double r_il = sqrt(fastpow2(coord_i[0] - coord_l[0], 2) + fastpow2(coord_i[1] - coord_l[1], 2) + fastpow2(coord_i[2] - coord_l[2], 2));
                                        if (r_il > r_c)
                                        {
                                            break;
                                        }
                                        
                                        int idx_j = frame_info[i].neighbour_list[j].index_neighbours[nb1];
                                        int idx_k = frame_info[i].neighbour_list[j].index_neighbours[nb2];
                                        int idx_l = frame_info[i].neighbour_list[j].index_neighbours[nb3];
                                        double cos_delta = cos_dihedral_angle(coord_i, coord_j, coord_k, coord_l);
                                        double d_cos_delta_d_coord[12];
                                        d_cos_dihedral_angle_d_coord(coord_i, coord_j, coord_k, coord_l, d_cos_delta_d_coord);
                                        
                                        double d_r_ij_d_x_i = (coord_i[0] - coord_j[0]) / r_ij;
                                        double d_r_ij_d_x_j = (coord_j[0] - coord_i[0]) / r_ij;
                                        double d_r_ik_d_x_i = (coord_i[0] - coord_k[0]) / r_ik;
                                        double d_r_ik_d_x_k = (coord_k[0] - coord_i[0]) / r_ik;
                                        double d_r_il_d_x_i = (coord_i[0] - coord_l[0]) / r_il;
                                        double d_r_il_d_x_l = (coord_l[0] - coord_i[0]) / r_il;

                                        double d_r_ij_d_y_i = (coord_i[1] - coord_j[1]) / r_ij;
                                        double d_r_ij_d_y_j = (coord_j[1] - coord_i[1]) / r_ij;
                                        double d_r_ik_d_y_i = (coord_i[1] - coord_k[1]) / r_ik;
                                        double d_r_ik_d_y_k = (coord_k[1] - coord_i[1]) / r_ik;
                                        double d_r_il_d_y_i = (coord_i[1] - coord_l[1]) / r_il;
                                        double d_r_il_d_y_l = (coord_l[1] - coord_i[1]) / r_il;
                                        
                                        double d_r_ij_d_z_i = (coord_i[2] - coord_j[2]) / r_ij;
                                        double d_r_ij_d_z_j = (coord_j[2] - coord_i[2]) / r_ij;
                                        double d_r_ik_d_z_i = (coord_i[2] - coord_k[2]) / r_ik;
                                        double d_r_ik_d_z_k = (coord_k[2] - coord_i[2]) / r_ik;
                                        double d_r_il_d_z_i = (coord_i[2] - coord_l[2]) / r_il;
                                        double d_r_il_d_z_l = (coord_l[2] - coord_i[2]) / r_il;
                                        
                                        double d_zeta_prefac = zeta * fastpown(1 + lambda * cos_delta, (int)(zeta - 1));
                                        double zeta_prefac = fastpown((1 + lambda * cos_delta), (int)zeta);
                                        double R_n_r_ij = R_sup_n(r_ij, n, r_c);
                                        double R_m_r_ik = R_sup_n(r_ik, m, r_c);
                                        double R_p_r_il = R_sup_n(r_il, p, r_c);
                                        double d_R_d_r_ij = d_R_sup_n_d_r(r_ij, n, r_c);
                                        double d_R_d_r_ik = d_R_sup_n_d_r(r_ik, m, r_c);
                                        double d_R_d_r_il = d_R_sup_n_d_r(r_il, p, r_c);

                                        sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_i] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[0] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * d_R_d_r_ij * d_r_ij_d_x_i * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_x_i * R_p_r_il + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_il * d_r_il_d_x_i;
                                        sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_i] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[1] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * d_R_d_r_ij * d_r_ij_d_y_i * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_y_i * R_p_r_il + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_il * d_r_il_d_y_i;
                                        sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_i] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[2] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * d_R_d_r_ij * d_r_ij_d_z_i * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_z_i * R_p_r_il + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_il * d_r_il_d_z_i;
                                        sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_j] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[3] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * d_R_d_r_ij * d_r_ij_d_x_j * R_m_r_ik * R_p_r_il;
                                        sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_j] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[4] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * d_R_d_r_ij * d_r_ij_d_y_j * R_m_r_ik * R_p_r_il;
                                        sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_j] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[5] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * d_R_d_r_ij * d_r_ij_d_z_j * R_m_r_ik * R_p_r_il;
                                        sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_k] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[6] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_x_k * R_p_r_il;
                                        sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_k] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[7] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_y_k * R_p_r_il;
                                        sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_k] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[8] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * d_R_d_r_ik * d_r_ik_d_z_k * R_p_r_il;
                                        sym_coord_LASP[i].d_x[j][N_PTSD_count_idx][idx_l] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[9] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_il * d_r_il_d_x_l;
                                        sym_coord_LASP[i].d_y[j][N_PTSD_count_idx][idx_l] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[10] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_il * d_r_il_d_y_l;
                                        sym_coord_LASP[i].d_z[j][N_PTSD_count_idx][idx_l] += prefac_2 * d_zeta_prefac * lambda * d_cos_delta_d_coord[11] * R_n_r_ij * R_m_r_ik * R_p_r_il + zeta_prefac * R_n_r_ij * R_m_r_ik * d_R_d_r_il * d_r_il_d_z_l;

                                        result += (zeta_prefac * R_n_r_ij * R_m_r_ik * R_p_r_il);
                                    }
                                }
                            }
                            sym_coord_LASP[i].coord_converted[j][N_PTSD_count_idx] = result * prefac_2;
                            N_PTSD_count_idx++;
                            break;
                        }
                        default:
                        {
                            break;
                        }
                    }
                }
            }
            printf_d("N_PTSD_count = %d\n", N_PTSD_count_idx);
        }
    }

    *(sym_coord_LASP_struct **)sym_coord = sym_coord_LASP;
    for (i = 0; i <= parameters_info->N_types_all_frame - 1; i++)
    {
        for (j = 0; j <= parameters_PTSDs_info->N_PTSD_types - 1; j ++)
        {
            for (k = 0; k <= parameters_PTSDs_info->N_cutoff_radius[i][j] - 1; k++)
            {
                free(parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j][k].neigh_type_array);
                free(parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j][k].params_array);
            }
            free(parameters_PTSDs_info->parameters_PTSDs_info_one_line[i][j]);
        }
        free(parameters_PTSDs_info->parameters_PTSDs_info_one_line[i]);
        free(parameters_PTSDs_info->N_cutoff_radius[i]);
        free(parameters_PTSDs_info->N_neigh_inter[i]);
    }
    free(parameters_PTSDs_info->parameters_PTSDs_info_one_line);
    free(parameters_PTSDs_info->PTSD_N_body_type);
    free(parameters_PTSDs_info->PTSD_N_params);
    free(parameters_PTSDs_info->N_cutoff_radius);
    free(parameters_PTSDs_info->N_neigh_inter);

    return 0;
    //return (printf("Not completed, return 999.\n"), 999);//incomplete

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
    parameters_PTSDs_info->PTSD_N_body_type = (int *)calloc(6, sizeof(int));
    parameters_PTSDs_info->PTSD_N_body_type[0] = 2; parameters_PTSDs_info->PTSD_N_body_type[1] = 2; parameters_PTSDs_info->PTSD_N_body_type[2] = 3; parameters_PTSDs_info->PTSD_N_body_type[3] = 3; parameters_PTSDs_info->PTSD_N_body_type[4] = 3; parameters_PTSDs_info->PTSD_N_body_type[5] = 4; 
    parameters_PTSDs_info->PTSD_N_params = (int *)calloc(6, sizeof(int));
    parameters_PTSDs_info->PTSD_N_params[0] = 1; parameters_PTSDs_info->PTSD_N_params[1] = 2; parameters_PTSDs_info->PTSD_N_params[2] = 4; parameters_PTSDs_info->PTSD_N_params[3] = 5; parameters_PTSDs_info->PTSD_N_params[4] = 4; parameters_PTSDs_info->PTSD_N_params[5] = 5; 
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
            if (ii >= parameters_info->N_types_all_frame)
            {
                printf("Type error: type %d in LASP.raw does not exist in type.raw!\n", center_type);
                return 33;
            }
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
                free(neighb_atom_array);
                free(params_array);
            }
            parameters_PTSDs_info->N_cutoff_radius[ii][jj] = cutoff_radius_pointer;
        }
    }

    fclose(fp);
    free(tmp_line);
    return 0;

}

