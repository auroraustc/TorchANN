/*
2019.04.13 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Compute partial_f / partial_x_i, partial_f / partial_y_i. partial_f / partial_z_i for each atom i in one frame. f is the sym_coord of every atom in the frame.
Args: Nframes_tot, frame_idx, SEL_A_max, N_Atoms
Read in the COORD.BIN, NEI_COORD.BIN, NEI_IDX.BIN; Save the D_SYM_D_COORD.BIN (an array of shape = [N_Atoms * SEL_A_max * 4])

Return code:
	double *: No errors.
    NULL: Read input files error.
	
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/*****************MACRO FOR DEBUG*****************/
#define DEBUG_DERI
#define DEBUG_FRAME_IDX 0
#define FRAME_TOT 2
#define N_ATOMS_ 57

#ifdef DEBUG_DERI
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

#define PI 3.141592653589793238462643383279

/*int main()
{
    double * compute_derivative_sym_coord_to_coord_one_frame_DeePMD(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms, double rc, double rcs, double * coord_start, double * nei_coord_start, int * nei_idx_start);
    double * init_read_coord(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms);
    double * init_read_nei_coord(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms);
    int * init_read_nei_idx(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms);

    double * derivative_cur_frame;

    derivative_cur_frame = compute_derivative_sym_coord_to_coord_one_frame_DeePMD(FRAME_TOT, DEBUG_FRAME_IDX, 200, N_ATOMS_, 8.0, 7.7, init_read_coord(FRAME_TOT, DEBUG_FRAME_IDX, 200, N_ATOMS_), init_read_nei_coord(FRAME_TOT, DEBUG_FRAME_IDX, 200, N_ATOMS_), init_read_nei_idx(FRAME_TOT, DEBUG_FRAME_IDX, 200, N_ATOMS_));
    free(derivative_cur_frame);
    return 0;
}*/

double * init_read_coord(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms)
{
    FILE * fp_coord;
    fp_coord = fopen("COORD.BIN", "rb");
    double * coord_start = (double *)calloc(Nframes_tot * N_Atoms * 3, sizeof(double));
    fread(coord_start, sizeof(double), Nframes_tot * N_Atoms * 3, fp_coord);
    fclose(fp_coord);
    return coord_start;
}

double * init_read_nei_coord(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms)
{
    FILE * fp_nei_coord;
    fp_nei_coord = fopen("NEI_COORD.BIN", "rb");
    double * coord_start = (double *)calloc(Nframes_tot * N_Atoms * SEL_A_max * 3, sizeof(double));
    fread(coord_start, sizeof(double), Nframes_tot * N_Atoms * SEL_A_max * 3, fp_nei_coord);
    fclose(fp_nei_coord);
    return coord_start;
}

int * init_read_nei_idx(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms)
{
    FILE * fp_nei_idx;
    fp_nei_idx = fopen("NEI_IDX.BIN", "rb");
    int * coord_start = (int *)calloc(Nframes_tot * N_Atoms * SEL_A_max, sizeof(int));
    fread(coord_start, sizeof(int), Nframes_tot * N_Atoms * SEL_A_max, fp_nei_idx);
    fclose(fp_nei_idx);
    return coord_start;
}

double * compute_derivative_sym_coord_to_coord_one_frame_DeePMD(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms, double rc, double rcs, double * coord_start, double * nei_coord_start, int * nei_idx_start)
{
    double fastpow2(double number, int dummy);

    /*FILE * fp_coord;
    FILE * fp_nei_coord;
    FILE * fp_nei_idx;*/
    FILE * fp_derivative;
    int i, j, k, l, m;
    int offset_coord = frame_idx * N_Atoms * 3;// * sizeof(double);
    int offset_nei_coord = frame_idx * N_Atoms * SEL_A_max * 3;// * sizeof(double);
    int offset_nei_idx = frame_idx * N_Atoms * SEL_A_max;// * sizeof(int);
    int read_coord = N_Atoms * 3;// * sizeof(double);
    int read_nei_coord = N_Atoms * SEL_A_max * 3;// * sizeof(double);
    int read_nei_idx = N_Atoms * SEL_A_max;// * sizeof(int)
    int save_size = N_Atoms * SEL_A_max * 4 * N_Atoms * 3;// * sizeof(double);
    double * coord_cur_frame = (double *)calloc(read_coord, sizeof(double));
    double * nei_coord_cur_frame = (double *)calloc(read_nei_coord, sizeof(double));
    int * nei_idx_cur_frame = (int *)calloc(read_nei_idx, sizeof(int));
    double * derivative_cur_frame = (double *)calloc(save_size, sizeof(double));
    int not_nei_count = 0;
    int zero_count = 0;//This should always be zero

    /*fp_coord = fopen("COORD.BIN", "rb");
    fp_nei_coord = fopen("NEI_COORD.BIN", "rb");
    fp_nei_idx = fopen("NEI_IDX.BIN", "rb");*/
    
    /*if ((fp_coord == NULL)||(fp_nei_coord == NULL)||(fp_nei_idx == NULL))
    {
        printf("Read in error. Do COORD.BIN, NEI_COORD.BIN and NEI_IDX.BIN exist?\n");
        return NULL;
    }*/
    /*else
    {
        fseek(fp_coord, offset_coord * sizeof(double), SEEK_SET);
        fseek(fp_nei_coord, offset_nei_coord * sizeof(double), SEEK_SET);
        fseek(fp_nei_idx, offset_nei_idx * sizeof(int), SEEK_SET);
        fread(coord_cur_frame, sizeof(double), read_coord, fp_coord);
        fread(nei_coord_cur_frame, read_nei_coord, sizeof(double), fp_nei_coord);
        fread(nei_idx_cur_frame, read_nei_idx, sizeof(int), fp_nei_idx);
        fclose(fp_coord); fclose(fp_nei_coord); fclose(fp_nei_idx);
        //printf("Read in complete.\n");
    }*/
    coord_cur_frame = coord_start + offset_coord;
    nei_coord_cur_frame = nei_coord_start + offset_nei_coord;
    nei_idx_cur_frame = nei_idx_start + offset_nei_idx;
    /*printf_d("DEBUG TEST frame_idx = %d\n", DEBUG_FRAME_IDX);
    for (i = 0; i <= 5; i++)
    {
        printf_d("Coord: %.6lf %.6lf %.6lf\n", coord_cur_frame[i * 3], coord_cur_frame[i * 3 + 1], coord_cur_frame[i * 3 + 2]);
    }*/

    /* \partial (The i-th atom's j-th row k-th column) / \partial (l-th atom's m coordinate) */
    /* size = N_Atoms * SEL_A_max * 4 * N_Atoms * 3 */
    #pragma omp parallel for private(j, k, l, m)
    for (i = 0; i <= N_Atoms - 1; i++)//loop over atoms
    {
        for (j = 0; j <= SEL_A_max - 1; j++)
        {
            for (k = 0; k <= 3; k++)
            {
                for (l = 0; l <= N_Atoms - 1; l++)
                {
                    for (m = 0; m <= 2; m++)
                    {
                        int index = ((((i * SEL_A_max) + j) * 4 + k) * N_Atoms + l) * 3 + m;
                        int idx_nei = nei_idx_cur_frame[i * SEL_A_max + j] % N_Atoms;
                        double coord_nei[3] = {nei_coord_cur_frame[(i * SEL_A_max + j) * 3 + 0], nei_coord_cur_frame[(i * SEL_A_max + j) * 3 + 1], nei_coord_cur_frame[(i * SEL_A_max + j) * 3 + 2]};
                        double coord_l[3] = {coord_cur_frame[l * 3 + 0], coord_cur_frame[l * 3 + 1], coord_cur_frame[l * 3 + 2]};
                        double coord_i_[3] = {coord_cur_frame[i * 3 + 0], coord_cur_frame[i * 3 + 1], coord_cur_frame[i * 3 + 2]};
                        double coord_nei_ori[3] = {idx_nei == -1 ? 9999 : coord_cur_frame[idx_nei * 3 + 0], idx_nei == -1 ? 9999 : coord_cur_frame[idx_nei * 3 + 1], idx_nei == -1 ? 9999 : coord_cur_frame[idx_nei * 3 + 2]};
                        double r_ji_check;
                        /*if ((i == 4) && (l == 3))
                        {
                            printf_d("Coord check:\n");
                            printf_d("idx_nei: %d\n", idx_nei);
                            printf_d("coord_nei: %.6lf %.6lf %.6lf\n", coord_nei[0], coord_nei[1], coord_nei[2]);
                            printf_d("coord_nei_ori: %.6lf %.6lf %.6lf\n", coord_nei_ori[0], coord_nei_ori[1], coord_nei_ori[2]);
                            printf_d("coord_l: %.6lf %.6lf %.6lf\n", coord_l[0], coord_l[1], coord_l[2]);
                        }*/

                        if ((l != idx_nei) || (l != i))
                        {
                            //derivative_cur_frame[index] = 0;
                            //not_nei_count ++;
                        }
                        else if ((l == i))//atom_l is atom_i; coord_diff = atom_[idx_nei] - atom_l
                        {
                            /*coord_l is coord_i; coord_nei is coord_j*/
                            double coord_i[3] = {coord_l[0], coord_l[1], coord_l[2]};
                            double coord_j[3] = {coord_nei[0], coord_nei[1], coord_nei[2]};
                            double coord_diff[3] = {coord_j[0] - coord_i[0], coord_j[1] - coord_i[1], coord_j[2] - coord_i[2]};
                            double r_ji = sqrt(fastpow2(coord_diff[0], 2) + fastpow2(coord_diff[1], 2) + fastpow2(coord_diff[2], 2));
                            r_ji_check = r_ji;
                            //if (r_ji >= rc) zero_count++;
                            if ((k == 0) && (m == 0))//\partial s_ji / \partial x_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[0] + coord_j[0]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) / (2.0 * (rc - rcs) * r_ji * r_ji) * sin(PI / (rc - rcs) * (0 - rcs + r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[0] + coord_j[0]) / (r_ji * r_ji * r_ji);
                                }
                            }
                            else if ((k == 0) && (m == 1))//\partial s_ji / \partial y_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) / (2.0 * (rc - rcs) * r_ji * r_ji) * sin(PI / (rc - rcs) * (0 - rcs + r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji);
                                }
                            }
                            else if ((k == 0) && (m == 2))//\partial s_ji / \partial z_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji) + PI * (0 - coord_i[2] + coord_j[2]) / (2.0 * (rc - rcs) * r_ji * r_ji) * sin(PI / (rc - rcs) * (0 - rcs + r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji);
                                }
                            }
                            else if ((k == 1) && (m == 0))//\partial s_ji * x_ji / r_ji / \partial x_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[0] + coord_j[0]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) - (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[0] + coord_j[0]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2.0 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[0] + coord_j[0]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 1) && (m == 1))//\partial s_ji * x_ji / r_ji / \partial y_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 1) && (m == 2))//\partial s_ji * x_ji / r_ji/ \partial z_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 2) && (m == 0))//\partial s_ji * y_ji / r_ji / \partial x_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 2) && (m == 1))//\partial s_ji * y_ji / r_ji / \partial y_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) - (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[1] + coord_j[1]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2.0 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 2) && (m == 2))//\partial s_ji * y_ji / r_ji/ \partial z_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 3) && (m == 0))//\partial s_ji * z_ji / r_ji / \partial x_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 3) && (m == 1))//\partial s_ji * z_ji / r_ji / \partial y_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                            else if ((k == 3) && (m == 2))//\partial s_ji * z_ji / r_ji/ \partial z_i
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[2] + coord_j[2]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) - (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji) + PI * (0 - coord_i[2] + coord_j[2]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2.0 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[2] + coord_j[2]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                }
                            }
                        }
                        else if (l == idx_nei)//atom_l is atom_j; coord_diff = atom_l - atom_i
                        {
                            /*coord_l is coord_j; coord_i_ is coord_i*/
                            double coord_i[3] = {coord_i_[0], coord_i_[1], coord_i_[2]};
                            double coord_j[3] = {coord_nei[0], coord_nei[1], coord_nei[2]};
                            double coord_diff[3] = {coord_j[0] - coord_i[0], coord_j[1] - coord_i[1], coord_j[2] - coord_i[2]};
                            double r_ji = sqrt(fastpow2(coord_diff[0], 2) + fastpow2(coord_diff[1], 2) + fastpow2(coord_diff[2], 2));
                            r_ji_check = r_ji;
                            //if (r_ji >= rc) zero_count++;
                            if ((k == 0) && (m == 0))//\partial s_ji / \partial x_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[0] + coord_j[0]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) / (2.0 * (rc - rcs) * r_ji * r_ji) * sin(PI / (rc - rcs) * (0 - rcs + r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[0] + coord_j[0]) / (r_ji * r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 0) && (m == 1))//\partial s_ji / \partial y_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) / (2.0 * (rc - rcs) * r_ji * r_ji) * sin(PI / (rc - rcs) * (0 - rcs + r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 0) && (m == 2))//\partial s_ji / \partial z_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji) + PI * (0 - coord_i[2] + coord_j[2]) / (2.0 * (rc - rcs) * r_ji * r_ji) * sin(PI / (rc - rcs) * (0 - rcs + r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 1) && (m == 0))//\partial s_ji * x_ji / r_ji / \partial x_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[0] + coord_j[0]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) - (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[0] + coord_j[0]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2.0 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[0] + coord_j[0]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 1) && (m == 1))//\partial s_ji * x_ji / r_ji / \partial y_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 1) && (m == 2))//\partial s_ji * x_ji / r_ji/ \partial z_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 2) && (m == 0))//\partial s_ji * y_ji / r_ji / \partial x_j
                            {
                               if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 2) && (m == 1))//\partial s_ji * y_ji / r_ji / \partial y_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[1] + coord_j[1]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) - (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[1] + coord_j[1]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2.0 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[1] + coord_j[1]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 2) && (m == 2))//\partial s_ji * y_ji / r_ji/ \partial z_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 3) && (m == 0))//\partial s_ji * z_ji / r_ji / \partial x_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[0] + coord_j[0]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 3) && (m == 1))//\partial s_ji * z_ji / r_ji / \partial y_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) + PI * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[1] + coord_j[1]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                            else if ((k == 3) && (m == 2))//\partial s_ji * z_ji / r_ji/ \partial z_j
                            {
                                if (r_ji >= rc)
                                {
                                    derivative_cur_frame[index] = 0;
                                }
                                else if (r_ji >= rcs)
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[2] + coord_j[2]) * (0 - coord_i[2] + coord_j[2]) * (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji * r_ji * r_ji) - (0.5 + 0.5 * cos(PI / (rc - rcs) * (0 - rcs + r_ji))) / (r_ji * r_ji) + PI * (0 - coord_i[2] + coord_j[2]) * (0 - coord_i[2] + coord_j[2]) * sin(PI / (rc - rcs) * (0 - rcs + r_ji)) / (2.0 * (rc - rcs) * (r_ji * r_ji * r_ji));
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                                else
                                {
                                    derivative_cur_frame[index] = 2 * (0 - coord_i[2] + coord_j[2]) * (0 - coord_i[2] + coord_j[2]) / (r_ji * r_ji * r_ji * r_ji) - 1.0 / (r_ji * r_ji);
                                    derivative_cur_frame[index] *= (-1.0);
                                }
                            }
                        }
                        //if (index >= 20000000) printf_d("%d\n", index);
                        //if (derivative_cur_frame[index] != 0.0) printf_d("i,j,k,l,m,%d %d %d %d %d :%lf, dist:%lf, idx_nei:%d\n",i, j, k, l, m, derivative_cur_frame[index], r_ji_check, idx_nei);
                    }
                }
            }
        }
    }

    //fp_derivative = fopen("./DERIVATIVE.BIN", "wb");
    //printf_d("save:%d\n",save_size);
    //fwrite(derivative_cur_frame, sizeof(double), save_size, fp_derivative);
    //fclose(fp_derivative);

    //free(coord_cur_frame); free(nei_coord_cur_frame); free(nei_idx_cur_frame);
    //printf_d("%d %d\n", not_nei_count, zero_count);
    //free(derivative_cur_frame);
    return derivative_cur_frame;
}

void freeme(double * ptr)
{
    free(ptr);
}