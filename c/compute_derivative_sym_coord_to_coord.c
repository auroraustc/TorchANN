/*
2019.04.13 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Compute partial_f / partial_x_i, partial_f / partial_y_i. partial_f / partial_z_i for each atom i in one frame. f is the sym_coord of every atom in the frame.
Args: Nframes_tot, frame_idx, SEL_A_max, N_Atoms
Read in the COORD.BIN, NEI_COORD.BIN, NEI_IDX.BIN; Save the D_SYM_D_COORD.BIN (an array of shape = [N_Atoms * SEL_A_max * 4])

Return code:
	0: No errors.
    1: Read input files error.
	
*/

#include <stdio.h>
#include <stdlib.h>
#include "Utilities.c"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_DERI
#define DEBUG_FRAME_IDX 5

#ifdef DEBUG_DERI
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int main()
{
    int compute_derivative_sym_coord_to_coord_one_frame(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms);

    compute_derivative_sym_coord_to_coord_one_frame(55, DEBUG_FRAME_IDX, 200, 93);
    return 0;
}

int compute_derivative_sym_coord_to_coord_one_frame(int Nframes_tot, int frame_idx, int SEL_A_max, int N_Atoms)
{
    FILE * fp_coord;
    FILE * fp_nei_coord;
    FILE * fp_nei_idx;
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

    fp_coord = fopen("COORD.BIN", "rb");
    fp_nei_coord = fopen("NEI_COORD.BIN", "rb");
    fp_nei_idx = fopen("NEI_IDX.BIN", "rb");
    
    if ((fp_coord == NULL)||(fp_nei_coord == NULL)||(fp_nei_idx == NULL))
    {
        printf("Read in error. Do COORD.BIN, NEI_COORD.BIN and NEI_IDX.BIN exist?\n");
        return 1;
    }
    else
    {
        fseek(fp_coord, offset_coord * sizeof(double), SEEK_SET);
        fseek(fp_nei_coord, offset_nei_coord * sizeof(double), SEEK_SET);
        fseek(fp_nei_idx, offset_nei_idx * sizeof(int), SEEK_SET);
        fread(coord_cur_frame, sizeof(double), read_coord, fp_coord);
        fread(nei_coord_cur_frame, read_nei_coord, sizeof(double), fp_nei_coord);
        fread(nei_idx_cur_frame, read_nei_idx, sizeof(int), fp_nei_idx);
        fclose(fp_coord); fclose(fp_nei_coord); fclose(fp_nei_idx);
        printf("Read in complete.\n");
    }
    printf_d("DEBUG TEST frame_idx = %d\n", DEBUG_FRAME_IDX);
    for (i = 0; i <= 5; i++)
    {
        printf_d("Coord: %.6lf %.6lf %.6lf\n", coord_cur_frame[i * 3], coord_cur_frame[i * 3 + 1], coord_cur_frame[i * 3 + 2]);
    }

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
                        double coord_nei_ori[3] = {idx_nei == -1 ? 9999 : coord_cur_frame[idx_nei * 3 + 0], idx_nei == -1 ? 9999 : coord_cur_frame[idx_nei * 3 + 1], idx_nei == -1 ? 9999 : coord_cur_frame[idx_nei * 3 + 2]};
                        

                        /*if ((i == 4) && (l == 3))
                        {
                            printf_d("Coord check:\n");
                            printf_d("idx_nei: %d\n", idx_nei);
                            printf_d("coord_nei: %.6lf %.6lf %.6lf\n", coord_nei[0], coord_nei[1], coord_nei[2]);
                            printf_d("coord_nei_ori: %.6lf %.6lf %.6lf\n", coord_nei_ori[0], coord_nei_ori[1], coord_nei_ori[2]);
                            printf_d("coord_l: %.6lf %.6lf %.6lf\n", coord_l[0], coord_l[1], coord_l[2]);
                        }*/

                        if ((l != i) && (l != nei_idx_cur_frame[i * SEL_A_max + j]))
                        {
                            derivative_cur_frame[index] = 0;
                        }
                        else
                        {

                        }
                    }
                }
            }
        }
    }

    fp_derivative = fopen("./DERIVATIVE.BIN", "wb");
    fwrite(derivative_cur_frame, sizeof(double), save_size, fp_derivative);
    fclose(fp_derivative);

    free(coord_cur_frame); free(nei_coord_cur_frame); free(nei_idx_cur_frame);
    free(derivative_cur_frame);
    return 0;
}