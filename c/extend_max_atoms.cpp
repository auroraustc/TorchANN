/*
2019.04.03 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Extend system according to parameters_info->N_Atoms_max by adding dummy atoms.

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
//#define DEBUG_EXT

#ifdef DEBUG_EXT
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int extend_max_atoms(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    int extend_max_atoms_cur_frame(frame_info_struct * frame_info, parameters_info_struct * parameters_info);

    int i, j, k;
    int N_Atoms_max = parameters_info->N_Atoms_max;
    int Nframes_tot = parameters_info->Nframes_tot;

    for (i = 0; i <= Nframes_tot - 1; i ++)
    {
        extend_max_atoms_cur_frame(&(frame_info[i]), parameters_info);
    }
    return 0;

}

int extend_max_atoms_cur_frame(frame_info_struct * frame_info_cur, parameters_info_struct * parameters_info)
{
    int i, j, k;
    int N_Atoms_max = parameters_info->N_Atoms_max;
    int N_Atoms_cur = frame_info_cur->N_Atoms;
    double ** coord_ext = (double **)calloc(N_Atoms_max, sizeof(double *));
    double ** force_ext = (double **)calloc(N_Atoms_max, sizeof(double *));
    int * type_ext = (int *)calloc(N_Atoms_max, sizeof(int));
    double interval = (double)(int)(parameters_info->cutoff_max + 1) * 100;
    double radius = (N_Atoms_max - N_Atoms_cur) * interval + 100000;

    for (i = 0; i <= N_Atoms_max - 1; i++)
    {
        coord_ext[i] = (double *)calloc(3, sizeof(double));
        force_ext[i] = (double *)calloc(3, sizeof(double));
    }

    for (i = 0; i <= N_Atoms_cur - 1; i++)
    {
        for (j = 0; j <= 2; j ++)
        {
            coord_ext[i][j] = frame_info_cur->coord[i][j];
            force_ext[i][j] = frame_info_cur->force[i][j];
        }
        type_ext[i] = frame_info_cur->type[i];
    }
    for (i = N_Atoms_cur; i <= N_Atoms_max - 1; i++)
    {
        coord_ext[i][0] = radius - i * interval;
        coord_ext[i][1] = sqrt(radius * radius - coord_ext[i][0] * coord_ext[i][0]);
        coord_ext[i][2] = 0.0;
        for (j = 0; j <= 2; j++)
        {
            force_ext[i][j] = 0.0;
        }
        type_ext[i] = frame_info_cur->type[0];
    }

    for (i = 0; i <= N_Atoms_cur - 1; i++)
    {
        free(frame_info_cur->coord[i]);
        free(frame_info_cur->force[i]);
    }
    free(frame_info_cur->coord);free(frame_info_cur->force);free(frame_info_cur->type);
    frame_info_cur->coord = coord_ext;
    frame_info_cur->force = force_ext;
    frame_info_cur->type = type_ext;
    frame_info_cur->N_Atoms = N_Atoms_max;

    return 0;
}
