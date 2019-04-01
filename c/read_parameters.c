/*
2019.03.28 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Read in parameters from file.

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_PARAM

#ifdef DEBUG_PARAM
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int read_parameters(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    FILE * fp_param = NULL;
    int i, j, k;
    int N_Atoms_max;

    parameters_info->cutoff_1 = 7.7;
    parameters_info->cutoff_2 = 8.0;
    parameters_info->cutoff_3 = 0.0;
    parameters_info->cutoff_max = 8.0;

    N_Atoms_max = 0;
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        N_Atoms_max = (N_Atoms_max <= frame_info[i].N_Atoms ? frame_info[i].N_Atoms : N_Atoms_max);
        printf_d("N_Atoms: %d\n", frame_info[i].N_Atoms);
    }
    parameters_info->N_Atoms_max = N_Atoms_max;

    return 0;
}