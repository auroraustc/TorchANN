/*
2019.03.28 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Build neighbour lists for each frame.

[Y] = set in this module, [N] = not set in this module:
typedef struct frame_info_struct_
{
[N]	int index;
[N]	int N_Atoms;
[N]	double box[3][3];
[N]	int * type;//type[0..N_Atoms-1]
[N]	double ** coord;//coord[0..N_Atoms-1][0..2]
[N]	double energy;
[N]	int no_force;
[N]	double ** force;//force[0..N_Atoms-1][0..2]
[Y]	neighbour_list_struct * neighbour_list;//neighbour_list[0..N_Atoms-1], neighbour list for each atom
}

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_BUILD

#ifdef DEBUG_BUILD
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int build_neighbour_list(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info)
{
    int build_neighbour_list_one_frame(frame_info_struct * frame_info_cur, parameters_info_struct * parameters_info);

    int i;
    int error_code = 0;
    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        error_code = build_neighbour_list_one_frame(&(frame_info[i]), parameters_info);
    }
    return error_code;
}

int build_neighbour_list_one_frame(frame_info_struct * frame_info_cur, parameters_info_struct * parameters_info)
{
    int expand_system_one_frame(frame_info_struct * frame_info_cur, system_info_expanded_struct * system_info_expanded, parameters_info_struct * parameters_info);

    int i, j, k;
    int error_code = 0;
    system_info_expanded_struct * system_info_expanded;
    neighbour_list_struct * neighbour_list_cur;
    
    expand_system_one_frame(frame_info_cur, system_info_expanded, parameters_info);

    return error_code;
}

int expand_system_one_frame(frame_info_struct * frame_info_cur, system_info_expanded_struct * system_info_expanded, parameters_info_struct * parameters_info)
{
    int i, j, k, l;
    int tmpi1;
    double cutoff_max;
    int expand_x_period;
    int expand_y_period;
    int expand_z_period;
    int * expand_x_array;
    int * expand_y_array;
    int * expand_z_array;
    
    cutoff_max = parameters_info->cutoff_max;


    return 0;
}