/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Return code:
    0: No errors.
    1: read_system() error.
    2: read_parameters() error.
    3: count types() error.
*/

#include <stdio.h>
#include <stdlib.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_MAIN

#ifdef DEBUG_MAIN
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int main()
{
    int read_system(frame_info_struct ** frame_info_, int * Nframes_tot_);
    int read_parameters(parameters_info_struct * parameters_info);
    int build_neighbour_list(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info);
    int count_types(frame_info_struct * frame_info, int Nframes_tot, int * N_types_all_frame_, int ** type_index_all_frame_);

    int read_system_flag;
    int read_parameters_info_flag;
    int build_neighbour_list_flag;
    int count_types_flag;
    frame_info_struct * frame_info = NULL;
    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));
    int Nframes_tot;
    int max_N_neighbours_all_frame = -1;
    int SEL_A_max;
    int N_types_all_frame = 0;
    int * type_index_all_frame = NULL;//type_index_all_frame[0..N_types_all_frame - 1]. For exampe, we have 3 elements, 6(C), 29(Cu), 1(H), then ...[0]=6, ...[1]=29, ...[2]=1.
    int i, j, k;

    read_system_flag = read_system(&frame_info, &Nframes_tot);
    if (read_system_flag != 0) 
    {
        printf("Error when reading raw data: read_flag = %d\n", read_system_flag);
        return 1;
    }
    printf("No error when reading raw data.\n");

    read_parameters_info_flag = read_parameters(parameters_info);
    if (read_parameters_info_flag != 0)
    {
        printf("Error when reading input parameters: read_parameters_info_flag = %d\n", read_parameters_info_flag);
        return 2;
    }
    printf("No error when reading parameters.\n");

    build_neighbour_list_flag = build_neighbour_list(frame_info, Nframes_tot, parameters_info);
    if (build_neighbour_list_flag != 0)
    {
        printf("Error when building neighbour list: build_neighbour_list_flag = %d\n", build_neighbour_list_flag);
        return 2;
    }
    printf_d("Check neighbour list from main():\n");
    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        if (frame_info[i].max_N_neighbours >= max_N_neighbours_all_frame) max_N_neighbours_all_frame = frame_info[i].max_N_neighbours;
        printf_d("max neighbour atoms of frame %d: %d\n", i + 1, frame_info[i].max_N_neighbours);
        printf_d("In this frame the number of neighbour atoms of 2nd atom is %d\n", frame_info[i].neighbour_list[1].N_neighbours);
    }
    SEL_A_max = 50 * (max_N_neighbours_all_frame / 50 + 1);
    printf("Max number of N_neighbour is %d. SEL_A would be %d\n", max_N_neighbours_all_frame, SEL_A_max);
    printf("No error when building neighbour list.\n");

    count_types_flag = count_types(frame_info, Nframes_tot, &N_types_all_frame, &type_index_all_frame);
    if (count_types_flag != 0)
    {
        printf("Error when counting types: count_types_flag = %d\n", count_types_flag);
        return 3;
    }
    printf_d("Check types from main:\n");
    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        printf_d("N_types of frame %d is %d\n", i + 1, frame_info[i].N_types);
    }
    printf_d("N_types_all_frame is %d\n", N_types_all_frame);
    printf_d("type_index_all_frame is :\n");
    for (i = 0; i <= N_types_all_frame - 1; i++)
    {
        printf_d("%d ", type_index_all_frame[i]);
    }
    printf_d("\n");

    return 0;
}