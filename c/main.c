/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Return code:
    0: No errors.
    1: read_system() error.
    2: read_parameters() error.
    3: build_neighbour_list() error at step1.
    4: count types() error.
    5: build_neighbour_list() error at step2.
    6: convert_coord() error.
    7: save_to_file() error.

*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
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
    int build_neighbour_list(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, int step);
    int count_types(frame_info_struct * frame_info, int Nframes_tot, int * N_types_all_frame_, int ** type_index_all_frame_);
    int convert_coord(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, int coord_type, void ** sym_coord_struct);
    int save_to_file(frame_info_struct * frame_info, parameters_info_struct * parameters_info, void * sym_coord);
    
    struct timeval start_main, end_main;
    double t_main;//Unit: ms

    int read_system_flag;
    int read_parameters_info_flag;
    int build_neighbour_list_flag1, build_neighbour_list_flag2;
    int count_types_flag;
    int convert_coord_flag;
    int save_to_file_flag;
    frame_info_struct * frame_info = NULL;
    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));
    sym_coord_DeePMD_struct * sym_coord_DeePMD = NULL;
    int Nframes_tot;
    int max_N_neighbours_all_frame = -1;
    int SEL_A_max;
    int N_types_all_frame = 0;
    int * type_index_all_frame = NULL;//type_index_all_frame[0..N_types_all_frame - 1]. For exampe, we have 3 elements, 6(C), 29(Cu), 1(H), then ...[0]=6, ...[1]=29, ...[2]=1.
    int sym_coord_type = 1;
    int i, j, k;

    /*Profiling main start*/
    gettimeofday(&start_main, NULL);

    read_system_flag = read_system(&frame_info, &Nframes_tot);
    if (read_system_flag != 0) 
    {
        printf("Error when reading raw data: read_flag = %d\n", read_system_flag);
        return 1;
    }
    printf("No error when reading raw data.\n");
    parameters_info->Nframes_tot = Nframes_tot;

    read_parameters_info_flag = read_parameters(parameters_info);
    if (read_parameters_info_flag != 0)
    {
        printf("Error when reading input parameters: read_parameters_info_flag = %d\n", read_parameters_info_flag);
        return 2;
    }
    printf("No error when reading parameters.\n");

    build_neighbour_list_flag1 = build_neighbour_list(frame_info, Nframes_tot, parameters_info, 1);
    if (build_neighbour_list_flag1 != 0)
    {
        printf("Error when building neighbour list: build_neighbour_list_flag1 = %d\n", build_neighbour_list_flag1);
        return 3;
    }
    printf_d("Check from main(): neighbour list number check:\n");
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
        return 4;
    }
    printf_d("Check from main(): types: \n");
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
    parameters_info->N_types_all_frame = N_types_all_frame;
    parameters_info->type_index_all_frame = type_index_all_frame;
    parameters_info->SEL_A_max = SEL_A_max;

    build_neighbour_list_flag2 = build_neighbour_list(frame_info, Nframes_tot, parameters_info, 2);
    if (build_neighbour_list_flag2 != 0)
    {
        printf("Error when building neighbour list: build_neighbour_list_flag1 = %d\n", build_neighbour_list_flag2);
        return 5;
    }
    printf_d("Check from main(): neighbour list of frame %d atom %d:\n", DEBUG_FRAME, DEBUG_ATOM);
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        printf_d("atom type %d coord %.3lf %.3lf %.3lf\n", frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].type[i], frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].coord_neighbours[i][0], frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].coord_neighbours[i][1], frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].coord_neighbours[i][2]);
    }

    sym_coord_type = 1;
    parameters_info->sym_coord_type = sym_coord_type;
    convert_coord_flag = convert_coord(frame_info, Nframes_tot, parameters_info, sym_coord_type, (void **)&sym_coord_DeePMD);
    if (convert_coord_flag != 0)
    {
        printf("Error when converting coordinates: convert_coord_flag = %d\n", convert_coord_flag);
        return 6;
    }
    printf("No errors converting coordinates\n");
    printf_d("Check from main(): sym_coord_DeePMD of frame %d atom %d:\n", DEBUG_FRAME, DEBUG_ATOM);
    printf_d("%-11s %-11s %-11s %-11s\n", "s_rij", "x_hat", "y_hat", "z_hat");
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        for (j = 0; j <= 3; j++)
        {
            int idx = i * 4 + j;
            printf_d("%+10.6lf ", sym_coord_DeePMD[DEBUG_FRAME].coord_converted[DEBUG_ATOM][idx]);
        }
        printf_d("\n");
    }

    save_to_file_flag = save_to_file(frame_info, parameters_info, (void *)sym_coord_DeePMD);
    if (save_to_file_flag != 0)
    {
        printf("Error when saving to files: save_to_file_flag = %d\n", save_to_file_flag);
        return 7;
    }

    /*Profiling main end*/
    gettimeofday(&end_main, NULL);
    t_main = (end_main.tv_usec - start_main.tv_usec) / 1000.0 + (end_main.tv_sec - start_main.tv_sec) * 1000;
    printf("Time profiling: main(): %.3lf s\n", t_main / 1000.0);
    return 0;
}