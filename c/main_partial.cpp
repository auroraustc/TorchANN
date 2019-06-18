/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Return code:
    0: No errors.
    1: read_system() error.
    2: read_parameters() error.
    3: extend_max_atoms() error.
    4: build_neighbour_list() error at step1.
    5: count types() error.
    6: build_neighbour_list() error at step2.
    7: convert_coord() error.
    8: save_to_file() error.

*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
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
    int read_parameters(frame_info_struct * frame_info, parameters_info_struct * parameters_info, char * filename);
    int extend_max_atoms(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    int build_neighbour_list(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, int step);
    int count_types(frame_info_struct * frame_info, int Nframes_tot, int * N_types_all_frame_, int ** type_index_all_frame_);
    /*int convert_coord(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, int coord_type, void ** sym_coord_struct);*/
    int save_to_file_partial(frame_info_struct * frame_info, parameters_info_struct * parameters_info, void * sym_coord);
    /*int free_sym_coord(void * sym_coord_, int sym_coord_type, parameters_info_struct * parameters_info);*/
    
    struct timeval start_main, end_main;
    double t_main;//Unit: ms

    int error_code = 1;
    int read_system_flag;
    int read_parameters_info_flag;
    int extend_max_atoms_flag;
    int build_neighbour_list_flag1, build_neighbour_list_flag2;
    int count_types_flag;
    int convert_coord_flag;
    int save_to_file_flag;
    frame_info_struct * frame_info = NULL;
    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));
    sym_coord_DeePMD_struct * sym_coord_DeePMD = NULL;
    sym_coord_LASP_struct * sym_coord_LASP = NULL;
    int Nframes_tot;
    int max_N_neighbours_all_frame = -1;
    int SEL_A_max;
    int N_types_all_frame = 0;
    int * type_index_all_frame = NULL;//type_index_all_frame[0..N_types_all_frame - 1]. For exampe, we have 3 elements, 6(C), 29(Cu), 1(H), then ...[0]=6, ...[1]=29, ...[2]=1.
    int sym_coord_type = 1;
    int i, j, k;

    /*Profiling main start*/
    gettimeofday(&start_main, NULL);

	srand(time(NULL));

    read_system_flag = read_system(&frame_info, &Nframes_tot);
    if (read_system_flag != 0) 
    {
        printf("Error when reading raw data: read_flag = %d\n", read_system_flag);
        return error_code;
    }
    printf("No error when reading raw data.\n");
    parameters_info->Nframes_tot = Nframes_tot;
    error_code ++;

    read_parameters_info_flag = read_parameters(frame_info, parameters_info, "PARAMS.json");
    if (read_parameters_info_flag != 0)
    {
        printf("Error when reading input parameters: read_parameters_info_flag = %d\n", read_parameters_info_flag);
        return error_code;
    }
    printf("No error when reading parameters.\n");
    error_code ++;

    /*Make the number of atoms in each frame aligned to parameters_info->N_Atoms_max by adding dummy atoms*/
    extend_max_atoms_flag = extend_max_atoms(frame_info, parameters_info);
    if (extend_max_atoms_flag != 0)
    {
        printf("Error when extending each frame: extend_max_atoms_flag = %d\n", extend_max_atoms_flag);
        return error_code;
    }
    printf("No error when extending each frame.\n");
    error_code ++;

    build_neighbour_list_flag1 = build_neighbour_list(frame_info, Nframes_tot, parameters_info, 1);
    if (build_neighbour_list_flag1 != 0)
    {
        printf("Error when building neighbour list: build_neighbour_list_flag1 = %d\n", build_neighbour_list_flag1);
        return error_code;
    }
    printf("No error when building neighbour list flag1.\n");
    error_code ++;
    printf_d("Check from main(): neighbour list number check:\n");
    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        if (frame_info[i].max_N_neighbours >= max_N_neighbours_all_frame) max_N_neighbours_all_frame = frame_info[i].max_N_neighbours;
        printf_d("max neighbour atoms of frame %d: %d\n", i + 1, frame_info[i].max_N_neighbours);
        printf_d("In this frame the number of neighbour atoms of 2nd atom is %d\n", frame_info[i].neighbour_list[1].N_neighbours);
    }
    SEL_A_max = 50 * (max_N_neighbours_all_frame / 50 + 1);
    printf("Max number of N_neighbour is %d. SEL_A would be %d\n", max_N_neighbours_all_frame, SEL_A_max);
    

    count_types_flag = count_types(frame_info, Nframes_tot, &N_types_all_frame, &type_index_all_frame);
    if (count_types_flag != 0)
    {
        printf("Error when counting types: count_types_flag = %d\n", count_types_flag);
        return error_code;
    }
    printf_d("No error when counting types.\n");
    error_code ++;
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
    if (parameters_info->type_index_all_frame != NULL)
    {
        free(parameters_info->type_index_all_frame);
    }
    parameters_info->type_index_all_frame = type_index_all_frame;
    parameters_info->SEL_A_max = SEL_A_max;


    build_neighbour_list_flag2 = build_neighbour_list(frame_info, Nframes_tot, parameters_info, 2);
    if (build_neighbour_list_flag2 != 0)
    {
        printf("Error when building neighbour list: build_neighbour_list_flag2 = %d\n", build_neighbour_list_flag2);
        return error_code;
    }
    printf_d("No error when building neighbour list flag2.\n");
    error_code ++;
    printf_d("Check from main(): neighbour list of frame %d atom %d:\n", DEBUG_FRAME, DEBUG_ATOM);
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        printf_d("atom type %d coord %.3lf %.3lf %.3lf\n", frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].type[i], frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].coord_neighbours[i][0], frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].coord_neighbours[i][1], frame_info[DEBUG_FRAME].neighbour_list[DEBUG_ATOM].coord_neighbours[i][2]);
    }

    /*sym_coord_type = parameters_info->sym_coord_type;
    switch (sym_coord_type)
    {
        case 1:
        {
            convert_coord_flag = convert_coord(frame_info, Nframes_tot, parameters_info, sym_coord_type, (void **)&sym_coord_DeePMD);
            break;
        }
        case 2:
        {
            convert_coord_flag = convert_coord(frame_info, Nframes_tot, parameters_info, sym_coord_type, (void **)&sym_coord_LASP);
            break;
        }
        default:
        {
            printf("Symmetry coordinate type not supported!\n");
            return 7;
            break;
        }
    }
    
    if (convert_coord_flag != 0)
    {
        printf("Error when converting coordinates: convert_coord_flag = %d\n", convert_coord_flag);
        return error_code;
    }
    printf("No errors converting coordinates.\n");
    error_code ++;
    printf_d("Check from main(): sym_coord of frame %d atom %d:\n", DEBUG_FRAME, DEBUG_ATOM);
    if (sym_coord_type == 1)
    {
        printf_d("%-11s %-11s %-11s %-11s\n", "s_rij", "x_hat", "y_hat", "z_hat");
        for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
        {
            for (j = 0; j <= 3; j++)
            {
                int idx = i * 4 + j;
                printf_d("%+18.6lf ", sym_coord_DeePMD[DEBUG_FRAME].coord_converted[DEBUG_ATOM][idx]);
            }
            printf_d("\n");
        }
    }
    if (sym_coord_type == 2)
    {
        for (i = 0; i <= parameters_info->N_sym_coord - 1; i++)
        {
            printf_d("%+18.6lf\n", sym_coord_LASP[DEBUG_FRAME].coord_converted[DEBUG_ATOM][i]);
        }
    }*/
    /*printf_d("%-11s %-11s %-11s %-11s\n", "ds_rijx", "dx_hatx", "dy_hatx", "dz_hatx");
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        for (j = 0; j <= 3; j++)
        {
            int idx = i * 4 + j;
            printf_d("%+10.6lf ", sym_coord_DeePMD[DEBUG_FRAME].d_to_center_x[DEBUG_ATOM][idx]);
        }
        printf_d("\n");
    }
    printf_d("%-11s %-11s %-11s %-11s\n", "ds_rijy", "dx_haty", "dy_haty", "dz_haty");
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        for (j = 0; j <= 3; j++)
        {
            int idx = i * 4 + j;
            printf_d("%+10.6lf ", sym_coord_DeePMD[DEBUG_FRAME].d_to_center_y[DEBUG_ATOM][idx]);
        }
        printf_d("\n");
    }
    printf_d("%-11s %-11s %-11s %-11s\n", "ds_rijz", "dx_hatz", "dy_hatz", "dz_hatz");
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        for (j = 0; j <= 3; j++)
        {
            int idx = i * 4 + j;
            printf_d("%+10.6lf ", sym_coord_DeePMD[DEBUG_FRAME].d_to_center_z[DEBUG_ATOM][idx]);
        }
        printf_d("\n");
    }*/
    
    save_to_file_flag = save_to_file_partial(frame_info, parameters_info, (void *)sym_coord_DeePMD);
    if (save_to_file_flag != 0)
    {
        printf("Error when saving to files: save_to_file_flag = %d\n", save_to_file_flag);
        return error_code;
    }
    printf_d("No error when saving to file.\n");
    error_code ++;

    /*Profiling main end*/
    gettimeofday(&end_main, NULL);
    t_main = (end_main.tv_usec - start_main.tv_usec) / 1000.0 + (end_main.tv_sec - start_main.tv_sec) * 1000;
    printf("Time profiling: main(): %.3lf s\n", t_main / 1000.0);

    /*free all the data*/
        /*frame_info*/
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            free(frame_info[i].coord[j]);
            free(frame_info[i].force[j]);
            for (k = 0; k <= parameters_info->SEL_A_max - 1; k++)
            {
                free(frame_info[i].neighbour_list[j].coord_neighbours[k]);
            }
            free(frame_info[i].neighbour_list[j].coord_neighbours);
            free(frame_info[i].neighbour_list[j].type);
            free(frame_info[i].neighbour_list[j].index_neighbours);
            free(frame_info[i].neighbour_list[j].dist_neighbours);
        }
        free(frame_info[i].coord);
        free(frame_info[i].force);
        free(frame_info[i].neighbour_list);
        free(frame_info[i].type);
    }
    free(frame_info);
        /*sym_coord*/
    /*void * sym_coord_;
    switch (sym_coord_type)
    {
        case 1:
        {
            sym_coord_ = (void *)sym_coord_DeePMD;
            break;
        }
        case 2:
        {
            sym_coord_ = (void *)sym_coord_LASP;
            break;
        }
    }
    free_sym_coord(sym_coord_, sym_coord_type, parameters_info);*/
        /*parameters_info*/
    free(parameters_info->filter_neuron);
    free(parameters_info->fitting_neuron);
    free(parameters_info->type_index_all_frame);
    free(parameters_info);

    printf("***Modify the ALL_PARAMS.json before training!***\n");
    return 0;
}