/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Return code:
    0: No errors.
    1: read_system() error.
    2: read_parameters() error.
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
    int read_system(frame_info_struct * frame_info_, int * Nframes_tot_);
    int read_parameters(parameters_info_struct * parameters_info);

    int read_system_flag;
    int read_parameters_info_flag;
    frame_info_struct * frame_info;
    parameters_info_struct * parameters_info = (parameters_info_struct *)calloc(1, sizeof(parameters_info_struct));
    int Nframes_tot;

    read_system_flag = read_system(frame_info, &Nframes_tot);
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

    return 0;
}