/*
2019.03.29 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Count N_types for each frame and all the frames.

[Y] = set in this module, [N] = not set in this module:
typedef struct frame_info_struct_
{
[N]	int index;
[N]	int N_Atoms;
[Y] int N_types;
[N]	double box[3][3];
[N]	int * type;//type[0..N_Atoms-1]
[N]	double ** coord;//coord[0..N_Atoms-1][0..2]
[N]	double energy;
[N]	int no_force;
[N]	double ** force;//force[0..N_Atoms-1][0..2]
[N]	neighbour_list_struct * neighbour_list;//neighbour_list[0..N_Atoms-1], neighbour list for each atom
}

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
//#define DEBUG_COUNT

#ifdef DEBUG_COUNT
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int count_types(frame_info_struct * frame_info, int Nframes_tot, int * N_types_all_frame_, int ** type_index_all_frame_)
{
    const int MAX_NUM_ELEMENTS = 172;//the number of elements will not exceed 172 at 2019
    int i, j, k;
    int type_all_frame[MAX_NUM_ELEMENTS];//type_all_frame[i] = 0: do not have this element; 1: have this element.
    int N_types_all_frame = 0;
    int * type_index_all_frame;

    for (i = 0; i <= MAX_NUM_ELEMENTS - 1; i++)
    {
        type_all_frame[i] = 0;
    }

    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        int * type_cur_frame = (int *)calloc(MAX_NUM_ELEMENTS, sizeof(int));
        int N_types_cur_frame = 0;
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            type_cur_frame[frame_info[i].type[j]] = 1;
            type_all_frame[frame_info[i].type[j]] = 1;
        }
        for (j = 0; j <= MAX_NUM_ELEMENTS - 1; j++)
        {
            if (type_cur_frame[j] != 0) N_types_cur_frame++;
        }
        frame_info[i].N_types = N_types_cur_frame;
        printf_d("N_types of frame %d is %d\n", i + 1, frame_info[i].N_types);
        free(type_cur_frame);
    }
    
    for (i = 0; i <= MAX_NUM_ELEMENTS - 1; i++)
    {
        (type_all_frame[i] != 0) ? N_types_all_frame ++ : 0;
    }
    printf_d("N_types_all_frame is %d\n", N_types_all_frame);
    * N_types_all_frame_ = N_types_all_frame;

    type_index_all_frame = (int *)calloc(N_types_all_frame, sizeof(int));
    j = 0;
    for (i = 0; i <= MAX_NUM_ELEMENTS - 1; i++)
    {
        if (type_all_frame[i] != 0)
        {
            type_index_all_frame[j] = i;
            j++;
        }
        //type_index_all_frame[j] = ((type_all_frame[i] != 0) ? (j++, i) : type_index_all_frame[j]);
    }
    * type_index_all_frame_ = type_index_all_frame;

    return 0;
}