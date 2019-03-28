/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn
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
    int read_system();
    int read_system_flag;
    frame_info_struct * frame_info;
    int Nframes_tot;

    read_system_flag = read_system(frame_info, &Nframes_tot);
    if (read_system_flag != 0) printf_d("Error when reading raw data: read_flag = %d\n", read_system_flag);

    return 0;
}