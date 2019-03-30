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

int read_parameters(parameters_info_struct * parameters_info)
{
    parameters_info->cutoff_1 = 7.7;
    parameters_info->cutoff_2 = 8.0;
    parameters_info->cutoff_3 = 0.0;
    parameters_info->cutoff_max = 8.0;
    return 0;
}