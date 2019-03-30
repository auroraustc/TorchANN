/*
2019.03.29 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Utilities functions.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_UTIL

#ifdef DEBUG_UTIL
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

#define PI 3.141592653589793238462643383279

double s_r(double r_ij, parameters_info_struct * parameters_info)
{
    double result;
    double rc = parameters_info->cutoff_2;
    double rcs = parameters_info->cutoff_1;
    result = (r_ij >= rc) ? 0 : ((r_ij >= rcs) ? 1 / r_ij * (0.5 * cos((r_ij - rcs) / (rc - rcs) * PI) + 0.5) : 1 / r_ij);
    return result;
}