/*
2019.03.29 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Utilities functions.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
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

double fastpow2(double number, int dummy)
{
    return number * number;
}

double f_c(double r_ij, parameters_info_struct * parameters_info)
{
    double result;
    double rc = parameters_info->cutoff_max;
    result = (r_ij <= rc) ? 0.5 * tanh(1 - r_ij / rc) * tanh(1 - r_ij / rc) * tanh(1 - r_ij / rc) : 0;
    return result;
}

double R_sup_n(double r_ij, double n, parameters_info_struct * parameters_info)
{
    double f_c(double r_ij, parameters_info_struct * parameters_info_);

    double result;
    return pow(r_ij, n) * f_c(r_ij, parameters_info);
}

double factorial(int n)
{
    if (n < 0)
    {
        printf("From factorial: n must be positive!\n");
        exit(999);
    }
    return ((n == 0)||(n == 1)) ? 1 : factorial(n - 1) * n;
}

double P_LM(double x, int l, int m)
{
    double factorial(int n);

    if (m < 0)
    {
        return pow(-1, -m) * factorial(l + m) / factorial(l - m) * P_LM(x, l, (-1) * m);
    }

    switch (l)
    {
        case 0:
        {
            return 1.0;
        }
        case 1:
        {
            switch (m)
            {
                case 0:
                {
                    return x;
                }
                case 1:
                {
                    return -1 * sqrt(1 - x * x);
                }
            }
        }
        case 2:
        {
            switch (m)
            {
                case 0:
                {
                    return 0.5 * (3 * x * x - 1);
                }
                case 1:
                {
                    return -3 * x * sqrt(1 - x * x);
                }
                case 2:
                {
                    return 3 * (1 - x * x);
                }
            }
        }
        case 3:
        {
            switch (m)
            {
                case 0:
                {
                    return 0.5 * x * (5 * x * x - 3);
                }
                case 1:
                {
                    return 1.5 * (1 - 5 * x * x) * sqrt(1 - x * x);
                }
                case 2:
                {
                    return 15 * x * (1 - x * x);
                }
                case 3:
                {
                    return -15 * sqrt(1 - x * x) * sqrt(1 - x * x) * sqrt(1 - x * x);
                }
            }
        }
        case 4:
        {
            switch (m)
            {
                case 0:
                {
                    return 0.125 * (35 * x * x * x * x - 30 * x * x + 3);
                }
                case 1:
                {
                    return 2.5 * x * (3 - 7 * x * x) * sqrt(1 - x * x); 
                }
                case 2:
                {
                    return 7.5 * (7 * x * x - 1) * (1 - x * x);
                }
                case 3:
                {
                    return -105 * x * sqrt(1 - x * x) * sqrt(1 - x * x) * sqrt(1 - x * x);
                }
                case 4:
                {
                    return 105 * (1 - x * x) * (1 - x * x);
                }
            }
        }
        case 5:
        {
            switch (m)
            {
                case 0:
                {
                    return 0.125 * x * (63 * x * x * x * x - 70 * x * x + 15);
                }
            }
        }
        default:
        {
            printf("From P_LM: Invalid value of l and m!\n");
            exit(999);
        }
    }
}

double complex Y_LM(double * r_ij, int l, int m)//here r_ij is a vector
{
    double P_LM(double cos_theta, int l, int m);
    double factorial(int n);

    double complex result;
    double theta, phi;
    double r = sqrt(r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]);
    theta = acos(r_ij[2] / r);
    phi = atan(r_ij[1] / r_ij[0]);
    result = sqrt((2 * l + 1) / (4 * PI) * factorial(l - m) / factorial(l + m)) * P_LM(cos(theta), l, m) * cexp(m * phi * I);
    return result;
}

int calc_N_neigh_inter(int K, int N)// total N types; K body interaction; MULTIPLY N to the return value to get correct answer!!!
{
    int i;
    int result = 0;
    if (K == 1) return 1;
    if (N == 1) return 1;
    for (i = 1; i <= N; i++)
    {
        result += calc_N_neigh_inter(K - 1, i);
    }
    return result;
}

double **** calloc_params_LASP(int dim1, int dim2, int ** dim3_, int ** dim4_)
{
    int N_types_all_frame = dim1;
    int N_PTSD_types = dim2;
    int ** N_cutoff_radius = dim3_;
    int ** N_neigh_inter = dim4_;
    int i, j, k, l;
    double **** result = NULL;
    result = (double ****)calloc(dim1, sizeof(double ***));
    for (i = 0; i <= dim1 - 1; i++)
    {
        result[i] = (double ***)calloc(dim2, sizeof(double **));
        for (j = 0 ; j <= dim2 - 1; j++)
        {
            result[i][j] = (double **)calloc(N_cutoff_radius[i][j], sizeof(double *));
            for (k = 0; k <= N_cutoff_radius[i][j] - 1; k++)
            {
                result[i][j][k] = (double *)calloc(N_neigh_inter[i][j], sizeof(double));
            }
        }
    }
    return result;
}

int free_params_LASP(double **** target, int dim1, int dim2, int ** dim3_, int ** dim4_)
{
    return 1;
}