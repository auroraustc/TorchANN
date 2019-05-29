/*
2019.03.29 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Utilities functions.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/*#include <complex.h>*/
#include <algorithm>
#include <boost/math/special_functions/spherical_harmonic.hpp>
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

double fastpown(double number, int power)
{
    int i;
    double result = 1;
    int N = power;
    if (power == 0)
    {
        return 1;
    }
    if (power < 0)
    {
        N *= -1;
    }
    for (i = 1; i <= N; i++)
    {
        result *= number;
    }
    if (power < 0)
    {
        return 1.0 / result;
    }
    else
    {
        return result;
    }
    
}

double f_c(double r_ij, double r_c)
{
    double result;
    double rc = r_c;
    result = (r_ij <= rc) ? 0.5 * tanh(1 - r_ij / rc) * tanh(1 - r_ij / rc) * tanh(1 - r_ij / rc) : 0;
    return result;
}

double R_sup_n(double r_ij, double n, double r_c)
{
    double f_c(double r_ij, double r_c);

    double result;
    return fastpown(r_ij, (int)n) * f_c(r_ij, r_c);
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

/*double P_LM(double x, int l, int m)
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
}*/

/*double complex Y_LM(double * r_ij, int l, int m)//here r_ij is a vector
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
}*/

double Y_LM_r(double * coord_ij, int L, int m)//here L is n of the boost::math::spherical_harmonics
{
    double theta, phi;
    double r = sqrt(coord_ij[0] * coord_ij[0] + coord_ij[1] * coord_ij[1] + coord_ij[2] * coord_ij[2]);
    double result_r;
    theta = acos(coord_ij[2] / r);
    phi = atan(coord_ij[1] / coord_ij[0]);
    result_r = boost::math::spherical_harmonic_r(L, m, theta, phi);
    return result_r;
}
double Y_LM_i(double * coord_ij, int L, int m)//here L is n of the boost::math::spherical_harmonics
{
    double theta, phi;
    double r = sqrt(coord_ij[0] * coord_ij[0] + coord_ij[1] * coord_ij[1] + coord_ij[2] * coord_ij[2]);
    double result_i;
    theta = acos(coord_ij[2] / r);
    phi = atan(coord_ij[1] / coord_ij[0]);
    result_i = boost::math::spherical_harmonic_i(L, m, theta, phi);
    return result_i;
}
std::complex<double> Y_LM(double * coord_ij, int L, int m)
{
    double theta, phi;
    double r = sqrt(coord_ij[0] * coord_ij[0] + coord_ij[1] * coord_ij[1] + coord_ij[2] * coord_ij[2]);
    std::complex<double> result;
    theta = acos(coord_ij[2] / r);
    phi = atan(coord_ij[1] / coord_ij[0]);
    if (phi < 0)
    {
        phi += (2 * PI);
    }
    //printf_d("L, m, theta, phi: %d, %d, %.6lf, %.6lf\n", L, m, theta, phi);
    result = boost::math::spherical_harmonic(L, m, theta, phi);
    return result;
}
std::complex<double> d_Y_LM_d_theta(double * coord_ij, int L, int m)
{
    double theta, phi;
    double r = sqrt(coord_ij[0] * coord_ij[0] + coord_ij[1] * coord_ij[1] + coord_ij[2] * coord_ij[2]);
    std::complex<double> result;
    std::complex<double> I(0.0, 1.0);
    theta = acos(coord_ij[2] / r);
    phi = atan(coord_ij[1] / coord_ij[0]);
    if (phi < 0)
    {
        phi += (2 * PI);
    }
    /*\partial Y/\partial \theta = m * cot(\theta) * Y_LM(\theta, \phi) + \sqrt((L-m) * (L + m + 1)) * exp(- I * \phi) * Y_L(M+1)(\theta, \phi)*/
    result = m * tan((double)PI / 2.0 - theta) * boost::math::spherical_harmonic(L, m, theta, phi) + sqrt((L-m) * (L + m + 1)) * exp(-I * phi) * boost::math::spherical_harmonic(L, m + 1, theta, phi);
    return result;
}
std::complex<double> d_Y_LM_d_phi(double * coord_ij, int L, int m)
{
    double theta, phi;
    double r = sqrt(coord_ij[0] * coord_ij[0] + coord_ij[1] * coord_ij[1] + coord_ij[2] * coord_ij[2]);
    std::complex<double> result;
    std::complex<double> I(0.0, 1.0);
    theta = acos(coord_ij[2] / r);
    phi = atan(coord_ij[1] / coord_ij[0]);
    if (phi < 0)
    {
        phi += (2 * PI);
    }
    /*\partial Y/\partial \phi = I * m * Y_LM(\theta, \phi)*/
    result = I * (double)m * boost::math::spherical_harmonic(L, m, theta, phi);
    return result;
}

double cos_bond_angle(double * coord_i, double * coord_j, double * coord_k)//centered at i
{
    double coord_ji[3];// In DeePMD type sym_coord, r_ji means r_(j - i). Here we keep the same.
    double coord_ki[3];
    double norm_ji, norm_ki;//norm = (x^2 + y^2 + z^2)
    double dot_jiki = 0;
    double result = 0;
    int i, j, k;
    norm_ji =0; norm_ki = 0;
    for (i = 0; i <= 2; i++)
    {
        coord_ji[i] = coord_j[i] - coord_i[i];
        coord_ki[i] = coord_k[i] - coord_i[i];
        norm_ji += (coord_ji[i] * coord_ji[i]);
        norm_ki += (coord_ki[i] * coord_ki[i]);
        dot_jiki += (coord_ji[i] * coord_ki[i]);
    }
    /*cos(\theta_ijk_centered_i) = dot_prod(coord_ji, coord_ki) / \sqrt(norm_ji * norm_ki)*/
    result = dot_jiki / sqrt(norm_ji * norm_ki);
}
int cross_prod(double * vec1, double * vec2, double * vec_result)
{
    vec_result[0] = (vec1[1] * vec2[2] - vec1[2] * vec2[1]);
    vec_result[1] = (vec1[2] * vec2[0] - vec1[0] * vec2[2]);
    vec_result[2] = (vec1[0] * vec2[1] - vec1[1] * vec2[0]);
    return 0;
}
double cos_dihedral_angle(double * coord_i, double * coord_j, double * coord_k, double * coord_l)//centered at i and j, plane_ijk and plane_ijl
{
    int cross_prod(double * vec1, double * vec2, double * vec_result);

    /*Use coord_ki, ji to calculate norm_vec of plane_kij = ji \times ki; Use coord_ij, lj to calculate norm_vec of plane_ijl = ij \times lj*/
    double coord_ki[3];// In DeePMD type sym_coord, r_ji means r_(j - i). Here we keep the same.
    double coord_ji[3];
    double coord_ij[3];
    double coord_lj[3];
    double norm_vec_kij[3];
    double norm_vec_ijl[3];
    double norm_norm_vec_kij = 0;//norm = (x^2 + y^2 + z^2)
    double norm_norm_vec_ijl = 0;
    double dot_kij_ijl = 0;
    double result;
    int error_code;
    int i, j, k;
    for (i = 0; i <= 2; i++)
    {
        coord_ki[i] = coord_k[i] - coord_i[i];
        coord_ji[i] = coord_j[i] - coord_i[i];
        coord_ij[i] = coord_i[i] - coord_j[i];
        coord_lj[i] = coord_l[i] - coord_j[i];
    }
    error_code = cross_prod(coord_ji, coord_ki, norm_vec_kij);
    error_code = cross_prod(coord_ij, coord_lj, norm_vec_ijl);
    for (i = 0 ; i <= 2; i++)
    {
        norm_norm_vec_kij += (norm_vec_kij[i] * norm_vec_kij[i]);
        norm_norm_vec_ijl += (norm_vec_ijl[i] * norm_vec_ijl[i]);
        dot_kij_ijl += norm_vec_kij[i] * norm_vec_ijl[i];
    }
    result = dot_kij_ijl / sqrt(norm_norm_vec_kij * norm_norm_vec_ijl) * ((double)-1.0);
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

int compare_Nei_type(int N_neighb_atom, int * current_type, int * params_type)//For example, current_type = {1, 8, 8}, params_type = {8, 1, 8}, then return 1
{
    int i, j, k;
    int sum = 0;
    std::vector<int> current_type_ (current_type, current_type + N_neighb_atom);
    std::vector<int> params_type_ (params_type, params_type + N_neighb_atom);
    if (params_type[0] == -1)
    {
        return 1;
    }
    std::sort(current_type_.begin(), current_type_.end());
    std::sort(params_type_.begin(), params_type_.end());
    for (i = 0; i <= N_neighb_atom - 1; i++)
    {
        sum += ((current_type_[i] - params_type_[i]) * (current_type_[i] - params_type_[i]));
    }
    return ((sum == 0) ? 1 : 0);
}

int find_index_int(int target, int * array, int array_length)
{
    std::vector<int> array_ (array, array + array_length);
    std::vector<int>::iterator it = std::find(array_.begin(), array_.end(), target);
    int index = std::distance(array_.begin(), it);
    return index;
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

int free_sym_coord(void * sym_coord_, int sym_coord_type, parameters_info_struct * parameters_info)
{
    switch (sym_coord_type)
    {
        case 1:
        {
            int i, j;
            sym_coord_DeePMD_struct * sym_coord_DeePMD = (sym_coord_DeePMD_struct *)sym_coord_;
            for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
            {
                //free(sym_coord_DeePMD[i].type);
                for (j = 0; j <= sym_coord_DeePMD[i].N_Atoms - 1; j++)
                {
                    free(sym_coord_DeePMD[i].coord_converted[j]);
                    free(sym_coord_DeePMD[i].d_to_center_x[j]);
                    free(sym_coord_DeePMD[i].d_to_center_y[j]);
                    free(sym_coord_DeePMD[i].d_to_center_z[j]);
                }

                free(sym_coord_DeePMD[i].coord_converted);
                free(sym_coord_DeePMD[i].d_to_center_x);
                free(sym_coord_DeePMD[i].d_to_center_y);
                free(sym_coord_DeePMD[i].d_to_center_z);
            }
            free(sym_coord_DeePMD);
            return 0;
            break;
        }
        case 2:
        {
            int i, j;
            sym_coord_LASP_struct * sym_coord_LASP = (sym_coord_LASP_struct *)sym_coord_;
            for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
            {
                //free(sym_coord_DeePMD[i].type);
                for (j = 0; j <= sym_coord_LASP[i].N_Atoms - 1; j++)
                {
                    free(sym_coord_LASP[i].coord_converted[j]);
                    /*Not completed
                    free(sym_coord_LASP[i].d_to_center_x[j]);
                    free(sym_coord_LASP[i].d_to_center_y[j]);
                    free(sym_coord_LASP[i].d_to_center_z[j]);*/
                }

                free(sym_coord_LASP[i].coord_converted);
                /*Not completed
                free(sym_coord_LASP[i].d_to_center_x);
                free(sym_coord_LASP[i].d_to_center_y);
                free(sym_coord_LASP[i].d_to_center_z);*/
            }
            free(sym_coord_LASP);
            return 0;
            break;
        }
        default:
        {
            break;
        }
    }
    return 0;

}