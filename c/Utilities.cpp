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

double d_f_c_d_r(double r_ij, double r_c)
{
    double result;
    double rc = r_c;
    result = (r_ij <= rc) ? (-3.0 / cosh(1 - r_ij / rc) / cosh(1 - r_ij / rc) * tanh(1 - r_ij / rc) * tanh(1 - r_ij / rc) / 2.0 / rc ) : (0);
    return result;
}

double R_sup_n(double r_ij, double n, double r_c)
{
    double f_c(double r_ij, double r_c);

    double result;
    return fastpown(r_ij, (int)n) * f_c(r_ij, r_c);
}

double d_R_sup_n_d_r(double r_ij, double n, double r_c)
{
    double f_c(double r_ij, double r_c);
    double d_f_c_d_r(double r_ij, double r_c);
    double result;

    result = n * fastpown(r_ij, (int)(n - 1)) * f_c(r_ij, r_c) + fastpown(r_ij, (int)n) * d_f_c_d_r(r_ij, r_c);
    return result;
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
    return result;
}
int d_cos_bond_angle_d_coord(double * coord_i, double * coord_j, double * coord_k, double * result)
{
    double fastpown(double number, int power);

    double coord_ji[3];// In DeePMD type sym_coord, r_ji means r_(j - i). Here we keep the same.
    double coord_ki[3];
    double norm_ji = 0;
    double norm_ki = 0;//norm = (x^2 + y^2 + z^2)
    double dot_jiki = 0;
    //double result = 0;
    int i, j, k;
    //norm_ji =0; norm_ki = 0;
    for (i = 0; i <= 2; i++)
    {
        coord_ji[i] = coord_j[i] - coord_i[i];
        coord_ki[i] = coord_k[i] - coord_i[i];
        norm_ji += (coord_ji[i] * coord_ji[i]);
        norm_ki += (coord_ki[i] * coord_ki[i]);
        dot_jiki += (coord_ji[i] * coord_ki[i]);
    }
    double dist_ji = sqrt(norm_ji);
    double dist_ki = sqrt(norm_ki);
    /*dxi, yi, zi, xj, yj, zj, xk, yk, zk*/
    result[0] = (- coord_ji[0] - coord_ki[0]) / (dist_ji * dist_ki) - (dot_jiki * (- 2 * coord_ki[0] * norm_ji - 2 * coord_ji[0] * norm_ki)) / ( 2 * fastpown((dist_ji * dist_ki), 3));
    result[1] = (- coord_ji[1] - coord_ki[1]) / (dist_ji * dist_ki) - (dot_jiki * (- 2 * coord_ki[1] * norm_ji - 2 * coord_ji[1] * norm_ki)) / ( 2 * fastpown((dist_ji * dist_ki), 3));
    result[2] = (- coord_ji[2] - coord_ki[2]) / (dist_ji * dist_ki) - (dot_jiki * (- 2 * coord_ki[2] * norm_ji - 2 * coord_ji[2] * norm_ki)) / ( 2 * fastpown((dist_ji * dist_ki), 3));
    result[3] = coord_ki[0] / (dist_ji * dist_ki) - coord_ji[0] * norm_ki * dot_jiki / fastpown((dist_ji * dist_ki), 3);
    result[4] = coord_ki[1] / (dist_ji * dist_ki) - coord_ji[1] * norm_ki * dot_jiki / fastpown((dist_ji * dist_ki), 3);
    result[5] = coord_ki[2] / (dist_ji * dist_ki) - coord_ji[1] * norm_ki * dot_jiki / fastpown((dist_ji * dist_ki), 3);
    result[6] = coord_ji[0] / (dist_ji * dist_ki) - coord_ki[0] * norm_ki * dot_jiki / fastpown((dist_ji * dist_ki), 3);
    result[7] = coord_ji[1] / (dist_ji * dist_ki) - coord_ki[1] * norm_ki * dot_jiki / fastpown((dist_ji * dist_ki), 3);
    result[8] = coord_ji[2] / (dist_ji * dist_ki) - coord_ki[1] * norm_ki * dot_jiki / fastpown((dist_ji * dist_ki), 3);
    
    return 0;
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
    if (norm_norm_vec_kij * norm_norm_vec_ijl == 0)//three points are on a line; no plane is formed.
    {
        return 999;
    }
    result = dot_kij_ijl / sqrt(norm_norm_vec_kij * norm_norm_vec_ijl) * ((double)-1.0);
    return result;
}

int d_cos_dihedral_angle_d_coord(double * coord_i, double * coord_j, double * coord_k, double * coord_l, double * result)//centered at i and j, plane_ijk and plane_ijl
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
    double xi, yi, zi, xj, yj, zj, xk, yk, zk, xl, yl, zl;
    xi = coord_i[0]; yi = coord_i[1]; zi=coord_i[2];
    xj = coord_j[0]; yj = coord_j[1]; zj=coord_j[2];
    xk = coord_k[0]; yk = coord_k[1]; zk=coord_k[2];
    xl = coord_l[0]; yl = coord_l[1]; zl=coord_l[2];
    //double result;
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
    /*dxi, yi, zi, xj, yj, zj, xk, yk, zk, xl, yl, zl*/
    result[0] = (-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-yj+yl))-(yj-yk)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*(zj-zl)-(-zj+zk)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))-((-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(-yj+yl)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))+2*(zj-zl)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl)))+(2*(yj-yk)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))+2*(-zj+zk)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5));
    result[1] = (-((xj-xl)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk)))-(-xj+xk)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-zj+zl)-(zj-zk)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))-((-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(xj-xl)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))+2*(-zj+zl)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))+(2*(-xj+xk)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))+2*(zj-zk)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5));
    result[2] = (-((-xj+xl)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk)))-(yj-yl)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))-(xj-xk)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-yj+yk)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))-((-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(-xj+xl)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))+2*(yj-yl)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))+(2*(xj-xk)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))+2*(-yj+yk)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5));
    result[3] = (-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(yi-yl))-(-yi+yk)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*(-zi+zl)-(zi-zk)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))-((-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(yi-yl)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))+2*(-zi+zl)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl)))+(2*(-yi+yk)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))+2*(zi-zk)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5));
    result[4] = (-((-xi+xl)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk)))-(xi-xk)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(zi-zl)-(-zi+zk)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))-((-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(-xi+xl)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))+2*(zi-zl)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))+(2*(xi-xk)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))+2*(-zi+zk)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5));
    result[5] = (-((xi-xl)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk)))-(-yi+yl)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))-(-xi+xk)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(yi-yk)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))-((-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(xi-xl)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))+2*(-yi+yl)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))+(2*(-xi+xk)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))+2*(yi-yk)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5));
    result[6] = -((2*(yi-yj)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))+2*(-zi+zj)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk)))*(-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5))+(-((yi-yj)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-(-zi+zj)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)));
    result[7] = -((2*(-xi+xj)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))+2*(zi-zj)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))*(-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5))+(-((-xi+xj)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-(zi-zj)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)));
    result[8] = -((2*(xi-xj)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))+2*(-yi+yj)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))*(-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5))+(-((xi-xj)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl)))-(-yi+yj)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)));
    result[9] = -((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(-yi+yj)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))+2*(zi-zj)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl)))*(-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5))+(-((-yi+yj)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk)))-(zi-zj)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)));
    result[10] = -((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(xi-xj)*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl))+2*(-zi+zj)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*(-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5))+(-((xi-xj)*(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk)))-(-zi+zj)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)));
    result[11] = -((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(2*(-xi+xj)*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))+2*(yi-yj)*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl)))*(-((-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk))*(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl)))-((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk))*((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl))-(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk))*(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl))))/(2.*pow((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)),1.5))+(-((-xi+xj)*((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk)))-(yi-yj)*(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk)))/sqrt((pow(-((-xi+xk)*(-yi+yj))+(-xi+xj)*(-yi+yk),2)+pow((-xi+xk)*(-zi+zj)+(xi-xj)*(-zi+zk),2)+pow(-((-yi+yk)*(-zi+zj))+(-yi+yj)*(-zi+zk),2))*(pow(-((-xj+xl)*(yi-yj))+(xi-xj)*(-yj+yl),2)+pow((-xj+xl)*(zi-zj)+(-xi+xj)*(-zj+zl),2)+pow(-((-yj+yl)*(zi-zj))+(yi-yj)*(-zj+zl),2)));
    //result = dot_kij_ijl / sqrt(norm_norm_vec_kij * norm_norm_vec_ijl) * ((double)-1.0);
    return 0;
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
    /*std::vector<int> current_type_ (current_type, current_type + N_neighb_atom);
    std::vector<int> params_type_ (params_type, params_type + N_neighb_atom);*/
    for (i = 0; i <= N_neighb_atom - 1; i++)
    {
        if (current_type[i] == -1)
        {
            return 0;
        }
    }
    if (params_type[0] == -1)
    {
        return 1;
    }
    /*std::sort(current_type_.begin(), current_type_.end());
    std::sort(params_type_.begin(), params_type_.end());
    for (i = 0; i <= N_neighb_atom - 1; i++)
    {
        sum += ((current_type_[i] - params_type_[i]) * (current_type_[i] - params_type_[i]));
    }
    return ( sum == 0 ? 1 : 0);*/
    switch (N_neighb_atom)
    {
        case 1://two body
        {
            if (current_type[0] == params_type[0])
            {
                return 1;
            }
            break;
        }
        case 2://three body
        {
            if ((((current_type[0] == params_type[0])&&(current_type[1] == params_type[1]))) || (((current_type[0] == params_type[1])&&(current_type[1] == params_type[0]))))
            {
                return 1;
            }
            break;
        }
        case 3://four body
        {
            int mul1 = current_type[0] * current_type[1] * current_type[2];
            int mul2 = params_type[0] * params_type[1] * params_type[2];
            int sum1 = current_type[0] + current_type[1] + current_type[2];
            int sum2 = params_type[0] + params_type[1] + params_type[2];
            if ((mul1 == mul2)&&(sum1 == sum2))
            {
                return 1;
            }
            break;
        }
    }
    return 0;
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
            int i, j, k;
            sym_coord_LASP_struct * sym_coord_LASP = (sym_coord_LASP_struct *)sym_coord_;
            for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
            {
                //free(sym_coord_DeePMD[i].type);
                for (j = 0; j <= sym_coord_LASP[i].N_Atoms - 1; j++)
                {
                    free(sym_coord_LASP[i].coord_converted[j]);
                    for (k = 0; k <= parameters_info->N_sym_coord - 1; k++)
                    {
                        free(sym_coord_LASP[i].idx_nei[j][k]);
                        free(sym_coord_LASP[i].d_x[j][k]);
                        free(sym_coord_LASP[i].d_y[j][k]);
                        free(sym_coord_LASP[i].d_z[j][k]);
                    }
                    free(sym_coord_LASP[i].idx_nei[j]);
                    free(sym_coord_LASP[i].d_x[j]);
                    free(sym_coord_LASP[i].d_y[j]);
                    free(sym_coord_LASP[i].d_z[j]);
                    /*Not completed
                    free(sym_coord_LASP[i].d_to_center_x[j]);
                    free(sym_coord_LASP[i].d_to_center_y[j]);
                    free(sym_coord_LASP[i].d_to_center_z[j]);*/
                }

                free(sym_coord_LASP[i].coord_converted);
                free(sym_coord_LASP[i].idx_nei);
                free(sym_coord_LASP[i].d_x);
                free(sym_coord_LASP[i].d_y);
                free(sym_coord_LASP[i].d_z);
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

int cart_to_frac(double * cart, double box[3][3], double * frac)
{
    double a1 = box[0][0], a2 = box[0][1], a3 = box[0][2], b1 = box[1][0], b2 = box[1][1], b3 = box[1][2], c1 = box[2][0], c2 = box[2][1], c3 = box[2][2];
    double denominator = (a1 * b2 * c3 + b1 * c2 * a3 + c1 * a2 * b3 - c1 * b2 * a3 - b1 * a2 * c2 - a1 * c2 * b3);
    if (denominator == 0)
    {
        return 1;//Two or more box vectors are parallel.
    }
    /*Don't forget the tranpose in the reverse-matrix formula!*/
    double prefact = 1.0 / denominator;
    double rev_a1 = prefact * (b2 * c3 - c2 * b3);
    double rev_b1 = prefact * (c1 * b3 - b1 * c3);
    double rev_c1 = prefact * (b1 * c2 - c1 * b2);
    double rev_a2 = prefact * (c2 * a3 - a2 * c3);
    double rev_b2 = prefact * (a1 * c3 - c1 * a3);
    double rev_c2 = prefact * (c1 * a2 - a1 * c2);
    double rev_a3 = prefact * (a2 * b3 - b2 * a3);
    double rev_b3 = prefact * (b1 * a3 - a1 * b3);
    double rev_c3 = prefact * (a1 * b2 - b1 * a2);
    double x = cart[0], y = cart[1], z = cart[2];
    *frac = rev_a1 * x + rev_b1 * y + rev_c1 * z;
    *(frac + 1) = rev_a2 * x + rev_b2 * y + rev_c2 * z;
    *(frac + 2) = rev_a3 * x + rev_b3 * y + rev_c3 * z;
    return 0;
}
int frac_to_cart(double * cart, double box[3][3], double * frac)
{
    double a1 = box[0][0], a2 = box[0][1], a3 = box[0][2], b1 = box[1][0], b2 = box[1][1], b3 = box[1][2], c1 = box[2][0], c2 = box[2][1], c3 = box[2][2];
    double x_ = frac[0], y_ = frac[1], z_ = frac[2];
    *cart = a1 * x_ + b1 * y_ + c1 * z_;
    *(cart + 1) = a2 * x_ + b2 * y_ + c2 * z_;
    *(cart + 2) = a3 * x_ + b3 * y_ + c3 * z_;
    return 0;
}



