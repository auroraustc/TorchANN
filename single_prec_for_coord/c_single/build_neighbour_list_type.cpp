/*
2019.04.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Build neighbour lists for each frame. The type of neighbour atoms should be the same type with center atom 
step:
    1: do count only
    2: build coord (and force) of the neighbour list according to the parameters_info->SEL_A_max

[Y] = set in this module, [N] = not set in this module:
typedef struct frame_info_struct_
{
[N]	int index;
[N]	int N_Atoms;
[N] int N_types;
[N]	float box[3][3];
[N]	int * type;//type[0..N_Atoms-1]
[N]	float ** coord;//coord[0..N_Atoms-1][0..2]
[N]	float energy;
[N]	int no_force;
[N]	float ** force;//force[0..N_Atoms-1][0..2]
[Y]	neighbour_list_struct * neighbour_list;//neighbour_list[0..N_Atoms-1], neighbour list for each atom
}
at s1:
typedef struct neighbour_list_struct_
{
[Y]	int index;//atom index
[Y]	float cutoff_1;
[Y]	float cutoff_2;
[Y]	float cutoff_3;
[Y]	float cutoff_max;//cutoff_1 min, cutoff_max max. Four cutoffs just in case.
[Y]	int N_neighbours;//number of neighbour atoms within cutoff raduis cutoff_max
[N]	float ** coord_neighbours;//coord_neighbour[0..0..SEL_A_max][0..2]
[N]	float ** force_neighbours;//Just in case; force_neighbours[0..0..SEL_A_max][0..2]
[N]	int * type;//type[0..N_neighbours]
}

at s2:
typedef struct neighbour_list_struct_
{
[Y]	int index;//atom index
[Y]	float cutoff_1;
[Y]	float cutoff_2;
[Y]	float cutoff_3;
[Y]	float cutoff_max;//cutoff_1 min, cutoff_max max. Four cutoffs just in case.
[Y]	int N_neighbours;//number of neighbour atoms within cutoff raduis cutoff_max
[Y]	float ** coord_neighbours;//coord_neighbour[0..0..SEL_A_max - 1][0..2]
[Y]	float ** force_neighbours;//Just in case; force_neighbours[0..0..SEL_A_max - 1][0..2]
[Y]	int * type;//type[0..N_neighbours]
}

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
//#define DEBUG_BUILD

#ifdef DEBUG_BUILD
#define printf_d printf
#else
#define printf_d //
#endif


/***************MACRO FOR DEBUG END***************/

int build_neighbour_list(frame_info_struct * frame_info, int Nframes_tot, parameters_info_struct * parameters_info, int step)
{
    int build_neighbour_list_one_frame(frame_info_struct * frame_info_cur, parameters_info_struct * parameters_info, int step);

    int i;
    int error_code = 0;
    #pragma omp parallel for
    for (i = 0; i <= Nframes_tot - 1; i++)
    {
        printf_d("Debug info of frame %d:\n", i + 1);
        error_code = build_neighbour_list_one_frame(&(frame_info[i]), parameters_info, step);
    }
    
    return error_code;
}

int build_neighbour_list_one_frame(frame_info_struct * frame_info_cur, parameters_info_struct * parameters_info, int step)
{
    int expand_system_one_frame(frame_info_struct * frame_info_cur, system_info_expanded_struct * system_info_expanded, parameters_info_struct * parameters_info);
    int build_neighbour_coord_cur_atom(frame_info_struct * frame_info_cur, neighbour_list_struct * neighbour_list_cur_atom, system_info_expanded_struct * system_info_expanded, parameters_info_struct * parameters_info);
    float fastpow2(float number, int dummy);

    int i, j, k;
    int max_num_N_nei_one_frame = -1;
    int error_code = 0;
    system_info_expanded_struct * system_info_expanded = (system_info_expanded_struct *)calloc(1, sizeof(system_info_expanded_struct ));
    neighbour_list_struct * neighbour_list_cur;
    
    expand_system_one_frame(frame_info_cur, system_info_expanded, parameters_info);
    if (frame_info_cur->index == DEBUG_FRAME) printf_d("Expanded:\n");
    for (i = 0; i <= system_info_expanded->N_Atoms - 1; i++)
    {
        if (frame_info_cur->index != DEBUG_FRAME) break;
        printf_d("%c %f %f %f\n", system_info_expanded->type[i] + 65, system_info_expanded->atom_info[i].coord[0], system_info_expanded->atom_info[i].coord[1],system_info_expanded->atom_info[i].coord[2]);
    }
    if (frame_info_cur->index == DEBUG_FRAME) printf_d("Origin:\n");
    for (i = 0; i <= frame_info_cur->N_Atoms - 1; i++)
    {
        if (frame_info_cur->index != DEBUG_FRAME) break;
        printf_d("%c %f %f %f\n", frame_info_cur->type[i] + 65, frame_info_cur->coord[i][0], frame_info_cur->coord[i][1], frame_info_cur->coord[i][2]);
    }

    if (step == 2) goto s2;
s1:
    neighbour_list_cur = (neighbour_list_struct *)calloc(frame_info_cur->N_Atoms, sizeof(neighbour_list_struct));
    //#pragma omp parallel for private(j)
    for (i = 0; i <= frame_info_cur->N_Atoms - 1; i++)
    {
        int N_nei = 0;
        neighbour_list_cur[i].index = i;
        neighbour_list_cur[i].cutoff_1 = parameters_info->cutoff_1;
        neighbour_list_cur[i].cutoff_2 = parameters_info->cutoff_2;
        neighbour_list_cur[i].cutoff_3 = parameters_info->cutoff_3;
        neighbour_list_cur[i].cutoff_max = parameters_info->cutoff_max;
        for (j = 0; j <= system_info_expanded->N_Atoms - 1; j++)
        {
            float dist_ij;
            dist_ij = sqrt(fastpow2(frame_info_cur->coord[i][0] - system_info_expanded->atom_info[j].coord[0], 2) + fastpow2(frame_info_cur->coord[i][1] - system_info_expanded->atom_info[j].coord[1], 2) + fastpow2(frame_info_cur->coord[i][2] - system_info_expanded->atom_info[j].coord[2], 2));
            if ((dist_ij <= neighbour_list_cur[i].cutoff_max)&&(frame_info_cur->type[i] == system_info_expanded->type[j])) N_nei++;
        }
        neighbour_list_cur[i].N_neighbours = N_nei;
        if (frame_info_cur->type[i] == -1)
        {
            N_nei = 0;
        }
        printf_d("Number of neighbours of atom %d: %d\n", i + 1,  N_nei);
        if (N_nei >= max_num_N_nei_one_frame) max_num_N_nei_one_frame = N_nei;
    }
    frame_info_cur->max_N_neighbours = max_num_N_nei_one_frame;
    printf_d("max number of neighbour atoms in this frame: %d\n", frame_info_cur->max_N_neighbours);

    frame_info_cur->neighbour_list = neighbour_list_cur;
    free(system_info_expanded->atom_info);free(system_info_expanded->type);free(system_info_expanded);
    return error_code;

s2:
    i = 0;
    float ** dist_ij_cur_frame;
    //#pragma omp parallel for
    for (i = 0; i <= frame_info_cur->N_Atoms - 1; i++)
    {
        build_neighbour_coord_cur_atom(frame_info_cur, &(frame_info_cur->neighbour_list[i]), system_info_expanded, parameters_info);
    }
    if ((frame_info_cur->index == DEBUG_FRAME))
    {
        printf_d("Neighbour list of atom %d of frame %d:\n", DEBUG_ATOM, DEBUG_FRAME);
        for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
        {
            printf_d("atom type %d coord %.3lf %.3lf %.3lf \n", frame_info_cur->neighbour_list[DEBUG_ATOM].type[i], frame_info_cur->neighbour_list[DEBUG_ATOM].coord_neighbours[i][0], frame_info_cur->neighbour_list[DEBUG_ATOM].coord_neighbours[i][1], frame_info_cur->neighbour_list[DEBUG_ATOM].coord_neighbours[i][2]);
        }
    }

    free(system_info_expanded->atom_info);free(system_info_expanded->type);free(system_info_expanded);
    return 0;
}

int expand_system_one_frame(frame_info_struct * frame_info_cur, system_info_expanded_struct * system_info_expanded, parameters_info_struct * parameters_info)
{
    float fastpow2(float number, int dummy);
    
    int i, j, k, l;
    int tmpi1;
    float cutoff_max;
    float box_vec_a, box_vec_b, box_vec_c;
    int expand_a_period;
    int expand_b_period;
    int expand_c_period;
    int * expand_a_array;
    int * expand_b_array;
    int * expand_c_array;
    int tot_num_replica;//including self

    
    cutoff_max = parameters_info->cutoff_max;
    box_vec_a = sqrt(fastpow2(frame_info_cur->box[0][0], 2) + fastpow2(frame_info_cur->box[0][1], 2) + fastpow2(frame_info_cur->box[0][2], 2));
    box_vec_b = sqrt(fastpow2(frame_info_cur->box[1][0], 2) + fastpow2(frame_info_cur->box[1][1], 2) + fastpow2(frame_info_cur->box[1][2], 2));
    box_vec_c = sqrt(fastpow2(frame_info_cur->box[2][0], 2) + fastpow2(frame_info_cur->box[2][1], 2) + fastpow2(frame_info_cur->box[2][2], 2));

    expand_a_period = ceil(cutoff_max / box_vec_a);
    expand_b_period = ceil(cutoff_max / box_vec_b);
    expand_c_period = ceil(cutoff_max / box_vec_c);
    printf_d("expand x, y, z is: %d %d %d\n", expand_a_period, expand_b_period, expand_c_period);
    expand_a_array = (int *)calloc(expand_a_period * 2 + 1, sizeof(int));
    expand_b_array = (int *)calloc(expand_b_period * 2 + 1, sizeof(int));
    expand_c_array = (int *)calloc(expand_c_period * 2 + 1, sizeof(int));
    printf_d("expand_a_array:\n");
    for (i = 0; i <= expand_a_period * 2; i++)
    {
        expand_a_array[i] = (-1) * expand_a_period + i;
        printf_d("%d ", expand_a_array[i]);
    }
    printf_d("\n");
    printf_d("expand_b_array:\n");
    for (i = 0; i <= expand_b_period * 2; i++)
    {
        expand_b_array[i] = (-1) * expand_b_period + i;
        printf_d("%d ", expand_b_array[i]);
    }
    printf_d("\n");
    printf_d("expand_c_array:\n");
    for (i = 0; i <= expand_c_period * 2; i++)
    {
        expand_c_array[i] = (-1) * expand_c_period + i;
        printf_d("%d ", expand_c_array[i]);
    }
    printf_d("\n");

    tot_num_replica = (expand_a_period * 2 + 1) * (expand_b_period * 2 + 1) * (expand_c_period * 2 + 1);
    system_info_expanded->N_Atoms = tot_num_replica * frame_info_cur->N_Atoms;
    system_info_expanded->atom_info = (atom_info_struct *)calloc(tot_num_replica * frame_info_cur->N_Atoms, sizeof(atom_info_struct));
    system_info_expanded->type = (int *)calloc(system_info_expanded->N_Atoms, sizeof(int));

    for (i = 0; i <= (expand_a_period * 2); i++)  //box_vec_a direction
    {
        for (j = 0; j <= (expand_b_period * 2); j++)  //box_vec_b direction
        {
            for (k = 0; k <= (expand_c_period * 2); k++)  //box_vec_c direction
            {
                int offset = (i * (expand_b_period * 2 + 1) + j) * (expand_c_period * 2 + 1) + k;
                printf_d("offset check: %d\n", offset);
                for (l = 0; l <= frame_info_cur->N_Atoms - 1; l++)
                {
                    system_info_expanded->atom_info[offset * frame_info_cur->N_Atoms + l].index = l;
                    system_info_expanded->atom_info[offset * frame_info_cur->N_Atoms + l].coord[0] = frame_info_cur->coord[l][0];
                    system_info_expanded->atom_info[offset * frame_info_cur->N_Atoms + l].coord[1] = frame_info_cur->coord[l][1];
                    system_info_expanded->atom_info[offset * frame_info_cur->N_Atoms + l].coord[2] = frame_info_cur->coord[l][2];
                    system_info_expanded->atom_info[offset * frame_info_cur->N_Atoms + l].coord[0] += ((expand_a_array[i] * frame_info_cur->box[0][0]) + (expand_b_array[j] * frame_info_cur->box[1][0]) + (expand_c_array[k] * frame_info_cur->box[2][0]));
                    system_info_expanded->atom_info[offset * frame_info_cur->N_Atoms + l].coord[1] += ((expand_a_array[i] * frame_info_cur->box[0][1]) + (expand_b_array[j] * frame_info_cur->box[1][1]) + (expand_c_array[k] * frame_info_cur->box[2][1]));
                    system_info_expanded->atom_info[offset * frame_info_cur->N_Atoms + l].coord[2] += ((expand_a_array[i] * frame_info_cur->box[0][2]) + (expand_b_array[j] * frame_info_cur->box[1][2]) + (expand_c_array[k] * frame_info_cur->box[2][2]));
                    system_info_expanded->type[offset * frame_info_cur->N_Atoms + l] = frame_info_cur->type[l];
                }
            }
        }
    }

    free(expand_a_array);free(expand_b_array);free(expand_c_array);
    return 0;
}

int build_neighbour_coord_cur_atom(frame_info_struct * frame_info_cur, neighbour_list_struct * neighbour_list_cur_atom, system_info_expanded_struct * system_info_expanded, parameters_info_struct * parameters_info)
{
    void quick_sort_dist_cur_atom(dist_info_struct *** a_tmp_, dist_info_struct *** b_tmp_, int start, int end, int tot_num);
    float fastpow2(float number, int dummy);

    int i, j, k;
    int index = neighbour_list_cur_atom->index;
    //int DEBUG_INDEX = 2;
    dist_info_struct * dist_info;

    /*Calculate all distances of atoms in system_info_expanded and current atom*/
    /*Only those 1e-6 < dist <= rc are need*/
    /**dist_info = (dist_info_struct *)calloc(system_info_expanded->N_Atoms, sizeof(dist_info_struct));
    //#pragma omp parallel for
    for (i = 0; i <= system_info_expanded->N_Atoms - 1; i++)
    {
        dist_info[i].atom_info = &(system_info_expanded->atom_info[i]);
        dist_info[i].dist = sqrt(pow(frame_info_cur->coord[index][0] - dist_info[i].atom_info->coord[0] , 2) + pow(frame_info_cur->coord[index][1] - dist_info[i].atom_info->coord[1], 2) + pow(frame_info_cur->coord[index][2] - dist_info[i].atom_info->coord[2], 2));
    }
    if ((frame_info_cur->index == DEBUG_FRAME)&&(index == DEBUG_ATOM))
    {
        printf_d("distance from atom %d of frame %d:\n", DEBUG_ATOM, DEBUG_FRAME);
        for (i = 0; i <= system_info_expanded->N_Atoms - 1; i++)
        {
            printf_d("atom index %d coord %.3lf %.3lf %.3lf dist %.6lf\n", dist_info[i].atom_info->index, dist_info[i].atom_info->coord[0], dist_info[i].atom_info->coord[1], dist_info[i].atom_info->coord[2], dist_info[i].dist);
        }
    }**/
    dist_info = (dist_info_struct *)calloc(parameters_info->SEL_A_max, sizeof(dist_info_struct));
    //#pragma omp parallel for
    int index_tmp_dist = 0;
    dist_info_struct dummy;
    atom_info_struct dummy_atom;
    dummy_atom.coord[0] = 9999; dummy_atom.coord[1] = 9999; dummy_atom.coord[2] = 9999;
    dummy_atom.index = parameters_info->SEL_A_max - 1;
    dummy.atom_info = &dummy_atom;
    dummy.dist = 9999;
    for (i = 0; i <= system_info_expanded->N_Atoms - 1; i++)
    {
        float dist;
        dist = sqrt(fastpow2(frame_info_cur->coord[index][0] - system_info_expanded->atom_info[i].coord[0] , 2) + fastpow2(frame_info_cur->coord[index][1] - system_info_expanded->atom_info[i].coord[1], 2) + fastpow2(frame_info_cur->coord[index][2] - system_info_expanded->atom_info[i].coord[2], 2));
        if ((dist > 1E-6)&&(dist <= parameters_info->cutoff_max)&&(frame_info_cur->type[index] != -1)&&(frame_info_cur->type[index]==system_info_expanded->type[i]))
        {
            dist_info[index_tmp_dist].atom_info = &(system_info_expanded->atom_info[i]);
            dist_info[index_tmp_dist].dist = dist;
            index_tmp_dist ++;
        }
    }
    
    for (i = index_tmp_dist; i <= parameters_info->SEL_A_max - 1; i ++)
    {
        dist_info[i].atom_info = &dummy_atom;
        dist_info[i].dist = 9999;
    } 


    if ((frame_info_cur->index == DEBUG_FRAME)&&(index == DEBUG_ATOM))
    {
        printf_d("distance from atom %d of frame %d:\n", DEBUG_ATOM, DEBUG_FRAME);
        for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
        {
            printf_d("atom index %d coord %.3lf %.3lf %.3lf dist %.6lf\n", dist_info[i].atom_info->index, dist_info[i].atom_info->coord[0], dist_info[i].atom_info->coord[1], dist_info[i].atom_info->coord[2], dist_info[i].dist);
        }
    }

    /*Sort dist_info_struct using tmp pointer(including self)*/
    dist_info_struct ** a_tmp, ** b_tmp;//b is the sorted result
    dist_info_struct ** a_tmp_, ** b_tmp_;

    /**dist_info_struct ** c_tmp;//Use an intermediate dist_info array. The dimension of c_tmp is SEL_A_max + 1. if (SEL_A_max + 1) is larger than all the elements with index larger than SEL_A_max-1 will be 9999 
    dist_info_struct for_c_tmp;
    atom_info_struct for_c_tmp_atom;
    int index_tmp = 0;
    for_c_tmp.atom_info = &(for_c_tmp_atom);
    for_c_tmp.atom_info->coord[0] = 9999; for_c_tmp.atom_info->coord[1] = 9999; for_c_tmp.atom_info->coord[2] = 9999;
    for_c_tmp.dist = 9999;
    c_tmp = (dist_info_struct **)calloc(parameters_info->SEL_A_max + 1, sizeof(dist_info_struct *));
    for (i = 0; i <= system_info_expanded->N_Atoms - 1; i++)
    {
        if (dist_info[i].dist <= parameters_info->cutoff_max)
        {
            c_tmp[index_tmp] = &(dist_info[i]);
            index_tmp ++;
        }
    }
    for (i = index_tmp; i <= parameters_info->SEL_A_max; i++)
    {
        c_tmp[i] = &(for_c_tmp);
    }**/

    /*a_tmp =  (dist_info_struct **)calloc(system_info_expanded->N_Atoms, sizeof(dist_info_struct *));
    b_tmp =  (dist_info_struct **)calloc(system_info_expanded->N_Atoms, sizeof(dist_info_struct *));
    //#pragma omp parallel for
    for (i = 0; i <= system_info_expanded->N_Atoms - 1; i++)
    {
        a_tmp[i] = &(dist_info[i]);
        b_tmp[i] = &(dist_info[i]);
    }
    quick_sort_dist_cur_atom(&a_tmp, &b_tmp, 0, system_info_expanded->N_Atoms - 1, system_info_expanded->N_Atoms);*/
    /**a_tmp =  (dist_info_struct **)calloc(parameters_info->SEL_A_max + 1, sizeof(dist_info_struct *));
    b_tmp =  (dist_info_struct **)calloc(parameters_info->SEL_A_max + 1, sizeof(dist_info_struct *));
    //a_tmp_ = &a_tmp[0]; b_tmp_ = &b_tmp[0];
    //#pragma omp parallel for
    for (i = 0; i <= parameters_info->SEL_A_max; i++)
    {
        a_tmp[i] = c_tmp[i];
        b_tmp[i] = c_tmp[i];
    }
    quick_sort_dist_cur_atom(&a_tmp, &b_tmp, 0, parameters_info->SEL_A_max, parameters_info->SEL_A_max + 1);**/
    a_tmp =  (dist_info_struct **)calloc(parameters_info->SEL_A_max, sizeof(dist_info_struct *));
    b_tmp =  (dist_info_struct **)calloc(parameters_info->SEL_A_max, sizeof(dist_info_struct *));
    //#pragma omp parallel for
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        a_tmp[i] = &(dist_info[i]);
        b_tmp[i] = &(dist_info[i]);
    }
    quick_sort_dist_cur_atom(&a_tmp, &b_tmp, 0, parameters_info->SEL_A_max - 1, parameters_info->SEL_A_max);
    #ifdef DEBUG_BUILD
    printf_d("Check if sorted.\n");
    int flag_sort = 1;
    /*for (i = 0; i <= system_info_expanded->N_Atoms - 2; i++)*/
    for (i = 0; i <= parameters_info->SEL_A_max - 2; i++)
    {
        if ((b_tmp[i + 1]->dist) < (b_tmp[i]->dist))
        {
            flag_sort = 0;
            printf_d("Not sorted: %d\n", i + 1);
        }
    }
    if (flag_sort == 1)
    {
        printf_d("Sorted frame %d atom %d.\n", frame_info_cur->index, neighbour_list_cur_atom->index);
    }
    if ((frame_info_cur->index == DEBUG_FRAME)&&(index == DEBUG_ATOM))
    {
        printf_d("Sorted distance from atom %d of frame %d:\n", DEBUG_ATOM, DEBUG_FRAME);
        /*for (i = 0; i <= system_info_expanded->N_Atoms - 1; i++)*/
        for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
        {
            printf_d("atom index %d coord %.3lf %.3lf %.3lf dist %.6lf\n", b_tmp[i]->atom_info->index, b_tmp[i]->atom_info->coord[0], b_tmp[i]->atom_info->coord[1], b_tmp[i]->atom_info->coord[2], b_tmp[i]->dist);
        }
    }
    #endif

    /*Choose the first SEL_A_max atoms to be the neighbour list atoms(excluding self)*/
    neighbour_list_cur_atom->coord_neighbours = (float **)calloc(parameters_info->SEL_A_max, sizeof(float));
    neighbour_list_cur_atom->type = (int *)calloc(parameters_info->SEL_A_max, sizeof(int));
    neighbour_list_cur_atom->index_neighbours = (int *)calloc(parameters_info->SEL_A_max, sizeof(int));
    /*for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)*/
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        neighbour_list_cur_atom->coord_neighbours[i] = (float *)calloc(3, sizeof(float));
        /*neighbour_list_cur_atom->type[i] = frame_info_cur->type[(b_tmp[i + 1]->atom_info->index)%(frame_info_cur->N_Atoms)];*/
        neighbour_list_cur_atom->type[i] = frame_info_cur->type[(b_tmp[i]->atom_info->index)%(frame_info_cur->N_Atoms)];
        neighbour_list_cur_atom->index_neighbours[i] = (b_tmp[i]->atom_info->index);
    }
    /*for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)*/
    for (i = 0; i <= parameters_info->SEL_A_max - 1; i++)
    {
        /*neighbour_list_cur_atom->coord_neighbours[i][0] = b_tmp[i + 1]->atom_info->coord[0];
        neighbour_list_cur_atom->coord_neighbours[i][1] = b_tmp[i + 1]->atom_info->coord[1];
        neighbour_list_cur_atom->coord_neighbours[i][2] = b_tmp[i + 1]->atom_info->coord[2];*/
        neighbour_list_cur_atom->coord_neighbours[i][0] = b_tmp[i]->atom_info->coord[0];
        neighbour_list_cur_atom->coord_neighbours[i][1] = b_tmp[i]->atom_info->coord[1];
        neighbour_list_cur_atom->coord_neighbours[i][2] = b_tmp[i]->atom_info->coord[2];
    }
    //corrupted size vs. prev_size NOT solved
    free(a_tmp);
    free(b_tmp);
    free(dist_info);
    return 0;
}

void quick_sort_dist_cur_atom(dist_info_struct *** a_tmp_, dist_info_struct *** b_tmp_, int start, int end, int tot_num)
{
    dist_info_struct ** a_tmp, ** b_tmp;
    int i, j, k;
    int left = start;
    int right = end;
    a_tmp = * a_tmp_; b_tmp = * b_tmp_;
    if (start > end)
    {
        return;
    }
    else
    {
        for (i = start + 1; i <= end; i++)
        {
            if (((a_tmp[i])->dist) > ((a_tmp[start])->dist))
            {
                b_tmp[right--] = a_tmp[i];
            }
            else
            {
                b_tmp[left++] = a_tmp[i];
            }
                       
        }
        b_tmp[left] = a_tmp[start];
        for (i = 0; i <= tot_num - 1; i++)
        {
            a_tmp[i] = b_tmp[i];
        }
        quick_sort_dist_cur_atom(a_tmp_, b_tmp_, start, left - 1, tot_num);
        quick_sort_dist_cur_atom(a_tmp_, b_tmp_, right + 1, end, tot_num);
    }
}
