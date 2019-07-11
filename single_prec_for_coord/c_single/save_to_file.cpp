/*
2019.03.30 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Save energy of each frame & force of each atom of each frame, all parameters, and sym_coord to file.
Energy , type and N_Atoms of each frame, force and sym_coord of each atom of each frame are saved as binary files.
Original coordinates are also saved using binary form.

Return code:
    0: No errors.
    1~9: save_to_file_energy_and_force() error.
    11~19: save_to_file_parameters() error.
    21~29: save_to_file_type_and_N_Atoms() error.
    31~39: save_to_file_sym_coord() error.
    41~49: check_sym_coord_from_bin() error.
    51~59: save_to_file_coord() error.
    61~69: save_to_file_nei() error.

*/

#include <stdio.h>
#include <stdlib.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
//#define DEBUG_SAV

#ifdef DEBUG_SAV
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int save_to_file(frame_info_struct * frame_info, parameters_info_struct * parameters_info, void * sym_coord)
{
    int save_to_file_energy_and_force(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    int save_to_file_parameters(parameters_info_struct * parameters_info);
    int save_to_file_type_and_N_Atoms(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    int save_to_file_sym_coord(void * sym_coord, parameters_info_struct * parameters_info);
    int save_to_file_coord(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    int save_to_file_nei(frame_info_struct * frame_info, parameters_info_struct * parameters_info);

    int ef_flag, p_flag, sc_flag, tna_flag, c_flag, nei_flag;

    ef_flag = save_to_file_energy_and_force(frame_info, parameters_info);
    if (ef_flag != 0)
    {
        return ef_flag;
    }

    p_flag = save_to_file_parameters(parameters_info);
    if (p_flag != 0)
    {
        return p_flag;
    }

    tna_flag = save_to_file_type_and_N_Atoms(frame_info, parameters_info);
    if (tna_flag != 0)
    {
        return tna_flag;
    }

    sc_flag = save_to_file_sym_coord(sym_coord, parameters_info);
    if (sc_flag != 0)
    {
        return sc_flag;
    }

    c_flag = save_to_file_coord(frame_info, parameters_info);
    if (c_flag != 0)
    {
        return c_flag;
    }

    nei_flag = save_to_file_nei(frame_info, parameters_info);
    if (nei_flag != 0)
    {
        return nei_flag;
    }

    return 0;
}

int save_to_file_partial(frame_info_struct * frame_info, parameters_info_struct * parameters_info, void * sym_coord)
{
    int save_to_file_energy_and_force(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    int save_to_file_parameters(parameters_info_struct * parameters_info);
    int save_to_file_type_and_N_Atoms(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    /*int save_to_file_sym_coord(void * sym_coord, parameters_info_struct * parameters_info);*/
    int save_to_file_coord(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    int save_to_file_nei(frame_info_struct * frame_info, parameters_info_struct * parameters_info);

    int ef_flag, p_flag, sc_flag, tna_flag, c_flag, nei_flag;

    ef_flag = save_to_file_energy_and_force(frame_info, parameters_info);
    if (ef_flag != 0)
    {
        return ef_flag;
    }

    p_flag = save_to_file_parameters(parameters_info);
    if (p_flag != 0)
    {
        return p_flag;
    }

    tna_flag = save_to_file_type_and_N_Atoms(frame_info, parameters_info);
    if (tna_flag != 0)
    {
        return tna_flag;
    }

    /*sc_flag = save_to_file_sym_coord(sym_coord, parameters_info);
    if (sc_flag != 0)
    {
        return sc_flag;
    }*/

    c_flag = save_to_file_coord(frame_info, parameters_info);
    if (c_flag != 0)
    {
        return c_flag;
    }

    nei_flag = save_to_file_nei(frame_info, parameters_info);
    if (nei_flag != 0)
    {
        return nei_flag;
    }

    return 0;
}

int save_to_file_predict(frame_info_struct * frame_info, parameters_info_struct * parameters_info, void * sym_coord)
{
    /*int save_to_file_energy_and_force(frame_info_struct * frame_info, parameters_info_struct * parameters_info);*/
    int save_to_file_parameters(parameters_info_struct * parameters_info);
    int save_to_file_type_and_N_Atoms(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    /*int save_to_file_sym_coord(void * sym_coord, parameters_info_struct * parameters_info);*/
    int save_to_file_coord(frame_info_struct * frame_info, parameters_info_struct * parameters_info);
    int save_to_file_nei(frame_info_struct * frame_info, parameters_info_struct * parameters_info);

    int ef_flag, p_flag, sc_flag, tna_flag, c_flag, nei_flag;

    /*ef_flag = save_to_file_energy_and_force(frame_info, parameters_info);
    if (ef_flag != 0)
    {
        return ef_flag;
    }*/

    p_flag = save_to_file_parameters(parameters_info);
    if (p_flag != 0)
    {
        return p_flag;
    }

    tna_flag = save_to_file_type_and_N_Atoms(frame_info, parameters_info);
    if (tna_flag != 0)
    {
        return tna_flag;
    }

    /*sc_flag = save_to_file_sym_coord(sym_coord, parameters_info);
    if (sc_flag != 0)
    {
        return sc_flag;
    }*/

    c_flag = save_to_file_coord(frame_info, parameters_info);
    if (c_flag != 0)
    {
        return c_flag;
    }

    nei_flag = save_to_file_nei(frame_info, parameters_info);
    if (nei_flag != 0)
    {
        return nei_flag;
    }

    return 0;
}

int save_to_file_energy_and_force(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    FILE * fp_energy;
    FILE * fp_force;
    int i, j, k;

    fp_energy = fopen("./ENERGY.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        fwrite(&(frame_info[i].energy), sizeof(double), 1, fp_energy);
    }
    fclose(fp_energy);

    fp_force = fopen("./FORCE.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            fwrite(frame_info[i].force[j], sizeof(double), 3, fp_force);
        }
    }
    fclose(fp_force);

    return 0;
}

int save_to_file_parameters(parameters_info_struct * parameters_info)
{
    FILE * fp_parameters;
    int i, j, k;

    fp_parameters = fopen("./ALL_PARAMS.json", "w");
    fprintf(fp_parameters, "{\n");//head

    fprintf(fp_parameters, "    \"cutoff_1\": %.3lf,\n", parameters_info->cutoff_1);
    fprintf(fp_parameters, "    \"cutoff_2\": %.3lf,\n", parameters_info->cutoff_2);
    fprintf(fp_parameters, "    \"cutoff_3\": %.3lf,\n", parameters_info->cutoff_3);
    fprintf(fp_parameters, "    \"cutoff_max\": %.3lf,\n", parameters_info->cutoff_max);
    fprintf(fp_parameters, "    \"N_types_all_frame\": %d,\n", parameters_info->N_types_all_frame);

    fprintf(fp_parameters, "    \"type_index_all_frame\": [\n");
    for (i = 0; i <= parameters_info->N_types_all_frame - 2; i++)
    {
        fprintf(fp_parameters, "        %d,\n", parameters_info->type_index_all_frame[i]);
    }
    fprintf(fp_parameters, "        %d\n", parameters_info->type_index_all_frame[i]);
    fprintf(fp_parameters, "    ],\n");

    fprintf(fp_parameters, "    \"N_Atoms_max\": %d,\n", parameters_info->N_Atoms_max);
    fprintf(fp_parameters, "    \"SEL_A_max\": %d,\n", parameters_info->SEL_A_max);
    fprintf(fp_parameters, "    \"Nframes_tot\": %d,\n", parameters_info->Nframes_tot);
    fprintf(fp_parameters, "    \"sym_coord_type\": %d,\n", parameters_info->sym_coord_type);
    fprintf(fp_parameters, "    \"N_sym_coord\": %d,\n", parameters_info->N_sym_coord);

    fprintf(fp_parameters, "    \"batch_size\": %d,\n", parameters_info->batch_size);
    fprintf(fp_parameters, "    \"stop_epoch\": %d,\n", parameters_info->stop_epoch);
    fprintf(fp_parameters, "    \"num_filter_layer\": %d,\n", parameters_info->num_filter_layer);

    fprintf(fp_parameters, "    \"filter_neuron\": [\n");
    for (i = 0; i <= parameters_info->num_filter_layer - 2; i++)
    {
        fprintf(fp_parameters, "        %d,\n", parameters_info->filter_neuron[i]);
    }
    fprintf(fp_parameters, "        %d\n", (parameters_info->sym_coord_type == 1 ? parameters_info->filter_neuron[i] : parameters_info->N_sym_coord));
    fprintf(fp_parameters, "    ],\n");

    fprintf(fp_parameters, "    \"axis_neuron\": %d,\n", (parameters_info->sym_coord_type == 1 ? parameters_info->axis_neuron : 1));
    fprintf(fp_parameters, "    \"num_fitting_layer\": %d,\n", parameters_info->num_fitting_layer);

    fprintf(fp_parameters, "    \"fitting_neuron\": [\n");
    for (i = 0; i <= parameters_info->num_fitting_layer - 2; i++)
    {
        fprintf(fp_parameters, "        %d,\n", parameters_info->fitting_neuron[i]);
    }
    fprintf(fp_parameters, "        %d\n", parameters_info->fitting_neuron[i]);
    fprintf(fp_parameters, "    ],\n");

    fprintf(fp_parameters, "    \"start_lr\": %.10e,\n", parameters_info->start_lr);
    fprintf(fp_parameters, "    \"decay_steps\": %d,\n", parameters_info->decay_steps);
    fprintf(fp_parameters, "    \"decay_epoch\": %d,\n", parameters_info->decay_epoch);
    fprintf(fp_parameters, "    \"decay_rate\": %.10e,\n", parameters_info->decay_rate);
    fprintf(fp_parameters, "    \"start_pref_e\": %.10e,\n", parameters_info->start_pref_e);
    fprintf(fp_parameters, "    \"limit_pref_e\": %.10e,\n", parameters_info->limit_pref_e);
    fprintf(fp_parameters, "    \"start_pref_f\": %.10e,\n", parameters_info->start_pref_f);
    fprintf(fp_parameters, "    \"limit_pref_f\": %.10e,\n", parameters_info->limit_pref_f);
    fprintf(fp_parameters, "    \"check_step\": %d,\n", parameters_info->check_step);
    fprintf(fp_parameters, "    \"check_batch\": %d,\n", parameters_info->check_batch);
    fprintf(fp_parameters, "    \"check_epoch\": %d,\n", parameters_info->check_epoch);
    fprintf(fp_parameters, "    \"output_step\": %d,\n", parameters_info->output_step);
    fprintf(fp_parameters, "    \"output_batch\": %d,\n", parameters_info->output_batch);
    fprintf(fp_parameters, "    \"output_epoch\": %d,\n", parameters_info->output_epoch);
    fprintf(fp_parameters, "    \"save_step\": %d,\n", parameters_info->save_step);
    fprintf(fp_parameters, "    \"save_batch\": %d,\n", parameters_info->save_batch);
    fprintf(fp_parameters, "    \"save_epoch\": %d\n", parameters_info->save_epoch);


    fprintf(fp_parameters, "}\n");//tail. The last line before tail should NOT end with a coma
    fclose(fp_parameters);

    /*Save a binary version of parameters*/
    /*Not completed. Parsing json seems to be more flexible.*/
    return 0;
}

int save_to_file_type_and_N_Atoms(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    FILE * fp_type;
    FILE * fp_N_Atoms;
    int i, j, k;

    fp_type = fopen("./TYPE.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        fwrite(frame_info[i].type, sizeof(int), frame_info[i].N_Atoms, fp_type);
    }
    fclose(fp_type);

    fp_N_Atoms = fopen("./N_ATOMS.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        fwrite(&(frame_info[i].N_Atoms), sizeof(int), 1, fp_N_Atoms);
        printf_d("%d ", frame_info[i].N_Atoms);
    }
    printf_d("\n");
    fclose(fp_N_Atoms);

    fp_N_Atoms = fopen("./N_ATOMS_ORI.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        fwrite(&(frame_info[i].N_Atoms_ori), sizeof(int), 1, fp_N_Atoms);
        printf_d("%d ", frame_info[i].N_Atoms_ori);
    }
    printf_d("\n");
    fclose(fp_N_Atoms);

    return 0;
}

int save_to_file_sym_coord(void * sym_coord, parameters_info_struct * parameters_info)
{
    int save_to_file_sym_coord_DeePMD(void * sym_coord, parameters_info_struct * parameters_info);
    int save_to_file_sym_coord_LASP(void * sym_coord, parameters_info_struct * parameters_info);

    int error_code;
    int sym_coord_type = parameters_info->sym_coord_type;

    switch (sym_coord_type)
    {
        case 1:
        {
            error_code = save_to_file_sym_coord_DeePMD(sym_coord, parameters_info);
            break;
        }
        case 2:
        {
            error_code = save_to_file_sym_coord_LASP(sym_coord, parameters_info);
            break;
        }
        default:
        {
            printf_d("Symmetry coordinate type not supported!\n");
            return 31;
        }
    }    

    return error_code;
}

int save_to_file_sym_coord_DeePMD(void * sym_coord, parameters_info_struct * parameters_info)
{
    int check_sym_coord_from_bin(parameters_info_struct * parameters_info);

    FILE * fp_sym_coord;
    FILE * fp_N_Atoms;
    sym_coord_DeePMD_struct * sym_coord_DeePMD = (sym_coord_DeePMD_struct *)sym_coord;
    int * N_Atoms_array = (int *)calloc(parameters_info->Nframes_tot, sizeof(int));
    int i, j, k;

    fp_N_Atoms = fopen("./N_ATOMS.BIN", "rb");
    fread(N_Atoms_array, sizeof(int), parameters_info->Nframes_tot, fp_N_Atoms);
    fclose(fp_N_Atoms);

    printf_d("N_Atoms_array read from file:\n");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        printf_d("%d ", N_Atoms_array[i]);
    }
    printf_d("\n");

    fp_sym_coord = fopen("./SYM_COORD.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            fwrite(sym_coord_DeePMD[i].coord_converted[j], sizeof(double), parameters_info->SEL_A_max * 4, fp_sym_coord);
        }
    }
    fclose(fp_sym_coord);

    fp_sym_coord = fopen("./SYM_COORD_DX.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            fwrite(sym_coord_DeePMD[i].d_to_center_x[j], sizeof(double), parameters_info->SEL_A_max * 4, fp_sym_coord);
        }
    }
    fclose(fp_sym_coord);
    fp_sym_coord = fopen("./SYM_COORD_DY.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            fwrite(sym_coord_DeePMD[i].d_to_center_y[j], sizeof(double), parameters_info->SEL_A_max * 4, fp_sym_coord);
        }
    }
    fclose(fp_sym_coord);
    fp_sym_coord = fopen("./SYM_COORD_DZ.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            fwrite(sym_coord_DeePMD[i].d_to_center_z[j], sizeof(double), parameters_info->SEL_A_max * 4, fp_sym_coord);
        }
    }
    fclose(fp_sym_coord);

    free(N_Atoms_array);
    
    check_sym_coord_from_bin(parameters_info);

    return 0;
}

int save_to_file_sym_coord_LASP(void * sym_coord, parameters_info_struct * parameters_info)
{
    int check_sym_coord_from_bin(parameters_info_struct * parameters_info);

    FILE * fp_sym_coord;
    FILE * fp_N_Atoms;
    sym_coord_LASP_struct * sym_coord_LASP = (sym_coord_LASP_struct *)sym_coord;
    int * N_Atoms_array = (int *)calloc(parameters_info->Nframes_tot, sizeof(int));
    int i, j, k, l;

    fp_N_Atoms = fopen("./N_ATOMS.BIN", "rb");
    fread(N_Atoms_array, sizeof(int), parameters_info->Nframes_tot, fp_N_Atoms);
    fclose(fp_N_Atoms);

    printf_d("N_Atoms_array read from file:\n");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        printf_d("%d ", N_Atoms_array[i]);
    }
    printf_d("\n");

    fp_sym_coord = fopen("./SYM_COORD.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            fwrite(sym_coord_LASP[i].coord_converted[j], sizeof(double), parameters_info->N_sym_coord, fp_sym_coord);
        }
    }
    fclose(fp_sym_coord);

    /*Incomplete*/
    fp_sym_coord = fopen("./SYM_COORD_DX.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            for (k = 0; k <= parameters_info->N_sym_coord - 1; k++)
            {
                fwrite(sym_coord_LASP[i].d_x[j][k], sizeof(double), N_Atoms_array[i], fp_sym_coord);
            }
        }
    }
    fclose(fp_sym_coord);
    fp_sym_coord = fopen("./SYM_COORD_DY.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            for (k = 0; k <= parameters_info->N_sym_coord - 1; k++)
            {
                fwrite(sym_coord_LASP[i].d_y[j][k], sizeof(double), N_Atoms_array[i], fp_sym_coord);
            }
        }
    }
    fclose(fp_sym_coord);
    fp_sym_coord = fopen("./SYM_COORD_DZ.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= N_Atoms_array[i] - 1; j++)
        {
            for (k = 0; k <= parameters_info->N_sym_coord - 1; k++)
            {
                fwrite(sym_coord_LASP[i].d_z[j][k], sizeof(double), N_Atoms_array[i], fp_sym_coord);
            }
        }
    }
    fclose(fp_sym_coord);

    free(N_Atoms_array);

    return 0;
}

int check_sym_coord_from_bin(parameters_info_struct * parameters_info)
{
    int check_sym_coord_from_bin_DeePMD(parameters_info_struct * parameters_info);
    
    int error_code;
    int sym_coord_type = parameters_info->sym_coord_type;

    switch (sym_coord_type)
    {
        case 1:
        {
            error_code = check_sym_coord_from_bin_DeePMD(parameters_info);
            break;
        }
        default:
        {
            printf_d("Symmetry coordinate type not supported!\n");
            return 41;
        }
    }

    return error_code;

}

int check_sym_coord_from_bin_DeePMD(parameters_info_struct * parameters_info)
{
    FILE * fp_sym_coord;
    FILE * fp_N_Atoms;
    int * N_Atoms_array = (int *)calloc(parameters_info->Nframes_tot, sizeof(int));
    double * sym_coord_DeePMD;
    int tot_atoms = 0;
    int offset = 0;
    int i, j, k;

    fp_N_Atoms = fopen("./N_ATOMS.BIN", "rb");
    fread(N_Atoms_array, sizeof(int), parameters_info->Nframes_tot, fp_N_Atoms);
    fclose(fp_N_Atoms);

    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        tot_atoms += N_Atoms_array[i];
    }
    sym_coord_DeePMD = (double *)calloc(tot_atoms * parameters_info->SEL_A_max * 4, sizeof(double));
    fp_sym_coord = fopen("./SYM_COORD.BIN", "rb");
    fread(sym_coord_DeePMD, sizeof(double), tot_atoms * parameters_info->SEL_A_max * 4, fp_sym_coord);
    fclose(fp_sym_coord);

    printf_d("Check sym_coord from bin file of atom %d of frame %d:\n", DEBUG_ATOM, DEBUG_FRAME);
    printf_d("%-11s %-11s %-11s %-11s\n", "s_rij", "x_hat", "y_hat", "z_hat");
    for (i = 0; i <= DEBUG_FRAME - 1; i++)
    {
        offset += N_Atoms_array[i];
    }
    offset = offset + DEBUG_ATOM;
    printf_d("Offset = %d\n", offset);
    offset = offset * parameters_info->SEL_A_max * 4;
    for (j = 0; j <= parameters_info->SEL_A_max - 1; j++)
    {
            for (k = 0; k <= 3; k++)
            {
                int idx = j * 4 + k;
                printf_d("%+10.6lf ", sym_coord_DeePMD[offset + idx]);
            }
            printf_d("\n");
    }

    free(N_Atoms_array);free(sym_coord_DeePMD);
    return 0;
}

int save_to_file_coord(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    FILE * fp_coord;
    int i, j, k;

    fp_coord = fopen("./COORD.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            fwrite((frame_info[i].coord[j]), sizeof(double), 3, fp_coord);
        }
    }
    fclose(fp_coord);
    return 0;
}

/*Save the indexes and coordinates of neighbour atoms of the atoms in frames*/
int save_to_file_nei(frame_info_struct * frame_info, parameters_info_struct * parameters_info)
{
    FILE * fp_nei;
    FILE * fp_nei_coord;
    FILE * fp_nei_type;
    FILE * fp_nei_dist;
    int i, j, k;

    fp_nei = fopen("./NEI_IDX.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            fwrite(frame_info[i].neighbour_list[j].index_neighbours, sizeof(int), parameters_info->SEL_A_max, fp_nei);
        }
    }
    fclose(fp_nei);
    fp_nei_coord = fopen("./NEI_COORD.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            for (k = 0; k <= parameters_info->SEL_A_max - 1; k++)
            {
                fwrite(frame_info[i].neighbour_list[j].coord_neighbours[k], sizeof(float), 3, fp_nei_coord);
            }
        }
    }
    fclose(fp_nei_coord);
    fp_nei_type = fopen("./NEI_TYPE.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            //for (k = 0; k <= parameters_info->SEL_A_max - 1; k++)
            //{
            fwrite(frame_info[i].neighbour_list[j].type, sizeof(int), parameters_info->SEL_A_max, fp_nei_type);
            //}
        }
    }
    fclose(fp_nei_type);
    fp_nei_dist = fopen("./NEI_DIST.BIN", "wb");
    for (i = 0; i <= parameters_info->Nframes_tot - 1; i++)
    {
        for (j = 0; j <= frame_info[i].N_Atoms - 1; j++)
        {
            fwrite(frame_info[i].neighbour_list[j].dist_neighbours, sizeof(double), parameters_info->SEL_A_max, fp_nei_dist);
        }
    }
    fclose(fp_nei_dist);
    return 0;
}
