/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Define all the struct used in the c code.
*/


/*****************MACRO FOR DEBUG*****************/
#define DEBUG_FRAME 0
#define DEBUG_ATOM 0
#define EXTEND_FLAG 32767
/***************MACRO FOR DEBUG END***************/

/*Neighbour list info for one atom in one frame*/
typedef struct neighbour_list_struct_
{
	int index;//atom index
	float cutoff_1;
	float cutoff_2;
	float cutoff_3;
	float cutoff_max;//cutoff_1 min, cutoff_max max. Four cutoffs just in case.
	/*For DeePMD, cutoff_max = cutoff_2 = rc, cutoff_1 = rcs*/
	int N_neighbours;//number of neighbour atoms within cutoff raduis cutoff_max
	float * coord_neighbours;//coord_neighbour[0..SEL_A_max - 1][0..2]
	float ** force_neighbours;//Just in case; force_neighbours[0..SEL_A_max - 1][0..2]
	//atom_info_struct * atom_info;
	int * type;//type[0..SEL_A_max - 1], type = all_types[0] for atoms with r_c > cutoff_max
	int * index_neighbours;//index of the neighbour atoms. index_neighbours[0..SEL_A_max]
	float * dist_neighbours;//dist[0..SEL_A_max - 1]
}neighbour_list_struct;

/*Frame info for one frame*/
typedef struct frame_info_struct_
{
	int index;//frame index
	int N_Atoms;
	int N_Atoms_ori;
	int N_types;
	float box[3][3];
	int * type;//type[0..N_Atoms-1], after expand type = -1 for dummy atoms
	float ** coord;//coord[0..N_Atoms-1][0..2]
	float energy;
	int no_force;
	float ** force;//force[0..N_Atoms-1][0..2]
	neighbour_list_struct * neighbour_list;//neighbour_list[0..N_Atoms-1], neighbour list for each atom
	int max_N_neighbours;//max number of neighbour atoms in this frame
	int * max_N_neighbours_ele;//max number of neighbour atoms (classified by element type) in this frame
} frame_info_struct;

/*Atom info for one atom, including index, coordinate and force*/
typedef struct atom_info_struct_
{
	int index;//index of this atom
	float coord[3];
	float force[3];
}atom_info_struct;

/*expanded system of one frame according to the cutoff_max and frame_info_struct.box*/
typedef struct system_info_expanded_struct_
{
	int N_Atoms;//Number of atoms in this expanded system;
	atom_info_struct * atom_info;//atom_info[0..N_Atoms-1]
	int * type;//type[0..N_Atoms]
}system_info_expanded_struct;

/*Store the distance between the current atom and atoms in its neighbour list(Not really. Shoubld be SEL_A_max)*/
typedef struct dist_info_struct_
{
	atom_info_struct * atom_info;
	float dist;
}dist_info_struct;

/*input parameters*/
typedef struct parameters_info_struct_
{
	float cutoff_1;
	float cutoff_2;
	float cutoff_3;
	float cutoff_max;
	int N_types_all_frame;
	int * type_index_all_frame;//type_index_all_frame[0..N_types_all_frame - 1]
	int N_Atoms_max;
	int SEL_A_max;
	int * SEL_A_ele;
	int * SEL_A_ele_max;
	int Nframes_tot;
	int sym_coord_type;
	int N_sym_coord;//Not used for DeePMD type
	
	int batch_size;
	int stop_epoch;
	int num_filter_layer;
	int * filter_neuron;
	int axis_neuron;
	int num_fitting_layer;
	int * fitting_neuron;
	float start_lr;
	int decay_steps;
	int decay_epoch;
	float decay_rate;
	float start_pref_e;
	float limit_pref_e;
	float start_pref_f;
	float limit_pref_f;
	int check_step;
	int check_batch;
	int check_epoch;
	int output_step;
	int output_batch;
	int output_epoch;
	int save_step;
	int save_batch;
	int save_epoch;

}parameters_info_struct;

/*converted coordinate of one frame(using DeePMD's method)*/
typedef struct sym_coord_DeePMD_struct_
{
	int N_Atoms;
	int SEL_A;
	int * type;//type[0..N_Atoms-1]
	float ** coord_converted;//coord_converted[0..N_Atoms-1][0..SEL_A*4-1]
	float ** d_to_center_x;//coord_converted[0..N_Atoms-1][0..SEL_A*4-1]
	float ** d_to_center_y;//coord_converted[0..N_Atoms-1][0..SEL_A*4-1]
	float ** d_to_center_z;//coord_converted[0..N_Atoms-1][0..SEL_A*4-1]
}sym_coord_DeePMD_struct;

typedef struct sym_coord_LASP_struct_
{
	int N_Atoms;
	int SEL_A;
	int N_PTSDs;//Number of PTSDs(Power-type structural descriptors) for each atom
	int * type;//type[0..N_Atoms-1]
	float ** coord_converted;//coord_converted[0..N_Atoms-1][0..N_PTSDs-1]
	int *** idx_nei;//idx_nei[0..N_Atoms - 1][0..N_PTSDs - 1][]. Record the index of all the atoms invloved in No.[atom_idx][PTSD_idx]'s PTSD.
	float *** d_x;//Record the d_PTSD / d_x.
	float *** d_y;//
	float *** d_z;//
}sym_coord_LASP_struct;

typedef struct parameters_PTSDs_info_one_line_struct_
{
	float cutoff_radius;
	int PTSD_type;//S1 or S2 or ... S6. Ranges from 0 to 5.
	int PTSD_N_body_type;//two or three or four
	int N_params;//Number of parameters in this PTSD
	int * neigh_type_array;//neigh_type_array[0..PTSD_N_body_type-1-1]
	/*The following values will be initialized as -9999*/
	/*Some of the values should be converted to int when calculating sym coords*/
	float n;
	float m;//Note: the m here is an different parameter from that appeared in the Y_LM function!
	float p;
	float L;
	float r_c;
	float zeta;
	float lambda;
	float Gmin;
	float Gmax;
	float * params_array;//params_array[0..N_params-1]
}parameters_PTSDs_info_one_line_struct;

typedef struct parameters_PTSDs_info_struct_
{
	int N_PTSD_types;//6, S_i^1~S_i^6
	int N_types_all_frame;// = parameters.N_types_all_frame
	int * PTSD_N_body_type;//PTSD_N_body_type[0..N_PTSD_types-1], {2, 2, 3, 3, 3, 4}
	int * PTSD_N_params;//Excluding rc. PTSD_N_params[0..N_PTSD_types-1], {1, 2, 4, 5, 4, 5}
	float cutoff_max;// = parameters.cutoff_max
	int ** N_cutoff_radius;//Number of cutoff layers for each type of PTSD, N_cutoff_radius[0..N_types_all_frame-1][0..N_PTSD_types-1]
	int ** N_neigh_inter;//N_neigh_inter[0..N_types_all_frame-1][0..N_PTSD_types-1], Number of neigh-type combinations for each type of PTSD, 
	                     //for example, K body N type interaction: N_neigh_inter = a[K][N] = N * b[K][N], b[K][N] = b[K-1][N] + b[K-1][N-1] + ... + b[K-1][1]
	float *** cutoff_radius;//cutoff_radius[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]]
	float **** n;//n[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]][N_neigh_inter[i, i=0..N_PTSD_types-1]]
	float **** m;//n[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]][N_neigh_inter[i, i=0..N_PTSD_types-1]]
	float **** p;//n[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]][N_neigh_inter[i, i=0..N_PTSD_types-1]]
	float **** L;//n[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]][N_neigh_inter[i, i=0..N_PTSD_types-1]]
	float **** r_c;//n[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]][N_neigh_inter[i, i=0..N_PTSD_types-1]]
	float **** zeta;//n[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]][N_neigh_inter[i, i=0..N_PTSD_types-1]]
	float **** lambda;//n[0..N_types_all_frame-1][0..N_PTSD_types-1][N_cutoff_radius[i, i=0..N_PTSD_types-1]][N_neigh_inter[i, i=0..N_PTSD_types-1]]
	float **** Gmin;
	float **** Gmax;
	/*from cutoff_radius to Gmax are not used.*/
	/*All the parameters are packed into the following struct.*/
	parameters_PTSDs_info_one_line_struct *** parameters_PTSDs_info_one_line;
}parameters_PTSDs_info_struct;



