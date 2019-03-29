/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Define all the struct used in the c code.
*/

/*Neighbour list info for one atom in one frame*/
typedef struct neighbour_list_struct_
{
	int index;//atom index
	double cutoff_1;
	double cutoff_2;
	double cutoff_3;
	double cutoff_max;//cutoff_1 min, cutoff_max max. Four cutoffs just in case.
	int N_neighbours;//number of neighbour atoms within cutoff raduis cutoff_max
	double ** coord_neighbours;//coord_neighbour[0..N_neighbours][0..2]
	double ** force_neighbours;//Just in case; force_neighbours[0..N_neighbours][0..2]
	int * type;//type[0..N_neighbours]
}neighbour_list_struct;

/*Frame info for one frame*/
typedef struct frame_info_struct_
{
	int index;
	int N_Atoms;
	int N_types;
	double box[3][3];
	int * type;//type[0..N_Atoms-1]
	double ** coord;//coord[0..N_Atoms-1][0..2]
	double energy;
	int no_force;
	double ** force;//force[0..N_Atoms-1][0..2]
	neighbour_list_struct * neighbour_list;//neighbour_list[0..N_Atoms-1], neighbour list for each atom
	int max_N_neighbours;//max number of neighbour atoms in this frame
} frame_info_struct;

/*Atom info for one atom, including index, coordinate and force*/
typedef struct atom_info_struct_
{
	int index;//index of this atom
	double coord[3];
	double force[3];
}atom_info_struct;

/*expanded system of one frame according to the cutoff_max and frame_info_struct.box*/
typedef struct system_info_expanded_struct_
{
	int N_Atoms;//Number of atoms in this expanded system;
	atom_info_struct * atom_info;//atom_info[0..N_Atoms-1]
	int * type;//type[0..N_Atoms]
}system_info_expanded_struct;

typedef struct parameters_info_struct_
{
	double cutoff_1;
	double cutoff_2;
	double cutoff_3;
	double cutoff_max;
}parameters_info_struct;


