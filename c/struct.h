/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Define all the struct used in the c code.
*/

typedef struct frame_info_struct_
{
	int index;
	int N_Atoms;
	double box[3][3];
	int * type;
	double ** coord;//coord[0..N_Atoms-1][0..2]
	double energy;
	int no_force;
	double ** force;//force[0..N_Atoms-1][0..2]
} frame_info_struct;
