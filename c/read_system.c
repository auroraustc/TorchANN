/*
2019.03.26 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Read (Nframes_tot*)box, (Nframes_tot*)coord, (Nframes_tot*)energy, (Nframes_tot*)force, (Nframes_tot*)type from box.raw, coord.raw, energy.raw, force.raw and type.raw from the current directory.
Save box, coord, energy, force and type of each frame in a struct array frame_info[0..Nframes_tot-1] and save this array to a temp binary file in the current directory.
The format of box.raw, coord.raw, energy.raw and force.raw is the same as DeePMD, containing Nframes_tot lines. The type.raw should also contain Nframes_tot lines for this code.

Return code:
	0: No errors.
	1: missing input file(s).
	2: The number of frames provided in box.raw, energy.raw and type.raw are different.
	3: Coordinates or/and forces in some frames are incorrect.
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_READ

#ifdef DEBUG_READ
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int read_system(frame_info_struct * frame_info_, int * Nframes_tot_)
{
	int compare_NAtoms(char * coord, char * force, char * type);
	int parse_coord_force_type(char * coord, char * force, char * type, int N_Atoms_this_frame, frame_info_struct * frame_info);
	void check_bin(int Nframes_tot);

	frame_info_struct * frame_info;
	FILE * fp_box, * fp_coord, * fp_energy, * fp_force, * fp_type;
	FILE * fp_out;
	int error_flag_file;
	int no_force;
	int i, j, k;
	int tot_lines_box, tot_lines_energy, tot_lines_type;//These three numbers should be equal to Nframes_tot
	int tot_lines_coord, tot_lines_force;//These two numbers should be equal to Nframes_tot
	int Nframes_tot, N_Atoms, N_Atoms_this_frame;
	double tmpd1;
	char * tmp_coord = (char *) calloc(100000 * 3, sizeof(char));
	char * tmp_force = (char *) calloc(100000 * 3, sizeof(char));
	char * tmp_type = (char *) calloc(100000 , sizeof(char));
	char * tmp_coord_cpy, * tmp_force_cpy, * tmp_type_cpy;
	int tmpi1;
	char tmpc1;

	fp_box = fopen("./box.raw", "r");
	fp_coord = fopen("./coord.raw", "r");
	fp_energy = fopen("./energy.raw", "r");
	fp_force = fopen("./force.raw", "r");
	no_force = (fp_force == NULL ? 1: 0);
	fp_type = fopen("./type.raw", "r");
	error_flag_file = !((fp_box != NULL) && (fp_coord != NULL) && (fp_energy != NULL) && (fp_type != NULL));
	if (error_flag_file)
	{
		printf("Please provide box.raw, coord.raw, energy.raw and type.raw\n");
		return 1;
	}
	if (no_force == 1)
	{
		printf("No force.raw is found. Force will not be added to the loss function.\n");
	}

	tot_lines_box = 0;
	while (!feof(fp_box))
	{
		if (fscanf(fp_box, "%lf%*[^\n]%c", &tmpd1, &tmpc1) < 2) break;
		tot_lines_box++;
	}
	printf_d("tot_lines_box: %d\n", tot_lines_box);
	tot_lines_energy = 0;
	while (!feof(fp_energy))
	{
		if (fscanf(fp_energy, "%lf", &tmpd1) < 1) break;
		tot_lines_energy ++;
	}
	printf_d("tot_lines_energy: %d\n", tot_lines_energy);
	tot_lines_type = 0;
	while (!feof(fp_type))
	{
		if (fscanf(fp_type, "%d%*[^\n]%c", &tmpi1, &tmpc1) < 2) break;
		tot_lines_type++;
	}
	printf_d("tot_lines_type: %d\n", tot_lines_type);
	if ((tot_lines_box == tot_lines_energy)&&(tot_lines_energy == tot_lines_type))
	{
		Nframes_tot = tot_lines_box;
		printf("Number of frames in total: %d\n", Nframes_tot);
	}
	else
	{
		printf("The number of frames provided in box.raw, energy.raw and type.raw are different: %d, %d, %d. Please check the input files.\n", tot_lines_box, tot_lines_energy, tot_lines_type);
		return 2;
	}
	printf_d("Seg Checkpoint 1\n");
	fclose(fp_box);fclose(fp_energy);fclose(fp_type);
	tot_lines_coord = 0;
	while (!feof(fp_coord))
	{
		if (fscanf(fp_coord, "%lf%*[^\n]%c", &tmpd1, &tmpc1) < 2) break;
		tot_lines_coord++;
	}
	printf_d("tot_lines_coord: %d\n", tot_lines_coord);
	printf_d("no_force = %d\n", no_force);
	if (no_force == 0)
	{
		tot_lines_force = 0;
		while ((!feof(fp_force)))
		{
			if (fscanf(fp_force, "%lf%*[^\n]%c", &tmpd1, &tmpc1) < 2) break;
			tot_lines_force++;
		}
		printf_d("tot_lines_force: %d\n", tot_lines_force);
	}
	if ((tot_lines_coord == Nframes_tot)&&((no_force == 1)||(tot_lines_force == Nframes_tot)))
	{
		;
	}
	else
	{
		printf("The number of frames provided in coord.raw or force.raw is incorrect: %d %d\n", tot_lines_coord, tot_lines_force);
	}
	fclose(fp_coord);if (no_force == 0) fclose(fp_force);

	printf_d("Seg Checkpoint 2\n");
	fp_box = fopen("./box.raw", "r");
	fp_energy = fopen("./energy.raw", "r");
	frame_info = (frame_info_struct *)calloc(Nframes_tot, sizeof(frame_info_struct));
	/*Read in box and energy info*/
	for (i = 0; i <= Nframes_tot - 1; i++)
	{
		fscanf(fp_box, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &frame_info[i].box[0][0], &frame_info[i].box[0][1], &frame_info[i].box[0][2], &frame_info[i].box[1][0], &frame_info[i].box[1][1], &frame_info[i].box[1][2], &frame_info[i].box[2][0], &frame_info[i].box[2][1], &frame_info[i].box[2][2]);
		printf_d("frame %d ", i + 1);
		printf_d("box:  %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf %.3lf ", frame_info[i].box[0][0], frame_info[i].box[0][1], frame_info[i].box[0][2], frame_info[i].box[1][0], frame_info[i].box[1][1], frame_info[i].box[1][2], frame_info[i].box[2][0], frame_info[i].box[2][1], frame_info[i].box[2][2]);
		fscanf(fp_energy, "%lf", &frame_info[i].energy);
		printf_d("energy:  %.3lf eV\n", frame_info[i].energy);
	}
	fclose(fp_box);fclose(fp_energy);
	/*Count NAtoms for each frame and read in coord, force and type*/
	fp_type = fopen("./type.raw", "r");
	fp_coord = fopen("./coord.raw", "r");
	if (no_force == 0) 
	{
		fp_force = fopen("./force.raw", "r");
	}
	else
	{
		fp_force = fopen("./coord.raw", "r");
	}
	for (i = 0; i <= Nframes_tot - 1; i++)
	{
		fgets(tmp_coord, 100000 * 3, fp_coord);
		fgets(tmp_force, 100000 * 3, fp_force);
		fgets(tmp_type, 100000, fp_type);
		/*Check NAtoms from coord.raw, force.raw and type.raw correspond.*/
		tmp_coord_cpy = (char *)calloc(strlen(tmp_coord), sizeof(char));
		tmp_force_cpy = (char *)calloc(strlen(tmp_force), sizeof(char));
		tmp_type_cpy = (char *)calloc(strlen(tmp_type), sizeof(char));
		strcpy(tmp_coord_cpy, tmp_coord);
		strcpy(tmp_force_cpy, tmp_force);
		strcpy(tmp_type_cpy, tmp_type);
		N_Atoms_this_frame = compare_NAtoms(tmp_coord_cpy, tmp_force_cpy, tmp_type_cpy);
		switch (N_Atoms_this_frame)
		{
			case -1:
			{
				printf("Frame %d: Number of atoms of from coord.raw and force.raw are different.\n", i + 1);
				return 3;
			}
			case -2:
			{
				printf("Frame %d: coord or force information of this frame is incomplete.\n", i + 1);
				return 3;
			}
			case -3:
			{
				printf("Frame %d: Number of atoms of coord.raw or force.raw is different from NAtoms of type.raw.\n", i + 1);
				return 3;
			}
			default:
			{
				printf_d("Number of atoms in frame %d is: %d \n", i + 1, N_Atoms_this_frame);
				break;
			}
		}
		/*Parse tmp_coord, tmp_force and tmp_type according to N_Atoms_this_frame and save the data in frame_info[i].*/
		frame_info[i].coord = (double **)calloc(N_Atoms_this_frame, sizeof(double *));
		frame_info[i].force = (double **)calloc(N_Atoms_this_frame, sizeof(double *));
		for (j = 0; j <= N_Atoms_this_frame - 1; j ++)
		{
			frame_info[i].coord[j] = (double *)calloc(3, sizeof(double));
			frame_info[i].force[j] = (double *)calloc(3, sizeof(double));
		}
		frame_info[i].type = (int *)calloc(N_Atoms_this_frame, sizeof(int));
		strcpy(tmp_coord_cpy, tmp_coord);
		strcpy(tmp_force_cpy, tmp_force);
		strcpy(tmp_type_cpy, tmp_type);
		parse_coord_force_type(tmp_coord_cpy, tmp_force_cpy, tmp_type_cpy, N_Atoms_this_frame, &(frame_info[i]));
		frame_info[i].no_force = no_force;
		frame_info[i].index = i;
		frame_info[i].N_Atoms = N_Atoms_this_frame;
	}
	fclose(fp_coord);fclose(fp_force);fclose(fp_type);
	/*All the data has been read in. Save the frame_into to a binary file all_frame_info.bin.temp*/
	fp_out = fopen("all_frame_info.bin.temp","wb");
	fwrite(frame_info, sizeof(frame_info_struct), Nframes_tot, fp_out);
	fclose(fp_out);
	check_bin(Nframes_tot);
	frame_info_ = frame_info; *Nframes_tot_ = Nframes_tot;
	return 0;
}

int compare_NAtoms(char * coord, char * force, char * type)
{
	char * token_coord, * token_force, * token_type;
	int NAtoms_coord, NAtoms_force, NAtoms_type;
	NAtoms_coord=0; NAtoms_force=0; NAtoms_type=0;
	token_coord = strtok(coord, " \n"); 
	while (token_coord != NULL) 
	{
		NAtoms_coord++;
		token_coord = strtok(NULL, " \n");
	}
	token_force = strtok(force, " \n"); 
	while (token_force != NULL) 
	{
		NAtoms_force++;
		token_force = strtok(NULL, " \n");
	}
	token_type = strtok(type, " \n");
	while (token_type != NULL) 
	{
		NAtoms_type++;
		token_type = strtok(NULL, " \n");
	}
	if (NAtoms_coord != NAtoms_force) return -1;//NAtoms from coord.raw and force.raw are different
	if (NAtoms_coord % 3 != 0) return -2;//coord or force information of this frame is incomplete
	if (NAtoms_coord / 3 != NAtoms_type) return -3;//NAtoms of coord.raw or force.raw is different from NAtoms of type.raw
	return NAtoms_type;//return number of atoms of this frame
}

int parse_coord_force_type(char * coord, char * force, char * type, int N_Atoms_this_frame, frame_info_struct * frame_info)
{
	int i, j, k;
	char * token_coord, * token_force, * token_type;
	i = 0;
	token_coord = strtok(coord, " \n");
	printf_d("Coord check:\n");
	while (token_coord != NULL) 
	{
		sscanf(token_coord, "%lf", &(frame_info->coord[i/3][i%3]));
		printf_d("%lf ", frame_info->coord[i/3][i%3]);
		i++;
		if (i%3 == 0) printf_d("\n");
		token_coord = strtok(NULL, " \n");
	}
	i = 0;
	token_force = strtok(force, " \n");
	printf_d("Force check:\n");
	while (token_force != NULL) 
	{
		sscanf(token_force, "%lf", &(frame_info->force[i/3][i%3]));
		printf_d("%lf ", frame_info->force[i/3][i%3]);
		i++;
		if (i%3 == 0) printf_d("\n");
		token_force = strtok(NULL, " \n");
	}
	i = 0;
	token_type = strtok(type, " \n");
	printf_d("Type check:\n");
	while (token_type != NULL)
	{
		sscanf(token_type, "%d", &(frame_info->type[i]));
		printf_d("%d ", frame_info->type[i]);
		i++;
		token_type = strtok(NULL, " \n");
	}
	printf_d("\n");
}

void check_bin(int Nframes_tot)
{
	FILE * fp_in = fopen("./all_frame_info.bin.temp", "rb");
	int r;
	int i, j, k;
	frame_info_struct * frame_info = (frame_info_struct *)calloc(Nframes_tot, sizeof(frame_info_struct));
	fread(frame_info, sizeof(frame_info_struct), Nframes_tot, fp_in);
	srand(time(NULL));
	r = rand() % Nframes_tot;
	printf("Check saved frames. Randomly select one frame: %d (No.%d)\n", r, r + 1);
	printf("frame_info.index: %d\n", frame_info[r].index);
	printf("frame_info.N_Atoms: %d\n", frame_info[r].N_Atoms);
	for (i = 0; i <= 2; i++)
	{
		printf("box vector %d: ", i + 1);
		for (j = 0; j <= 2; j++)
		{
			printf("%lf ", frame_info[r].box[i][j]);
		}
		printf("\n");
	}
	printf("frame_info.type: \n");
	for (i = 0; i <= frame_info[r].N_Atoms - 1; i++)
	{
		printf("%d ", frame_info[r].type[i]);
	}
	printf("\n");
	printf("Coord:\n");
	for (i = 0; i <= frame_info[r].N_Atoms - 1; i++)
	{
		for (j = 0; j <= 2; j++)
		{
			printf("%.3lf ", frame_info[r].coord[i][j]);
		}
		printf("\n");
	}
	printf("frame_info.nergy: %.6lf\n", frame_info[r].energy);
	printf("frame_info.no_force: %d\n", frame_info[r].no_force);
	printf("Force:\n");
	for (i = 0; i <= frame_info[r].N_Atoms - 1; i++)
	{
		for (j = 0; j <= 2; j++)
		{
			printf("%+.6lf ", frame_info[r].force[i][j]);
		}
		printf("\n");
	}
}
