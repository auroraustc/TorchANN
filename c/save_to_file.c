/*
2019.03.30 by Aurora. Contact:fanyi@mail.ustc.edu.cn

Save energy of each frame & force of each atom of each fram, all parameters, and sym_coord to file.

Return code:
    0: No errors.

*/

#include <stdio.h>
#include <stdlib.h>
#include "struct.h"

/*****************MACRO FOR DEBUG*****************/
#define DEBUG_SAV

#ifdef DEBUG_SAV
#define printf_d printf
#else
#define printf_d //
#endif
/***************MACRO FOR DEBUG END***************/

int save_to_file(frame_info_struct * frame_info, parameters_info_struct * parameters_info, void * sym_coord)
{
    return 0;
}