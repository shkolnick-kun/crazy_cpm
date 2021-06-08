/**************************************************************************
    CrazyCPM
    Copyright (C) 2021 anonimous

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Please contact with me by E-mail: shkolnick.kun@gmail.com
**************************************************************************/
#include <stdio.h>
#include <stdlib.h>

#include "ccmp.h"

/*===========================================================================*/
#define WBS              \
X(0                     )\
X(1,  5,19              )\
X(2,  1,4,16,17,10,12,14)\
X(3,  2,18,6,7,20       )\
X(4,  5,19              )\
X(5                     )\
X(6,  1,4,16,17,10,12,14)\
X(7,  4,16,17,10,12,14  )\
X(8,  6,7,20            )\
X(9,  2,18,6,7,20       )\
X(10, 5,19,11,13        )\
X(11                    )\
X(12, 5,19,11,13        )\
X(13                    )\
X(14, 5,19,11,13,15     )\
X(15                    )\
X(16, 5,19              )\
X(17, 5,19              )\
X(18, 1,4,16,17,10,12,14)\
X(19                    )\
X(20, 5,19,11,13,15     )\
X(21, 3,8,9             )

/*===========================================================================*/
/*Work index*/
uint16_t wrk_index[] = {
#   define X(id, ...) id,
    WBS
#   undef X
};

/*===========================================================================*/
/*Compile time calculation of CCPM_WRK_NUM and work_pool size*/
typedef enum
{
#   define X(id, ...) CCPM_WRK_##id,
    WBS
#   undef X
    CCPM_WRK_NUM /*Number of works*/
}ccpmWrkEn;

/*===========================================================================*/
/*AoN links will be initialized with these info*/
#define X CCPM_DEP_BUF
WBS
#undef X


static const uint16_t * link_initializer[] = {
#   define X(id, ...) _ccpm_dep_buf##id,
    WBS
#   undef X
};

/*===========================================================================*/
/*Pool of works*/
/*Number of works + number of links*/
#define X(id, ...) +CCPM_ARRAY_SZ(_ccpm_dep_buf##id)
ccpmWorkSt wrk_pool[CCPM_WRK_NUM WBS];
#undef X

/*===========================================================================*/

int idx[11] =
{
    0,1,2,3,4,5,6,7,8,9,10
};


int num[11] =
{
    1,3,6,5,8,7,9,6,2,0,15
};

int tmp[11];

extern int * merge_sort(int * arr, int * tmp, int n, int * val);

int main(void)
{
    int i;

    printf("%ld\n", CCPM_ARRAY_SZ(link_initializer));
    printf("Original array: ");
    for (i=0; i<11; i++) printf("%d ",num[idx[i]]);
    printf ("\n");

    int * s = merge_sort(idx, tmp, 11, num);

    printf("Sorted array: %x %x %x \n", s, idx, tmp);
    for(i = 0; i <11; i++ ) printf("%d ", num[s[i]]);
    return 0;
}
