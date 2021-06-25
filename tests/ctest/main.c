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
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>


#include "ccpm.h"

///*===========================================================================*/
//#define WBS              \
//X(1,  5,19              )\
//X(2,  1,4,16,17,10,12,14)\
//X(3,  2,18,6,7,20       )\
//X(4,  5,19              )\
//X(5                     )\
//X(6,  1,4,16,17,10,12,14)\
//X(7,  4,16,17,10,12,14  )\
//X(8,  6,7,20            )\
//X(9,  2,18,6,7,20       )\
//X(10, 5,19,11,13        )\
//X(11                    )\
//X(12, 5,19,11,13        )\
//X(13                    )\
//X(14, 5,19,11,13,15     )\
//X(15                    )\
//X(16, 5,19              )\
//X(17, 5,19              )\
//X(18, 1,4,16,17,10,12,14)\
//X(19                    )\
//X(20, 5,19,11,13,15     )\
//X(21, 3,8,9             )

/*===========================================================================*/
//#define WBS              \
//X(0,                    )\
//X(1,                    )\
//X(2,  0                 )\
//X(3,                    )\
//X(4,  3                 )\
//X(5,  3                 )\
//X(7,  5                 )\
//X(8,  1,5               )\
//X(9,  1,7               )

#define WBS              \
X(1,                    )\
X(2,                    )\
X(3,                    )\
X(4,                    )\
X(5,  1,2,3             )\
X(6,  2,3               )\
X(7,  3,4               )\
X(8,  1,6,7             )\
X(9,  5,6,7             )\
X(10, 3,6,7             )\
X(11, 6,8,9             )\
X(12, 7,8,9,10          )
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

uint16_t wrk_src[CCPM_WRK_NUM];
uint16_t wrk_dst[CCPM_WRK_NUM];
uint16_t wrk_pos[CCPM_WRK_NUM];
uint16_t tmp[CCPM_WRK_NUM];
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

static const uint16_t link_num[] = {
#   define X(id, ...) CCPM_ARRAY_SZ(_ccpm_dep_buf##id),
    WBS
#   undef X
};

/*===========================================================================*/
/*Pool of works*/
/*Number of works + number of links*/
#define X(id, ...) +CCPM_ARRAY_SZ(_ccpm_dep_buf##id)
uint16_t link_src[0 WBS];
uint16_t link_dst[0 WBS];
#undef X

/*===========================================================================*/
int main(void)
{
    printf("Link num: %d\n", (int)CCPM_ARRAY_SZ(link_src));
    uint16_t l = CCPM_ARRAY_SZ(link_src) - 1;
    for (uint16_t s = 0; s < CCPM_WRK_NUM; s++)
    {
        for (uint16_t d = 0; d < link_num[s]; d++)
        {
            link_src[l] = link_initializer[s][d];
            link_dst[l] = wrk_index[s];
            printf("Link[%d] = (%d, %d)\n", l, link_src[l], link_dst[l]);
            l--;
        }
    }

    uint16_t n_lnk = CCPM_ARRAY_SZ(link_src);

    ccpm_make_aoa(wrk_index, wrk_src, wrk_dst, CCPM_WRK_NUM, link_src, link_dst, &n_lnk);

    for (uint16_t i = 0; i < CCPM_WRK_NUM; i++)
    {
        wrk_pos[i] = i;
    }

    ccpm_sort(tmp, wrk_pos, wrk_dst, CCPM_WRK_NUM);
    ccpm_sort(tmp, wrk_pos, wrk_src, CCPM_WRK_NUM);

    printf("Scheduled works: \n");
    for (uint16_t i = 0; i < CCPM_WRK_NUM; i++)
    {
        printf("%5d: %5d %5d\n", wrk_index[wrk_pos[i]], wrk_src[wrk_pos[i]], wrk_dst[wrk_pos[i]]);
    }

    printf("Scheduled dummy works: \n");
    for (uint16_t i = 0; i < n_lnk; i++)
    {
        printf("%5d: %5d %5d\n", i + 1, link_src[i], link_dst[i]);
    }

    return 0;
}
