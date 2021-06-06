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

#define N 22

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

#define X CCPM_DEP_BUF
WBS
#undef X

ccpmWorkSt wrk_pool[N*2] =
{
#   define X CCPM_WRK_INITIALIZER
    WBS
#   undef X
};


int num[10] =
{
    1,3,6,5,8,7,9,6,2,0
};

/* сравнение двух целых */
int comp (const int *i, const int *j)
{
    return *i - *j;
}

int main(void)
{
    int i;

    printf("%d\n", sizeof(wrk_pool)/sizeof(wrk_pool[0]));
    printf("Original array: ");
    for (i=0; i<10; i++) printf("%d ",num[i]);
    printf ("\n");
    qsort(num, 10, sizeof (int), (int(*) (const void *, const void *)) comp);
    printf("Sorted array: ");
    for(i = 0; i <10; i++ ) printf("%d ", num[i]);
    return 0;
}
