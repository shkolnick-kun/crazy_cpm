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
#include "ccmp.h"
#include <stdio.h>

/*=========================================================================
Merge sort based on:
    https://github.com/abranhe/mergesort.c/blob/master/mergesort.c
=========================================================================*/

// Merge the two half into a sorted data.
static inline void merge(int * m, int * l, int nl, int * r, int nr, int * val)
{
    int i = 0;
    int j = 0;
    int k = 0;

    while (i < nl && j < nr)
    {
        if (val[l[i]] <= val[r[j]])
        {
            m[k] = l[i++];
        }
        else
        {
            m[k] = r[j++];
        }
        k++;
    }

    // Copy the remaining elements of l[], if there are any
    while (i < nl)
    {
        m[k++] = l[i++];
    }

    // Copy the remaining elements of r[], if there are any
    while (j < nr)
    {
        m[k++] = r[j++];
    }
}

// l is for left index and r is right index of the
// sub-array of arr to be sorted

int * merge_sort(int * arr, int * tmp, int n, int * val)
{
    if (1 >= n)
    {
        return arr;
    }

    // Same as (l+r)/2, but avoids overflow for large l and h
    int m = n/2;
    // Sort first and second halves
    int * l = merge_sort(arr     ,tmp     ,m     ,val); /*Will return arr or tmp*/
    int * r = merge_sort(arr + m ,tmp + m ,n - m ,val);

    int * ret = (l != arr) ? arr : tmp;

    merge(ret, l, m, r, n - m, val);

    return ret;
}
