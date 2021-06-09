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

#include "ccmp.h"

/*=========================================================================
Merge sort based on:
    https://github.com/abranhe/mergesort.c/blob/master/mergesort.c
=========================================================================*/

// Merge the two half uint16_to a sorted data.
static inline void _merge(uint16_t * m, uint16_t * l, uint16_t nl, uint16_t * r, uint16_t nr, uint16_t * val)
{
    uint16_t i = 0;
    uint16_t j = 0;
    uint16_t k = 0;

    while ((i < nl) && (j < nr))
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
static inline uint16_t * _merge_sort(uint16_t * tmp, uint16_t * key, uint16_t * val, uint16_t n)
{
    if (1 == n)
    {
        return key;
    }

    uint16_t nl = n/2;
    uint16_t nr = n - nl;
    // Sort first and second halves
    uint16_t * l = _merge_sort(tmp      ,key      ,val ,nl); /*Will return arr or tmp*/
    uint16_t * r = _merge_sort(tmp + nl ,key + nl ,val ,nr);

    uint16_t * ret = (l != key) ? key : tmp;

    _merge(ret, l, nl, r, nr, val);

    return ret;
}

void ccpm_sort(uint16_t * tmp, uint16_t * key, uint16_t * val, uint16_t n)
{
    uint16_t * ms = _merge_sort(tmp, key, val, n);
    if (ms != key)
    {
        for (uint16_t i = 0; i < n; i++)
        {
            key[i] = tmp[i];
        }
    }
}

/*===========================================================================*/
typedef struct _ccpmMemStackSt ccpmMemStackSt;

struct _ccpmMemStackSt {
    ccpmMemStackSt * next;
    void * data;
};

void * _ccpm_mem_alloc(ccpmMemStackSt * item, ccpmMemStackSt ** stack, size_t sz)
{
    if ((!item) || (!stack))
    {
        return 0;
    }

    void * _data = malloc(sz);
    if (!_data)
    {
        return 0;
    }

    item->data = _data;
    item->next = *stack;
    *stack = item;
    return _data;
}

void _ccpm_mem_free(ccpmMemStackSt ** stack)
{
    while (*stack)
    {
        free((*stack)->data);
        *stack = (*stack)->next;
    }
}

#define CCPM_MEM_INIT() ccpmMemStackSt * mem_stack = 0

#define _CCPM_MEM_ALLOC(type, var, n, l)                                                     \
    ccpmMemStackSt CCPM_CAT(_item_,l);                                                       \
    type * var = (type *)_ccpm_mem_alloc(&CCPM_CAT(_item_,l), &mem_stack, n * sizeof(type)); \
    if (!var)                                                                                \
    {                                                                                        \
        printf("Not enough memory at %s, %d", __FILE__, l);                                  \
        _ccpm_mem_free(&mem_stack);                                                          \
        return CCMP_ENOMEM;                                                                  \
    }                                                                                        \
    (void)mem_stack

#define CCMP_MEM_ALLOC(type, var, n) _CCPM_MEM_ALLOC(type, var, n, __LINE__)
#define CCMP_MEM_FREE_ALL() _ccpm_mem_free(&mem_stack)

/*===========================================================================*/

ccpmResultEn ccpm_make_aoa(uint16_t * wrk_index, uint16_t * wrk_src, uint16_t * wrk_dst, uint16_t n_wrk, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk)
{
    ccpmResultEn ret = CCMP_OK;
    CCPM_MEM_INIT();
    CCMP_MEM_ALLOC(uint16_t * ,wrk_dep           ,n_wrk        ); /*Array of dependencies*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_dep_data      ,n_wrk * n_wrk); /*Array of dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_ndep          ,n_wrk        ); /*Number of dependencies*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_rem_dep       ,n_wrk        ); /*Number of remaining dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(bool *     ,wrk_dep_map       ,n_wrk        ); /*Work dependency map*/
    CCMP_MEM_ALLOC(bool       ,wrk_dep_map_data  ,n_wrk * n_wrk); /*Work dependency map*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(bool       ,wrk_started       ,n_wrk        ); /*Work started flag*/
    CCMP_MEM_ALLOC(bool       ,wrk_no_dummy      ,n_wrk        ); /*Work does not have dummy successor*/

    CCMP_MEM_ALLOC(uint16_t   ,chk_wrk           ,n_wrk        ); /*Work check list (array)*/
    /*-------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t * ,wrk_grp           ,n_wrk        ); /*Work groups*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_grp_data      ,n_wrk * n_wrk); /*Work groups*/
    /*--------------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_grp_pred      ,n_wrk        ); /*Work groups dependencies (indexes of first works with specific dependency lists)*/
    /*--------------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,no_dummy_works    ,n_wrk        ); /*Array of works with no dummies successors*/
    CCMP_MEM_ALLOC(uint16_t   ,dummy_idx         ,n_lnk        ); /*Dummy work index*/
    CCMP_MEM_ALLOC(bool       ,dummy_map         ,n_lnk        ); /*Dummy work map*/
    CCMP_MEM_ALLOC(bool       ,subgroup_map      ,n_lnk        ); /*Work subgroup map*/

    /*Temporary array for sortings*/
    CCMP_MEM_ALLOC(uint16_t   ,tmp  ,((n_wrk > n_lnk) ? n_wrk : n_lnk));

    for (uint16_t i = 0; i < n_wrk; i++)
    {
        wrk_src[i] = 0;
        wrk_dst[i] = 0;
    }

end:
    CCMP_MEM_FREE_ALL();
    return ret;
}
