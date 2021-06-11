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
static int _qs_comp (const uint16_t *i, const uint16_t *j)
{
    return *i - *j;
}

/*===========================================================================*/
ccpmResultEn ccpm_make_aoa(uint16_t * wrk_index, uint16_t * wrk_src, uint16_t * wrk_dst, uint16_t n_wrk, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t *n_lnk)
{
    if ((!wrk_index) || (!wrk_src) || (!wrk_dst) || \
        (!n_wrk) || (!lnk_src) || (!lnk_dst) || (!n_lnk))
    {
        return CCMP_EINVAL;
    }

    ccpmResultEn ret = CCMP_OK;
    uint16_t _n_lnk = *n_lnk;
    uint16_t i;
    uint16_t ii;
    uint16_t j;
    uint16_t k;
    uint16_t l;
    uint16_t m;

    /*Check work index*/
    for (i = 0; i < n_wrk; i++)
    {
        for (j = i + 1; j < n_wrk; j++)
        {
            if (wrk_index[i] == wrk_index[j])
            {
                printf("Work indexes are not unique: %d, %d", i, j);
                return CCMP_EINVAL;
            }
        }
    }

    /*Check links*/
    for (i = 0; i < _n_lnk; i++)
    {
        for (j = i + 1; j < _n_lnk; j++)
        {
            if ((lnk_src[i] == lnk_src[j]) && (lnk_dst[i] == lnk_dst[j]))
            {
                printf("Links are not unique: %d, %d", i, j);
                return CCMP_EINVAL;
            }
        }
    }

    CCPM_MEM_INIT();
    CCMP_MEM_ALLOC(uint16_t   ,wrk_pos           ,n_wrk         ); /*Works positions in sorted lists*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_ndep          ,n_wrk         ); /*Number of dependencies*/
    CCMP_MEM_ALLOC(uint16_t * ,wrk_dep           ,n_wrk         ); /*Array of dependencies*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_dep_data      ,_n_lnk         ); /*Array of dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(bool       ,wrk_dep_map_data  ,n_wrk * n_wrk); /*Work dependency map*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_rem_dep       ,n_wrk         ); /*Number of remaining dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(bool       ,wrk_started       ,n_wrk         ); /*Work started flag*/
    CCMP_MEM_ALLOC(bool       ,wrk_no_dummy      ,n_wrk         ); /*Work does not have dummy successor*/

    CCMP_MEM_ALLOC(uint16_t   ,chk_wrk           ,n_wrk         ); /*Work check list (array)*/
    /*-------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,grp_sz            ,n_wrk         ); /*Work group sizes*/
    CCMP_MEM_ALLOC(uint16_t   ,grp_data          ,n_wrk * n_wrk ); /*Work groups (first member of the group has dependency list for the group)*/
    /*--------------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,no_dummy_works    ,n_wrk         ); /*Array of works with no dummies successors*/
    CCMP_MEM_ALLOC(uint16_t   ,dummy_idx         ,_n_lnk        ); /*Dummy work index*/
    CCMP_MEM_ALLOC(bool       ,dummy_map         ,n_wrk * 2     ); /*Dummy work map*/
    CCMP_MEM_ALLOC(bool       ,subgroup_map      ,n_wrk * 2     ); /*Work subgroup map*/

    /*Temporary array for sortings*/
    CCMP_MEM_ALLOC(uint16_t   ,tmp  ,((n_wrk > _n_lnk) ? n_wrk : _n_lnk));

    for (i = 0; i < n_wrk; i++)
    {
        wrk_src[i]  = 0;
        wrk_dst[i]  = 0;

        wrk_dep[i]  = 0;
        wrk_ndep[i] = 0;
        for (j = 0; j < n_wrk; j++)
        {
            wrk_dep_map_data[n_wrk * i + j] = false;
        }
    }

    printf("Translate work indexes to work array positions...\n");
    for (l = 0; l < _n_lnk; l++)
    {
        for (i = 0; i < n_wrk; i++)
        {
            if (lnk_src[l] == wrk_index[i])
            {
                lnk_src[l] = i;
            }
            if (lnk_dst[l] == wrk_index[i])
            {
                lnk_dst[l] = i;
            }
        }
    }

    /*Build dependencys lists and maps*/
    printf("Sort links by dst...\n");
    for (i = 0; i < _n_lnk; i++)
    {
        dummy_idx[i] = i;
    }
    ccpm_sort(tmp, dummy_idx, lnk_dst, _n_lnk);

    printf("Populate dependencies data...\n");
    for (l = 0; l < _n_lnk; l++)
    {
        /*Populate depenency data*/
        i = lnk_src[dummy_idx[l]];
        wrk_dep_data[l] = i;

        /*Populate dependency arrays*/
        j = lnk_dst[dummy_idx[l]];
        if (!wrk_dep[j])
        {
            wrk_dep[j] = wrk_dep_data + l;
        }

        /*Count dependencies in arrays*/
        wrk_ndep[j]++;

        /*Populate dependency maps*/
        wrk_dep_map_data[n_wrk * j + i] = true;
        printf("link[%d] = [%d, %d]\n", l, wrk_index[i], wrk_index[j]);
    }

    printf("Dependency map:\n");
    for (i = 0; i < n_wrk; i++)
    {
        printf("%5d: ", wrk_index[i]);
        for (j = 0; j < n_wrk; j++)
        {
            printf("%d  ", wrk_dep_map_data[n_wrk * i + j]);
        }
        printf("\n");
    }

    printf("Sort dependency arrays: \n");
    for (i = 0; i < n_wrk; i++)
    {
        k = wrk_ndep[i];
        /*Process dependency data*/
        printf("%5d: n=%d dep=[", wrk_index[i], k);

        qsort(wrk_dep[i], k, sizeof(uint16_t), (int(*) (const void *, const void *)) _qs_comp);
        for (j = 0; j < k; j++)
        {
            printf("%5d", wrk_index[wrk_dep[i][j]]);
        }
        printf(" ]\n");

        /*Initiate other work properties*/
        wrk_rem_dep[i]  = k;
        wrk_started[i]  = false;
        wrk_no_dummy[i] = true;
    }

    printf("Sort works by ndep...\n");
    for (i = 0; i < n_wrk; i++)
    {
        wrk_pos[i] = i;
    }
    ccpm_sort(tmp, wrk_pos, wrk_ndep, n_wrk);

    printf("Collect started works...\n");
    uint16_t evt_id = 1;
    uint16_t n_chk_wrk = 0;
    uint16_t n_dummys = 0;

    for (ii = 0; ii < n_wrk; ii++)
    {
        i = wrk_pos[ii];

        if (wrk_rem_dep[i])
        {
            break;
        }

        wrk_started[i] = true;
        wrk_src[i] = evt_id;
        chk_wrk[n_chk_wrk++] = i; /*Append work to chk_wrk*/
        printf("%5d",  wrk_index[i]);
    }
    evt_id++;

    printf("\nProcess started works...\n");
    for (j = 0; j < n_chk_wrk; j++)
    {
        uint16_t n_grp = 0;
        for (ii = 0; ii < n_wrk; ii++)
        {
            i = wrk_pos[ii];
            if (wrk_started[i])
            {
                continue;
            }

            if (wrk_dep_map_data[n_wrk * i + chk_wrk[j]])
            {
                 wrk_rem_dep[i]--;
            }

            if (0 == wrk_rem_dep[i])
            {
                /*
                Some work gets started.
                Later we will do <wrk_started[i] = true;> so we will never repeat next steps:
                */
                /*Check if a work has common dependencies with some group of started works*/
                bool is_in_pred = false;
                for (k = 0; k < n_grp; k++)
                {
                    /*First work in a group is used to get groups dependency list(array)*/
                    uint16_t pred = grp_data[n_wrk * k];
                    if (wrk_ndep[pred] != wrk_ndep[i])
                    {
                        continue;
                    }

                    /*Compare dependency lists*/
                    bool is_equal = true;
                    for (l = 0; l < wrk_ndep[pred]; l++)
                    {
                        if (wrk_dep[pred][l] != wrk_dep[i][l])
                        {
                            is_equal = false;
                            break;
                        }
                    }

                    if (is_equal)
                    {
                        is_in_pred = true;
                        break;
                    }
                }

                if (is_in_pred)
                {
                    /*Append a work to some group*/
                    grp_data[n_wrk * k + grp_sz[k]++] = i;
                }
                else
                {
                    /*Create new group*/
                    grp_sz[n_grp] = 1;
                    grp_data[n_wrk * n_grp++] = i;
                }
            }
        }
        printf("%5d: Current work is: %d. Found %d gpoups of started works...\n", j, wrk_index[chk_wrk[j]], n_grp);
        if (!n_grp)
        {
            continue;
        }

        /*Process gpoups*/
        for (k = 0; k < n_grp; k++)
        {
            printf("Process group %d\n", k);
            uint16_t n_added_dummys = 0;
            uint16_t n_no_dummys    = 0;

            for (l = 0; l < n_wrk * 2; l++)
            {
                dummy_map[l]    = false;
                subgroup_map[l] = false;
            }

            /*Process groups dependency list(array)*/
            uint16_t pred = grp_data[n_wrk * k];
            uint16_t ndep = wrk_ndep[pred];
            uint16_t *dep = wrk_dep[pred];

            printf("Dependencies:\n");
            for (l = 0; l < ndep; l++)
            {
                i = dep[l];
                printf("%5d: %5d %5d %d\n", wrk_index[i], wrk_src[i], wrk_dst[i], wrk_no_dummy[i]);
                if (wrk_dst[i])
                {
                    if (wrk_no_dummy[i])
                    {
                        if (!dummy_map[wrk_dst[i]])
                        {
                            dummy_map[wrk_dst[i]] = true;
                            /*Append a dummy work*/
                            lnk_src[n_dummys + n_added_dummys++] = wrk_dst[i];
                            //printf("Added dummy 1: %d %d\n", n_dummys + n_added_dummys, wrk_dst[i]);
                        }
                    }
                }
                else if (!subgroup_map[wrk_src[i]])
                {
                    subgroup_map[wrk_src[i]] = true;
                    no_dummy_works[n_no_dummys++] = i;
                }
                else
                {
                    wrk_no_dummy[i] = false;
                    wrk_dst[i] = evt_id;
                    /*Append a dummy work*/
                    lnk_src[n_dummys + n_added_dummys++] = evt_id++;
                    //printf("Added dummy 2: %d %d\n", n_dummys + n_added_dummys, wrk_dst[i]);
                }
            }
            printf("\n");

            /*Finalize a group list processing*/
            printf("Group works: ");
            for (l = 0; l < grp_sz[k]; l++)
            {
                i = grp_data[n_wrk * k + l];
                wrk_src[i]     = evt_id;
                wrk_started[i] = true;
                printf("%5d", wrk_index[i]);

                /*Add this work to work check list(array)*/
                chk_wrk[n_chk_wrk++] = i;
            }
            printf("\n");

            printf("Dummy works:\n");
            for (l = 0; l < n_added_dummys; l++)
            {
                lnk_dst[n_dummys + l] = evt_id;
                printf("%d %d\n", lnk_src[n_dummys + l], lnk_dst[n_dummys + l]);
            }
            n_dummys += n_added_dummys;
            printf("\n");

            printf("No dummy works:");
            for (l = 0; l < n_no_dummys; l++)
            {
                i = no_dummy_works[l];
                printf("%5d", wrk_index[i]);
                wrk_dst[i] = evt_id;
            }
            printf("\n");

            evt_id++;
        }
    }

    for (i = 0; i < n_wrk; i++)
    {
        /*Loop detection*/
        if (!wrk_started[i])
        {
            ret = CCMP_ELOOP;
            goto end;
        }

        /*Finish last works*/
        if (!wrk_dst[i])
        {
            wrk_dst[i] = evt_id;
        }
    }

    printf("Removing redundant dummies...\n");
    /*Sort dummys*/
    for (l = 0; l < n_dummys; l++)
    {
        dummy_idx[l] = l;
    }
    ccpm_sort(tmp, dummy_idx, lnk_src, n_dummys);

    /*Mark redundant dummies*/
    for (l = 0; l < n_dummys; l++)
    {
        tmp[l] = 1; /*Will use tmp for marks*/
    }
    for (k = 0; k < n_dummys; k++)
    {
        i = dummy_idx[k];
        uint16_t src = lnk_src[i];
        uint16_t pvt = lnk_dst[i];
        for (l = k + 1; l < n_dummys; l++)
        {
            j = dummy_idx[l];
            if (lnk_src[j] != src)
            {
                break;
            }
            uint16_t cur = lnk_dst[j];
            for (m = l + 1; m < n_dummys; m++)
            {
                ii = dummy_idx[m];
                if ((lnk_src[ii] == pvt) && (lnk_dst[ii] == cur))
                {
                    tmp[j] = 0; /*Must drop this dummy*/
                    break;
                }
            }
        }
    }

    /*Drop redundant dumies*/
    _n_lnk = 0;
    for (l = 0; l < n_dummys; l++)
    {
        if (tmp[l])
        {
            wrk_dep_data[_n_lnk  ] = lnk_src[l];
            dummy_idx   [_n_lnk++] = lnk_dst[l];
        }
    }

    printf("Dummy works:\n");
    for (l = 0; l < _n_lnk; l++)
    {
        lnk_src[l] = wrk_dep_data[l];
        lnk_dst[l] = dummy_idx   [l];
        printf("%d %d\n", lnk_src[l], lnk_dst[l]);
    }
    *n_lnk = _n_lnk;
end:
    CCMP_MEM_FREE_ALL();
    return ret;
}
