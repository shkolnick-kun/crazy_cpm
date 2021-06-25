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

#ifndef CCPM_LOG_PRINTF
#define CCPM_LOG_PRINTF(...)
#endif/*CCPM_LOG_PRINTF*/

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
    uint16_t * l = _merge_sort(tmp,      key,      val, nl);              /*Will return arr or tmp*/
    uint16_t * r = _merge_sort(tmp + nl, key + nl, val, nr);

    uint16_t * ret = (l != key) ? key : tmp;

    _merge(ret, l, nl, r, nr, val);

    return ret;
}

void ccpm_sort(uint16_t * tmp, uint16_t * key, uint16_t * val, uint16_t n)
{
    if (!n)
    {
        return;
    }
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

struct _ccpmMemStackSt
{
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
        CCPM_LOG_PRINTF("Not enough memory at %s, %d", __FILE__, l);                         \
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
static inline bool _ccpm_lookup_wrk_pos(uint16_t * link, uint16_t * wrk_id, uint16_t n_wrk)
{
    for (uint16_t i = 0; i < n_wrk; i++)
    {
        if (0 == *link - wrk_id[i])
        {
            *link = i;
            return true;
        }
    }
    return false;
}

/*===========================================================================*/
ccpmResultEn ccpm_make_aoa(uint16_t * wrk_id, uint16_t * wrk_src, uint16_t * wrk_dst, uint16_t n_wrk, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t *n_lnk)
{
    if ((!wrk_id) || (!wrk_src) || (!wrk_dst) || \
            (!n_wrk) || (!lnk_src) || (!lnk_dst) || (!n_lnk))
    {
        CCPM_LOG_PRINTF("ERROR: Incorrect input parameters.\n");
        return CCMP_EINVAL;
    }

    ccpmResultEn ret = CCMP_OK;
    uint16_t _n_lnk = *n_lnk;
    uint16_t evt_id = 1;
    uint16_t sg_id = 1;
    uint16_t n_chk_wrk = 0;
    uint16_t n_dummys = 0;
    uint16_t i;
    uint16_t j;
    uint16_t k;
    uint16_t l;
    uint16_t m;
    uint16_t p;

    /*Check work index*/
    for (i = 0; i < n_wrk; i++)
    {
        for (j = i + 1; j < n_wrk; j++)
        {
            if (wrk_id[i] == wrk_id[j])
            {
                CCPM_LOG_PRINTF("ERROR: Work ids are not unique: %d, %d\n", i, j);
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
                CCPM_LOG_PRINTF("ERROR: Links are not unique: %d, %d\n", i, j);
                return CCMP_EINVAL;
            }
        }
    }

    CCPM_MEM_INIT();
    CCMP_MEM_ALLOC(uint16_t   ,wrk_pos      ,n_wrk        ); /*Works positions in sorted lists*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_ndep     ,n_wrk        ); /*Number of dependencies*/
    CCMP_MEM_ALLOC(uint16_t * ,wrk_dep      ,n_wrk        ); /*Array of dependencies*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_dep_data ,_n_lnk       ); /*Array of dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(bool       ,wrk_dep_map  ,n_wrk * n_wrk); /*Work dependency map*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_rem_dep  ,n_wrk        ); /*Number of remaining dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(bool       ,wrk_started  ,n_wrk        ); /*Work started flag*/
    CCMP_MEM_ALLOC(uint16_t   ,wrk_sg_id    ,n_wrk        ); /*Work does not have dummy successor*/

    CCMP_MEM_ALLOC(uint16_t   ,chk_wrk      ,n_wrk        ); /*Work check list (array)*/
    /*-------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,grp_sz       ,n_wrk        ); /*Work group sizes*/
    CCMP_MEM_ALLOC(uint16_t   ,grp_data     ,n_wrk * n_wrk); /*Work groups (first member of the group has dependency list for the group)*/
    /*--------------------------------------------------------------------------------------*/
    CCMP_MEM_ALLOC(uint16_t   ,new_sg_wrk   ,n_wrk        ); /*Array of works with no dummies successors*/
    CCMP_MEM_ALLOC(uint16_t   ,dummy_pos    ,_n_lnk       ); /*Dummy work index*/
    CCMP_MEM_ALLOC(int16_t    ,old_sg_map   ,n_wrk * 2    ); /*Dummy work map*/
    CCMP_MEM_ALLOC(int16_t    ,new_sg_map   ,n_wrk * 2    ); /*Work subgroup map*/

    /*Temporary array for sortings*/
    CCMP_MEM_ALLOC(uint16_t,tmp,((n_wrk > _n_lnk) ? n_wrk : _n_lnk));

    for (i = 0; i < n_wrk; i++)
    {
        wrk_src[i]  = 0;
        wrk_dst[i]  = 0;

        wrk_dep [i] = 0;
        wrk_ndep[i] = 0;
        for (j = 0; j < n_wrk; j++)
        {
            wrk_dep_map[n_wrk * i + j] = false;
        }
    }

    CCPM_LOG_PRINTF("Translate work indexes to work array positions...\n");
    for (i = 0; i < _n_lnk; i++)
    {
        CCPM_LOG_PRINTF("L[%d]=(%d,%d)->", i, lnk_src[i], lnk_dst[i]);

        bool found_src = _ccpm_lookup_wrk_pos(lnk_src + i, wrk_id, n_wrk);
        bool found_dst = _ccpm_lookup_wrk_pos(lnk_dst + i, wrk_id, n_wrk);

        CCPM_LOG_PRINTF("[%d,%d]=(%d,%d)\n", lnk_src[i], lnk_dst[i], wrk_id[lnk_src[i]], wrk_id[lnk_dst[i]]);
        if (!found_src || !found_dst)
        {
            CCPM_LOG_PRINTF("ERROR: Invalid work id in link[%d] = (%d, %d)\n", i, lnk_src[i], lnk_dst[i]);
            ret = CCMP_EINVAL;
            goto end;
        }
    }

    /*Build dependencys lists and maps*/
    CCPM_LOG_PRINTF("Sort links by dst...\n");
    for (i = 0; i < _n_lnk; i++)
    {
        dummy_pos[i] = i;
    }
    ccpm_sort(tmp, dummy_pos, lnk_dst, _n_lnk);

    CCPM_LOG_PRINTF("Populate dependencies data...\n");
    for (l = 0; l < _n_lnk; l++)
    {
        /*Populate depenency data*/
        i = lnk_src[dummy_pos[l]];
        wrk_dep_data[l] = i;

        /*Populate dependency arrays*/
        j = lnk_dst[dummy_pos[l]];
        if (!wrk_dep[j])
        {
            wrk_dep[j] = wrk_dep_data + l;
        }

        /*Count dependencies in arrays*/
        wrk_ndep[j]++;

        /*Populate dependency maps*/
        wrk_dep_map[n_wrk * j + i] = true;
        CCPM_LOG_PRINTF("link[%d] = [%d, %d]\n", l, wrk_id[i], wrk_id[j]);
    }

    CCPM_LOG_PRINTF("Remove redundant dependencies\nDependency map:\n");
    for (i = 0; i < n_wrk; i++)
    {
        CCPM_LOG_PRINTF("%5d: ", wrk_id[i]);
        for (j = 0; j < n_wrk; j++)
        {
            CCPM_LOG_PRINTF("%d  ", wrk_dep_map[n_wrk * i + j]);
            for (k = 0; k < n_wrk; k++)
            {
                if (k == j)
                {
                    continue;
                }
                if (wrk_dep_map[n_wrk * i + j] && wrk_dep_map[n_wrk * i + k] && wrk_dep_map[n_wrk * k + j])
                {
                    wrk_dep_map[n_wrk * i + j] = false;
                }
            }
        }
        CCPM_LOG_PRINTF("\n");
    }

    CCPM_LOG_PRINTF("Dependency arrays:\n");
    for (i = 0; i < n_wrk; i++)
    {
        k = wrk_ndep[i];
        CCPM_LOG_PRINTF("%5d: n=%d dep=[", wrk_id[i], k);
        for (j = 0; j < k; j++)
        {
            CCPM_LOG_PRINTF("%5d", wrk_id[wrk_dep[i][j]]);
        }
        CCPM_LOG_PRINTF(" ]\n");

        wrk_ndep[i] = 0;
        for (j = 0; j < n_wrk; j++)
        {
            if (wrk_dep_map[n_wrk * i + j])
            {
                wrk_dep[i][wrk_ndep[i]++] = j;
            }
        }

    }

    CCPM_LOG_PRINTF("Sorted optimized dependency arrays:\n");
    for (i = 0; i < n_wrk; i++)
    {
        k = wrk_ndep[i];
        /*Process dependency data*/
        CCPM_LOG_PRINTF("%5d: n=%d dep=[", wrk_id[i], k);

        qsort(wrk_dep[i], k, sizeof(uint16_t), (int(*) (const void *, const void *)) _qs_comp);
        for (j = 0; j < k; j++)
        {
            CCPM_LOG_PRINTF("%5d", wrk_id[wrk_dep[i][j]]);
        }
        CCPM_LOG_PRINTF(" ]\n");

        /*Initiate other work properties*/
        wrk_rem_dep[i]  = k;
        wrk_started[i]  = false;
        wrk_sg_id[i]    = 0;
    }

    CCPM_LOG_PRINTF("Sort works by ndep...\n");
    for (i = 0; i < n_wrk; i++)
    {
        wrk_pos[i] = i;
    }
    ccpm_sort(tmp, wrk_pos, wrk_ndep, n_wrk);

    CCPM_LOG_PRINTF("Collect started works...\n");
    for (p = 0; p < n_wrk; p++)
    {
        i = wrk_pos[p];

        if (wrk_rem_dep[i])
        {
            break;
        }

        wrk_started[i] = true;
        wrk_src[i] = evt_id;
        chk_wrk[n_chk_wrk++] = i; /*Append work to chk_wrk*/
        CCPM_LOG_PRINTF("%5d",  wrk_id[i]);
    }
    evt_id++;

    CCPM_LOG_PRINTF("\nProcess started works...\n");
    for (j = 0; j < n_chk_wrk; j++)
    {
        /*Find new started works and their dependencies*/
        uint16_t n_grp = 0;
        for (p = 0; p < n_wrk; p++)
        {
            i = wrk_pos[p];
            if (wrk_started[i])
            {
                continue;
            }

            if (wrk_dep_map[n_wrk * i + chk_wrk[j]])
            {
                wrk_rem_dep[i]--;
            }

            if (0 == wrk_rem_dep[i])
            {
                /*
                Some work gets started.

                Later we will do <wrk_started[i] = true;> as a result, this block of code
                will be executed exactly once for each started work.

                So overall time complexity of ccpm_make_aoa is O(n^3).
                */
                /*Check if a work has common dependencies with some group of started works*/
                bool is_in_pred = false;
                for (k = 0; k < n_grp; k++)
                {
                    /*First work in a group is used to get groups dependency list(array)*/
                    uint16_t pred = grp_data[n_wrk * k];

                    /*Compare dependency lists*/
                    if (wrk_ndep[pred] != wrk_ndep[i])
                    {
                        continue;
                    }

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
                    /*Create a new group*/
                    grp_sz[n_grp] = 1;
                    grp_data[n_wrk * n_grp++] = i;
                }
            }
        }
        CCPM_LOG_PRINTF("%5d: Current work in check list is: %d. Found %d groups of started works...\n", j, wrk_id[chk_wrk[j]], n_grp);

        /*Check if we found some started works*/
        if (!n_grp)
        {
            continue;
        }

        /*Process groups*/
        for (k = 0; k < n_grp; k++)
        {
            CCPM_LOG_PRINTF("Process group %d\n", k);
            uint16_t n_added_dummys = 0;
            uint16_t n_new_sg       = 0;
            uint16_t n_old_sg       = 0;

            for (l = 0; l < n_wrk * 2; l++)
            {
                old_sg_map[l] = -1;
                new_sg_map[l] = -1;
            }

            /*Process groups dependency list(array)*/
            uint16_t pred = grp_data[n_wrk * k];
            uint16_t ndep = wrk_ndep[pred];
            uint16_t *dep = wrk_dep [pred];

            CCPM_LOG_PRINTF("Dependencies:\n");
            for (l = 0; l < ndep; l++)
            {
                i = dep[l];
                CCPM_LOG_PRINTF("%5d: %5d %5d %5d\n", wrk_id[i], wrk_src[i], wrk_dst[i], wrk_sg_id[i]);
                if (wrk_dst[i])
                {
                    if (0 == wrk_sg_id[i])
                    {
                        ret = CCMP_EUNK;
                        goto end;
                    }

                    if (old_sg_map[wrk_sg_id[i]] < 0)
                    {
                        /*Remind an old subgroup*/
                        old_sg_map[wrk_sg_id[i]] = n_old_sg;
                        tmp[n_old_sg++] = i; /*Old subgroup works*/

                        /*Append a dummy work*/
                        //lnk_src[n_dummys + n_added_dummys++] = wrk_dst[i];
                        //CCPM_LOG_PRINTF("Added dummy 1: %d %d\n", n_dummys + n_added_dummys, wrk_dst[i]);
                    }
                    else if (wrk_dst[i] > wrk_dst[tmp[old_sg_map[wrk_sg_id[i]]]])
                    {
                        /*Find a work with bigest dst in an old subgroup*/
                        tmp[old_sg_map[wrk_sg_id[i]]] = i;
                    }
                }
                else if (new_sg_map[wrk_src[i]] < 0)
                {
                    /*Create new subgroup*/
                    wrk_sg_id[i] = sg_id++;
                    new_sg_map[wrk_src[i]] = i;
                    new_sg_wrk[n_new_sg++] = i;
                }
                else
                {
                    /*Assign a work to some old subgroup*/
                    wrk_sg_id[i] = wrk_sg_id[new_sg_map[wrk_src[i]]];
                    wrk_dst[i]   = evt_id;

                    /*Append a dummy work*/
                    lnk_src[n_dummys + n_added_dummys++] = evt_id++;
                    //CCPM_LOG_PRINTF("Added dummy 2: %d %d\n", n_dummys + n_added_dummys, wrk_dst[i]);
                }
            }
            CCPM_LOG_PRINTF("\n");

            if (n_old_sg)
            {
                CCPM_LOG_PRINTF("Process old subgroups:\n");
                for (l = 0; l < n_wrk * 2; l++)
                {
                    old_sg_map[l] = -1;
                }

                for (l = 0; l < n_old_sg; l++)
                {
                    i = tmp[l];
                    if (old_sg_map[wrk_dst[i]] < 0)
                    {
                        old_sg_map[wrk_dst[i]] = 1;
                        lnk_src[n_dummys + n_added_dummys++] = wrk_dst[i];
                        CCPM_LOG_PRINTF("Added dummy: %d %d\n", n_dummys + n_added_dummys, wrk_dst[i]);
                    }
                }
            }

            /*Finalize a group list processing*/
            CCPM_LOG_PRINTF("Group works: ");
            for (l = 0; l < grp_sz[k]; l++)
            {
                i = grp_data[n_wrk * k + l];
                wrk_src[i]     = evt_id;
                wrk_started[i] = true;
                CCPM_LOG_PRINTF("%5d", wrk_id[i]);

                /*Add this work to work check list(array)*/
                chk_wrk[n_chk_wrk++] = i;
            }
            CCPM_LOG_PRINTF("\n");

            CCPM_LOG_PRINTF("Dummy works:\n");
            for (l = 0; l < n_added_dummys; l++)
            {
                lnk_dst[n_dummys + l] = evt_id;
                CCPM_LOG_PRINTF("%d %d\n", lnk_src[n_dummys + l], lnk_dst[n_dummys + l]);
            }
            n_dummys += n_added_dummys;
            CCPM_LOG_PRINTF("\n");

            CCPM_LOG_PRINTF("No dummy works:");
            for (l = 0; l < n_new_sg; l++)
            {
                i = new_sg_wrk[l];
                CCPM_LOG_PRINTF("%5d", wrk_id[i]);
                wrk_dst[i] = evt_id;
            }
            CCPM_LOG_PRINTF("\n");

            evt_id++;
        }
    }

    for (i = 0; i < n_wrk; i++)
    {
        /*Loop detection*/
        if (!wrk_started[i])
        {
            CCPM_LOG_PRINTF("ERROR: Found a loop, check work: %5d\n", wrk_id[i]);
            ret = CCMP_ELOOP;
            goto end;
        }

        /*Finish last works*/
        if (!wrk_dst[i])
        {
            wrk_dst[i] = evt_id;
        }
    }

    if(!n_dummys)
    {
        goto end;
    }

    CCPM_LOG_PRINTF("Removing redundant dummies...\n");
    /*Dummies are sorted by "dst" now, sort dummies by "src"*/
    CCPM_LOG_PRINTF("Unsorted dummies:\n");
    for (l = 0; l < n_dummys; l++)
    {
        dummy_pos[l] = l;
        CCPM_LOG_PRINTF("%d %d\n", lnk_src[l], lnk_dst[l]);
    }
    ccpm_sort(tmp, dummy_pos, lnk_src, n_dummys);
    /*Dummies are sorted by "src" and "dst" now as merge sort is stable*/

    /*Mark redundant dummies*/
    CCPM_LOG_PRINTF("Sorted dummies:\n");
    for (l = 0; l < n_dummys; l++)
    {
        CCPM_LOG_PRINTF("%d %d\n", lnk_src[dummy_pos[l]], lnk_dst[dummy_pos[l]]);
        tmp[l] = 1; /*Will use tmp for marks*/
    }
    for (k = 0; k < n_dummys; k++)
    {
        i = dummy_pos[k];
        uint16_t src = lnk_src[i];
        uint16_t pvt = lnk_dst[i];
        for (l = k + 1; l < n_dummys; l++)
        {
            j = dummy_pos[l];
            if (lnk_src[j] != src)
            {
                break;
            }

            uint16_t cur = lnk_dst[j];
            for (m = l + 1; m < n_dummys; m++)
            {
                p = dummy_pos[m];

                if (lnk_src[p] > pvt)
                {
                    break;
                }

                if ((lnk_src[p] == pvt) && (lnk_dst[p] == cur))
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
            dummy_pos   [_n_lnk++] = lnk_dst[l];
        }
    }

    CCPM_LOG_PRINTF("Dummy works:\n");
    for (l = 0; l < _n_lnk; l++)
    {
        lnk_src[l] = wrk_dep_data[l];
        lnk_dst[l] = dummy_pos   [l];
        CCPM_LOG_PRINTF("%d %d\n", lnk_src[l], lnk_dst[l]);
    }
    *n_lnk = _n_lnk;
end:
    CCMP_MEM_FREE_ALL();
    return ret;
}
