/**************************************************************************
    CrazyCPM
    Copyright (C) 2025 anonimous

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

/*===========================================================================*/
#ifdef __GNUC__
#   define CCPM_UNLIKELY(x) __builtin_expect((x), 0)
#else /*__GNUC__*/
#   define CCPM_UNLIKELY(x) (x)
#endif/*__GNUC__*/

/*===========================================================================*/
#ifndef CCPM_CFG_PRINTF
#   define CCPM_LOG_PRINTF(...) do {} while (0)
#   define CCPM_ERR(...)        fprintf(stderr, __VA_ARGS__)
#else /*CCPM_CFG_PRINTF*/
#   define CCPM_LOG_PRINTF(...) CCPM_CFG_PRINTF(__VA_ARGS__)
#   define CCPM_ERR(...)        CCPM_CFG_PRINTF(__VA_ARGS__)
#endif/*CCPM_CFG_PRINTF*/

/*===========================================================================*/
#define _CCPM_CHECK_RETURN(cond, err, file, func, line)                    \
do {                                                                       \
    if (CCPM_UNLIKELY(!(cond)))                                            \
    {                                                                      \
        CCPM_ERR("CCPM:The expression (%s) is false in \n function: %s",   \
                 #cond, func);                                             \
        CCPM_ERR("\n file: %s\n line: %d\n will return: %s\n",             \
                 file, line, #err);                                        \
        return err;                                                        \
    }                                                                      \
} while (0)

#define CCPM_CHECK_RETURN(cond, err) _CCPM_CHECK_RETURN(cond, err, __FILE__, __func__, __LINE__)

/*===========================================================================*/
ccpmResultEn ccpm_check_act_ids(uint16_t * act_id, uint16_t n_act)
{
    CCPM_CHECK_RETURN(act_id, CCPM_EINVAL);

    for (uint16_t i = 0; i < n_act; i++)
    {
        for (uint16_t j = i + 1; j < n_act; j++)
        {
            CCPM_CHECK_RETURN(act_id[i] != act_id[j], CCPM_EINVAL);
        }
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_check_links(uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk)
{
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);

    for (uint16_t i = 0; i < n_lnk; i++)
    {
        for (uint16_t j = i + 1; j < n_lnk; j++)
        {
            CCPM_CHECK_RETURN((lnk_src[i] != lnk_src[j]) || (lnk_dst[i] != lnk_dst[j]), CCPM_EINVAL);
        }
    }
    return CCPM_OK;
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
    CCPM_CHECK_RETURN(item,  0);
    CCPM_CHECK_RETURN(stack, 0);

    void * _data = malloc(sz);
    CCPM_CHECK_RETURN(_data, 0);

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
    if (CCPM_UNLIKELY(!var))                                                                 \
    {                                                                                        \
        CCPM_LOG_PRINTF("Not enough memory at %s, %d", __FILE__, l);                         \
        _ccpm_mem_free(&mem_stack);                                                          \
        return CCPM_ENOMEM;                                                                  \
    }                                                                                        \
    (void)mem_stack

#define CCPM_MEM_ALLOC(type, var, n) _CCPM_MEM_ALLOC(type, var, n, __LINE__)
#define CCPM_MEM_FREE_ALL() _ccpm_mem_free(&mem_stack)

/*===========================================================================*/
static inline bool _ccpm_lookup_act_pos(uint16_t * link, uint16_t * act_id, uint16_t n_act)
{
    for (uint16_t i = 0; i < n_act; i++)
    {
        if (0 == *link - act_id[i])
        {
            *link = i;
            return true;
        }
    }
    return false;
}

ccpmResultEn ccpm_links_prepare(uint16_t * act_id, uint16_t n_act, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk)
{
    CCPM_CHECK_RETURN(act_id,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Translate work indexes to work array positions...\n");
    for (uint16_t i = 0; i < n_lnk; i++)
    {
        CCPM_LOG_PRINTF("L[%d]=(%d,%d)->", i, lnk_src[i], lnk_dst[i]);

        bool found_src = _ccpm_lookup_act_pos(lnk_src + i, act_id, n_act);
        bool found_dst = _ccpm_lookup_act_pos(lnk_dst + i, act_id, n_act);

        CCPM_LOG_PRINTF("[%d,%d]=(%d,%d)\n", lnk_src[i], lnk_dst[i], act_id[lnk_src[i]], act_id[lnk_dst[i]]);
        CCPM_CHECK_RETURN(found_src && found_dst, CCPM_EINVAL);
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_populate_dep_info(uint16_t * act_id,  uint16_t * dep, \
                                    uint16_t * n_dep,   bool     * dep_map, uint16_t n_act, \
                                    uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk)
{
    uint16_t i;
    uint16_t j;

    CCPM_CHECK_RETURN(act_id,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(dep,     CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_dep,   CCPM_EINVAL);
    CCPM_CHECK_RETURN(dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);

    for (i = 0; i < n_act; i++)
    {
        n_dep[i] = 0;
        for (j = 0; j < n_act; j++)
        {
            dep_map[n_act * i + j] = false;
        }
    }

    CCPM_LOG_PRINTF("Populate dependencies data...\n");
    for (uint16_t l = 0; l < n_lnk; l++)
    {
        i = lnk_src[l];
        j = lnk_dst[l];

        if ((i >= n_act) || (j >= n_act))
        {
            return CCPM_EINVAL;
        }

        /*Populate dependency maps*/
        dep_map[n_act * j + i] = true;

        /*Append dependencies*/
        dep[(n_act - 1) * j + n_dep[j]++] = i;

        CCPM_LOG_PRINTF("link[%d] = [%d, %d]\n", l, act_id[i], act_id[j]);
    }
    return CCPM_OK;
}

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

ccpmResultEn ccpm_sort(uint16_t * tmp, uint16_t * key, uint16_t * val, uint16_t n)
{
    CCPM_CHECK_RETURN(tmp, CCPM_EINVAL);
    CCPM_CHECK_RETURN(key, CCPM_EINVAL);
    CCPM_CHECK_RETURN(val, CCPM_EINVAL);

    if (!n)
    {
        return CCPM_OK;
    }
    uint16_t * ms = _merge_sort(tmp, key, val, n);
    if (ms != key)
    {
        for (uint16_t i = 0; i < n; i++)
        {
            key[i] = tmp[i];
        }
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_build_dep(uint16_t    n_act, int16_t     map_len,
                            uint16_t *    tmp, uint16_t *   act_id, uint16_t * act_pos,
                            uint16_t * full_n, uint16_t * full_dep, bool     * full_map,
                            uint16_t *  opt_n, uint16_t *  opt_dep, bool     * opt_map)
{
    uint16_t i;
    uint16_t j;
    uint16_t k;
    uint16_t l;
    uint16_t m;
    uint16_t p;
    uint16_t q;

    CCPM_CHECK_RETURN(act_id,   CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_pos,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(tmp,      CCPM_EINVAL);
    CCPM_CHECK_RETURN(opt_n,    CCPM_EINVAL);
    CCPM_CHECK_RETURN(opt_map,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(opt_dep,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_n,   CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep, CCPM_EINVAL);

    if (full_n != opt_n)
    {
        /*The information is in full_n, full_dep, full_map*/
        for (i = 0; i < n_act; i++)
        {
            for (j = 0; j < map_len; j++)
            {
                opt_map[map_len * i + j] = full_map[map_len * i + j];
            }
        }
    }
    else
    {
        /*In place preprocessing*/
        opt_dep = full_dep;
        opt_map = full_map;
    }

    CCPM_LOG_PRINTF("Building full dependency arrays and maps\n");
    for (i = 0; i < n_act; i++)
    {
        for (j = 0; j < full_n[i]; j++)
        {
            k = full_dep[(n_act - 1) * i + j];
            for (l = 0; l < full_n[k]; l++)
            {
                m = full_dep[(n_act - 1) * k + l];
                if (!full_map[map_len * i + m])
                {
                    /*Add a dependency to full map*/
                    full_map[map_len * i + m] = true;

                    /*Loop detection must be here for segfault protection*/
                    CCPM_CHECK_RETURN(i != m, CCPM_ELOOP);

                    /*Add a dependency to optimized map*/
                    opt_map [map_len * i + k] = true;

                    /*Append a dependency*/
                    full_dep[(n_act - 1) * i + full_n[i]++] = m;
                }
            }
        }
    }

    CCPM_LOG_PRINTF("Sorting works\n");
    /*
    This sorting gives as some cool features:
    1. In place preprocessing of dependencies by ccpm_build_dep.
    2. No need to look back: all current work dependencies after
       ccpm_build_dep call are behind its position.
    3. No need to process added dummies in Postovalova algorithm:
       a) all dummies added before work been processed are behind it!
       b) all dummies added before current dummy are behind it!
    */
    for (i = 0; i < n_act; i++)
    {
        act_pos[i] = i;
    }
    CCPM_CHECK_RETURN(CCPM_OK == ccpm_sort(tmp, act_pos, full_n, n_act), CCPM_EUNK);

    CCPM_LOG_PRINTF("Removing redundant dependencies\n");
    /*
    We are going backward here so if dependency processing is done inplace
    then previous works have full_dep untouched for any processed work.
    */
    for (p = n_act; p > 0; p--)
    {
        i = act_pos[p - 1];
        q = full_n[i];
        for (l = 0; l < q; l++)
        {
            j = full_dep[(n_act - 1) * i + l];
            for (m = 0; m < q; m++)
            {
                k = full_dep[(n_act - 1) * i + m];
                if (k == j)
                {
                    continue;
                }
                if (full_map[map_len * k + j])
                {
                    opt_map[map_len * i + j] = false;
                }
            }
        }
    }

    CCPM_LOG_PRINTF("Full dependency arrays:\n");
    for (p = 0; p < n_act; p++)
    {
        i = act_pos[p];
        k = full_n[i];
        CCPM_LOG_PRINTF("%5d: n=%d dep=[", act_id[i], k);
        for (j = 0; j < k; j++)
        {
            CCPM_LOG_PRINTF("%5d", act_id[full_dep[(n_act - 1) * i + j]]);
        }
        CCPM_LOG_PRINTF(" ]\n");

        /*Populate optimized dependency arrays*/
        opt_n[i] = 0;
        for (j = 0; j < map_len; j++)
        {
            if (opt_map[map_len * i + j])
            {
                /*Append a dependency*/
                opt_dep[(n_act - 1) * i + opt_n[i]++] = j;
            }
        }
    }

    return CCPM_OK;
}

/*===========================================================================*/
static int _qs_comp (const uint16_t *i, const uint16_t *j)
{
    return *i - *j;
}

/*===========================================================================*/

static inline ccpmResultEn _ccpm_restore_links(uint16_t * act_id,  uint16_t * dep, \
                                               uint16_t * n_dep,   uint16_t   n_act, \
                                               uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t * n_lnk)
{
    uint16_t i;
    uint16_t j;
    uint16_t _n_lnk;

    CCPM_CHECK_RETURN(act_id,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(dep,     CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_dep,   CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_lnk,   CCPM_EINVAL);

    CCPM_LOG_PRINTF("Restoring optimized links:\n");
    _n_lnk = 0;
    for (i = 0; i < n_act; i++)
    {
        for (j = 0; j < n_dep[i]; j++)
        {
            lnk_dst[_n_lnk] = act_id[i];
            lnk_src[_n_lnk] = act_id[dep[(n_act - 1) * i + j]];

            CCPM_LOG_PRINTF("link[%d] = [%d, %d]\n", _n_lnk, lnk_src[_n_lnk], lnk_dst[_n_lnk]);
            _n_lnk++;
        }
    }
    *n_lnk = _n_lnk;
    return CCPM_OK;
}
/*===========================================================================*/
#define _CCPM_CHECK_GOTO_END(cond, err, file, func, line)                    \
do {                                                                       \
    if (CCPM_UNLIKELY(!(cond)))                                            \
    {                                                                      \
        CCPM_ERR("CCPM:The expression (%s) is false in \n function: %s",   \
                 #cond, func);                                             \
        CCPM_ERR("\n file: %s\n line: %d\n will return: %s\n",             \
                 file, line, #err);                                        \
        goto end;                                                          \
    }                                                                      \
} while (0)

#define CCPM_CHECK_GOTO_END(cond, err) _CCPM_CHECK_GOTO_END(cond, err, __FILE__, __func__, __LINE__)

/*===========================================================================*/
#define _CCPM_TRY_RETURN(exp, file, func, line)                               \
do {                                                                          \
    ret = (exp);                                                              \
    if (CCPM_UNLIKELY(CCPM_OK != ret))                                        \
    {                                                                         \
        CCPM_ERR("CCPM:The expression (%s) gave an error in \n function: %s", \
                 #exp, func);                                                 \
        CCPM_ERR("\n file: %s\n line: %d\n will return: %d\n",                \
                 file, line, ret);                                            \
        return ret;                                                           \
    }                                                                         \
} while (0)

#define CCPM_TRY_RETURN(exp) _CCPM_TRY_RETURN(exp, __FILE__, __func__, __LINE__)

/*===========================================================================*/
#define _CCPM_TRY_GOTO_END(exp, file, func, line)                             \
do {                                                                          \
    ret = (exp);                                                              \
    if (CCPM_UNLIKELY(CCPM_OK != ret))                                        \
    {                                                                         \
        CCPM_ERR("CCPM:The expression (%s) gave an error in \n function: %s", \
                 #exp, func);                                                 \
        CCPM_ERR("\n file: %s\n line: %d\n will return: %d\n",                \
                 file, line, ret);                                            \
        goto end;                                                             \
    }                                                                         \
} while (0)

#define CCPM_TRY_GOTO_END(exp) _CCPM_TRY_GOTO_END(exp, __FILE__, __func__, __LINE__)

/*===========================================================================*/
#define _CCPM_SG_LIM 0xffff

ccpmResultEn ccpm_make_aoa(uint16_t * act_id, uint16_t * act_src, uint16_t * act_dst, uint16_t n_act, uint16_t * n_dum, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t * n_lnk)
{
    ccpmResultEn ret = CCPM_OK;
    const uint16_t _n_lnk = *n_lnk;
    uint16_t evt_id = 1;
    uint16_t sg_id = 1;
    uint16_t n_chk_act = 0;
    uint16_t _n_dum = 0;
    uint16_t i;
    uint16_t j;
    uint16_t k;
    uint16_t l;
    uint16_t m;
    uint16_t p;
    uint16_t q;

    CCPM_CHECK_RETURN(act_id,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_act,   CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_lnk,   CCPM_EINVAL);

    CCPM_TRY_RETURN(ccpm_check_act_ids(act_id, n_act));
    CCPM_TRY_RETURN(ccpm_check_links(lnk_src, lnk_dst, _n_lnk));

    CCPM_MEM_INIT();
    CCPM_MEM_ALLOC(uint16_t   ,act_pos      ,n_act              ); /*Works positions in sorted lists*/
    CCPM_MEM_ALLOC(uint16_t   ,act_ndep     ,n_act              ); /*Number of dependencies*/
    CCPM_MEM_ALLOC(uint16_t   ,act_dep      ,n_act * (n_act - 1)); /*Array of dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCPM_MEM_ALLOC(bool       ,act_dep_map  ,n_act * n_act      ); /*Work dependency map*/
    /*---------------------------------------------------------------------------------*/
    CCPM_MEM_ALLOC(uint16_t   ,act_rem_dep  ,n_act              ); /*Number of remaining dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCPM_MEM_ALLOC(bool       ,act_started  ,n_act              ); /*Work started flag*/
    CCPM_MEM_ALLOC(uint16_t   ,act_sg_id    ,n_act              ); /*Work does not have dummy successor*/

    CCPM_MEM_ALLOC(uint16_t   ,chk_act      ,n_act              ); /*Work check list (array)*/
    /*-------------------------------------------------------------------------*/
    CCPM_MEM_ALLOC(uint16_t   ,grp_sz       ,n_act              ); /*Work group sizes*/
    CCPM_MEM_ALLOC(uint16_t   ,grp_data     ,n_act * n_act      ); /*Work groups (first member of the group has dependency list for the group)*/
    /*--------------------------------------------------------------------------------------*/
    CCPM_MEM_ALLOC(uint16_t   ,new_sg_act   ,n_act              ); /*Array of works with no dummies successors*/
    CCPM_MEM_ALLOC(uint16_t   ,dummy_pos    ,_n_lnk             ); /*Dummy work index*/
    CCPM_MEM_ALLOC(uint16_t   ,old_sg_map   ,n_act * 2          ); /*Dummy work map*/
    CCPM_MEM_ALLOC(uint16_t   ,new_sg_map   ,n_act * 2          ); /*Work subgroup map*/

    /*Temporary array for sortings*/
    CCPM_MEM_ALLOC(uint16_t,tmp,((n_act > _n_lnk) ? n_act : _n_lnk));

    CCPM_TRY_GOTO_END(ccpm_links_prepare(act_id, n_act, lnk_src, lnk_dst, _n_lnk));

    CCPM_TRY_GOTO_END(ccpm_populate_dep_info(act_id, act_dep, act_ndep, act_dep_map, n_act, \
                                              lnk_src, lnk_dst, _n_lnk));

    CCPM_TRY_GOTO_END(ccpm_build_dep(n_act, n_act, tmp, act_id, act_pos, \
                                      act_ndep, act_dep, act_dep_map,     \
                                      act_ndep, act_dep, act_dep_map));

    CCPM_LOG_PRINTF("Sorted optimized dependency arrays:\n");
    for (p = 0; p < n_act; p++)
    {
        /*Initiate src and dst*/
        act_src[p] = 0;
        act_dst[p] = 0;

        /*Process dependency data*/
        i = act_pos[p];
        k = act_ndep[i];
        CCPM_LOG_PRINTF("%5d: n=%d dep=[", act_id[i], k);

        qsort(act_dep + (n_act - 1) * i, k, sizeof(uint16_t), (int(*) (const void *, const void *)) _qs_comp);
        for (j = 0; j < k; j++)
        {
            CCPM_LOG_PRINTF("%5d", act_id[act_dep[(n_act - 1) * i + j]]);
        }
        CCPM_LOG_PRINTF(" ]\n");

        /*Initiate other work properties*/
        act_rem_dep[i]  = k;
        act_started[i]  = false;
        act_sg_id[i]    = 0;
    }

    CCPM_LOG_PRINTF("Collect started works...\n");
    for (p = 0; p < n_act; p++)
    {
        i = act_pos[p];

        if (act_rem_dep[i])
        {
            break;
        }

        act_started[i] = true;
        act_src[i] = evt_id;
        chk_act[n_chk_act++] = p; /*Append work to chk_act*/
        CCPM_LOG_PRINTF("%5d",  act_id[i]);
    }
    evt_id++;

    CCPM_LOG_PRINTF("\nProcess started works...\n");
    for (j = 0; j < n_chk_act; j++)
    {
        q = act_pos[chk_act[j]];
        /*Find new started works and their dependencies*/
        uint16_t n_grp = 0;
        for (p = chk_act[j] + 1; p < n_act; p++)
        {
            i = act_pos[p];
            if (act_started[i])
            {
                continue;
            }

            if (act_dep_map[n_act * i + q])
            {
                act_rem_dep[i]--;
            }

            if (0 == act_rem_dep[i])
            {
                /*
                Some work gets started.

                Later we will do <act_started[i] = true;> as a result, this block of code
                will be executed exactly once for each started work.

                So overall time complexity of ccpm_make_aoa is O(n^3).
                */
                /*Check if a work has common dependencies with some group of started works*/
                bool is_in_pred = false;
                for (k = 0; k < n_grp; k++)
                {
                    /*First work in a group is used to get groups dependency list(array)*/
                    uint16_t pred = act_pos[grp_data[n_act * k]];

                    /*Compare dependency lists*/
                    if (act_ndep[pred] != act_ndep[i])
                    {
                        continue;
                    }

                    bool is_equal = true;
                    for (l = 0; l < act_ndep[pred]; l++)
                    {
                        if (act_dep[(n_act - 1) * pred + l] != act_dep[(n_act - 1) * i + l])
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
                    grp_data[n_act * k + grp_sz[k]++] = p;
                }
                else
                {
                    /*Create a new group*/
                    grp_sz[n_grp] = 1;
                    grp_data[n_act * n_grp++] = p;
                }
            }
        }
        CCPM_LOG_PRINTF("%5d: Current work in check list is: %d. Found %d groups of started works...\n", j, act_id[q], n_grp);

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
            bool bump_evt = false;

            for (l = 0; l < n_act * 2; l++)
            {
                old_sg_map[l] = _CCPM_SG_LIM;
                new_sg_map[l] = _CCPM_SG_LIM;
            }

            /*Process groups dependency list(array)*/
            uint16_t pred = act_pos[grp_data[n_act * k]];
            uint16_t ndep = act_ndep[pred];
            uint16_t *dep = act_dep + (n_act - 1) * pred;

            CCPM_LOG_PRINTF("Dependencies:\n");
            for (l = 0; l < ndep; l++)
            {
                i = dep[l];
                CCPM_LOG_PRINTF("%5d: %5d %5d %5d\n", act_id[i], act_src[i], act_dst[i], act_sg_id[i]);
                if (act_dst[i])
                {
                    CCPM_CHECK_GOTO_END(0 != act_sg_id[i], CCPM_EUNK);

                    if (_CCPM_SG_LIM == old_sg_map[act_sg_id[i]])
                    {
                        /*Remind an old subgroup*/
                        old_sg_map[act_sg_id[i]] = n_old_sg;
                        tmp[n_old_sg++] = i; /*Old subgroup works*/
                    }
                    else if (act_dst[i] > act_dst[tmp[old_sg_map[act_sg_id[i]]]])
                    {
                        /*Find a work with bigest dst in an old subgroup*/
                        tmp[old_sg_map[act_sg_id[i]]] = i;
                    }
                }
                else if (_CCPM_SG_LIM == new_sg_map[act_src[i]])
                {
                    /*Create new subgroup*/
                    act_sg_id[i] = sg_id++;
                    new_sg_map[act_src[i]] = i;
                    new_sg_act[n_new_sg++] = i;
                }
                else
                {
                    /*Assign a work to some old subgroup*/
                    act_sg_id[i] = act_sg_id[new_sg_map[act_src[i]]];
                    act_dst[i]   = evt_id;

                    /*Append a dummy work*/
                    lnk_src[_n_dum + n_added_dummys++] = evt_id++;
                }
            }
            CCPM_LOG_PRINTF("\n");

            if (n_old_sg)
            {
                CCPM_LOG_PRINTF("Process old subgroups:\n");
                for (l = 0; l < n_act * 2; l++)
                {
                    old_sg_map[l] = _CCPM_SG_LIM;
                }

                for (l = 0; l < n_old_sg; l++)
                {
                    i = tmp[l];
                    if (_CCPM_SG_LIM == old_sg_map[act_dst[i]])
                    {
                        old_sg_map[act_dst[i]] = 1;
                        lnk_src[_n_dum + n_added_dummys++] = act_dst[i];
                        CCPM_LOG_PRINTF("Added dummy: %d %d\n", _n_dum + n_added_dummys, act_dst[i]);
                    }
                }
            }

            /*Finalize a group list processing*/
            CCPM_LOG_PRINTF("Group works: ");
            for (l = 0; l < grp_sz[k]; l++)
            {
                p = grp_data[n_act * k + l];
                i = act_pos[p];
                act_src[i]     = evt_id;
                act_started[i] = true;
                CCPM_LOG_PRINTF("%5d", act_id[i]);

                /*Add this work to work check list(array)*/
                chk_act[n_chk_act++] = p;
            }
            CCPM_LOG_PRINTF("\n");

            CCPM_LOG_PRINTF("Dummy works:\n");
            for (l = 0; l < n_added_dummys; l++)
            {
                lnk_dst[_n_dum + l] = evt_id;
                bump_evt = true;
                CCPM_LOG_PRINTF("%5d %5d\n", lnk_src[_n_dum + l], lnk_dst[_n_dum + l]);
            }
            _n_dum += n_added_dummys;
            CCPM_LOG_PRINTF("\n");

            CCPM_LOG_PRINTF("No dummy works:");
            for (l = 0; l < n_new_sg; l++)
            {
                i = new_sg_act[l];
                CCPM_LOG_PRINTF("%5d", act_id[i]);
                act_dst[i] = evt_id;
                bump_evt = true;
            }
            CCPM_LOG_PRINTF("\n");

            if(bump_evt)
            {
                evt_id++;
            }
        }
    }

    /*Wee need to add last dummies to unfinished works now if necessary*/
    l = 0;
    for (i = 0; i < n_act; i++)
    {
        if (act_dst[i])
        {
            continue;
        }
        for (j = i + 1; j < n_act; j++)
        {
            if (act_dst[j])
            {
                continue;
            }

            if (act_src[i] != act_src[j])
            {
                continue;
            }
            /*Found one more unfinished work with the current source, must add a dummy*/
            /*Finish the work*/
            act_dst[j] = evt_id;
            /*Add a dummy*/
            lnk_src[_n_dum + l++] = evt_id;
            /*Next event id*/
            evt_id++;
        }
    }
    /*Finish last dummies*/
    for (i = 0; i < l; i++)
    {
        lnk_dst[_n_dum + i] = evt_id;
    }
    _n_dum += l;

    /*Finish last works*/
    for (i = 0; i < n_act; i++)
    {
        if (!act_dst[i])
        {
            act_dst[i] = evt_id;
        }
    }

    if(!_n_dum)
    {
        goto end;
    }

    CCPM_LOG_PRINTF("Removing redundant dummies...\n");
    /*Dummies are sorted by "dst" now, sort dummies by "src"*/
    CCPM_LOG_PRINTF("Unsorted dummies:\n");
    for (l = 0; l < _n_dum; l++)
    {
        dummy_pos[l] = l;
        CCPM_LOG_PRINTF("%5d: %5d %5d\n", l, lnk_src[l], lnk_dst[l]);
    }
    CCPM_TRY_GOTO_END(ccpm_sort(tmp, dummy_pos, lnk_src, _n_dum));

    /*Dummies are sorted by "src" and "dst" now as merge sort is stable*/

    /*Mark redundant dummies*/
    CCPM_LOG_PRINTF("Sorted dummies:\n");
    for (l = 0; l < _n_dum; l++)
    {
        CCPM_LOG_PRINTF("%5d: %5d %5d\n", l, lnk_src[dummy_pos[l]], lnk_dst[dummy_pos[l]]);
        tmp[l] = 1; /*Will use tmp for marks*/
    }

    for (k = 0; k < _n_dum; k++)
    {
        i = dummy_pos[k];
        uint16_t src = lnk_src[i];
        uint16_t pvt = lnk_dst[i];
        for (l = k + 1; l < _n_dum; l++)
        {
            j = dummy_pos[l];
            if (lnk_src[j] != src)
            {
                break;
            }

            uint16_t cur = lnk_dst[j];
            for (m = l + 1; m < _n_dum; m++)
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

    /*Append dummies without redundant ones*/
    CCPM_LOG_PRINTF("Dummy works:\n");
    act_src += n_act;
    act_dst += n_act;
    q = 0;
    for (l = 0; l < _n_dum; l++)
    {
        j = dummy_pos[l];
        if (tmp[j])
        {
            CCPM_LOG_PRINTF("%5d: %5d %5d\n", j, lnk_src[j], lnk_dst[j]);
            act_src[q]   = lnk_src[j];
            act_dst[q++] = lnk_dst[j];
        }
    }
    *n_dum = q;

    /*Restore optimized links*/
    CCPM_TRY_GOTO_END(_ccpm_restore_links(act_id, act_dep, act_ndep, n_act, lnk_src, lnk_dst, n_lnk));
end:
    CCPM_MEM_FREE_ALL();
    return ret;
}


ccpmResultEn ccpm_make_full_map(uint16_t * act_id, uint16_t n_act, \
                                uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk,
                                bool * full_dep_map)
{
    ccpmResultEn ret = CCPM_OK;

    CCPM_CHECK_RETURN(act_id,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_act,   CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_lnk,   CCPM_EINVAL);

    CCPM_TRY_RETURN(ccpm_check_act_ids(act_id, n_act));
    CCPM_TRY_RETURN(ccpm_check_links(lnk_src, lnk_dst, n_lnk));

    CCPM_MEM_INIT();
    CCPM_MEM_ALLOC(uint16_t   ,act_pos      ,n_act              ); /*Works positions in sorted lists*/
    /*---------------------------------------------------------------------------------*/
    CCPM_MEM_ALLOC(uint16_t   ,full_ndep     ,n_act              ); /*Number of dependencies*/
    CCPM_MEM_ALLOC(uint16_t   ,full_dep      ,n_act * (n_act - 1)); /*Array of dependencies*/
    /*---------------------------------------------------------------------------------*/
    CCPM_MEM_ALLOC(uint16_t   ,opt_ndep     ,n_act              ); /*Number of dependencies*/
    CCPM_MEM_ALLOC(uint16_t   ,opt_dep      ,n_act * (n_act - 1)); /*Array of dependencies*/
    CCPM_MEM_ALLOC(bool       ,opt_dep_map  ,n_act * n_act      ); /*Work dependency map*/

    /*Temporary array for sortings*/
    CCPM_MEM_ALLOC(uint16_t,tmp,((n_act > n_lnk) ? n_act : n_lnk));

    CCPM_TRY_GOTO_END(ccpm_links_prepare(act_id, n_act, lnk_src, lnk_dst, n_lnk));

    CCPM_TRY_GOTO_END(ccpm_populate_dep_info(act_id, full_dep, full_ndep, full_dep_map, n_act, \
                                             lnk_src, lnk_dst, n_lnk));

    CCPM_TRY_GOTO_END(ccpm_build_dep(n_act, n_act, tmp, act_id, act_pos, \
                                     full_ndep, full_dep, full_dep_map,   \
                                     opt_ndep, opt_dep, opt_dep_map));
#ifdef CCPM_CFG_PRINTF
    CCPM_LOG_PRINTF("Full dependency map:\n");
    for (uint16_t i = 0; i < n_act; i++)
    {
        CCPM_LOG_PRINTF("[");
        for (uint16_t j = 0; j < n_act; j++)
        {
            CCPM_LOG_PRINTF("%1d ", (int)full_dep_map[n_act * i + j]);
        }
        CCPM_LOG_PRINTF("]\n");
    }
#endif/*CCPM_CFG_PRINTF*/
end:
    CCPM_MEM_FREE_ALL();
    return ret;
}
