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
#include <string.h>

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
#ifdef CCPM_CFG_PRINTF
#define CCPM_PRINT_DEPS(n_act, n_max, dep, dep_map)                      \
do {                                                                      \
    CCPM_LOG_PRINTF("Dependencies:\n");                                   \
    for (uint16_t i = 0; i < n_act; i++)                                  \
    {                                                                     \
        CCPM_LOG_PRINTF("%5d: [", (int)i);                                \
        for (uint16_t j = 0; j < CCPM_LLEN(dep + n_max * i); j++)         \
        {                                                                 \
            CCPM_LOG_PRINTF("%5d ", (int)CCPM_LGET(dep + n_max * i, j));  \
        }                                                                 \
        CCPM_LOG_PRINTF("]\n");                                           \
    }                                                                     \
    CCPM_LOG_PRINTF("Dependcy map:\n");                                   \
    for (uint16_t i = 0; i < n_act; i++)                                  \
    {                                                                     \
        CCPM_LOG_PRINTF("%5d: [", (int)i);                                \
        for (uint16_t j = 0; j < n_max; j++)                              \
        {                                                                 \
            CCPM_LOG_PRINTF("%d ", (int)dep_map[n_max * i + j]);          \
        }                                                                 \
        CCPM_LOG_PRINTF("]\n");                                           \
    }                                                                     \
} while (0)
#else
#define CCPM_PRINT_DEPS(n_act, n_max, dep, dep_map) do {} while (0)
#endif


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
/*"List" manipulation macros */
/*len(x)*/
#define CCPM_LLEN(x)      (*(x))
/*x[i]*/
#define CCPM_LGET(x, i)   (*((x) + 1 + i))
/*x = []*/
#define CCPM_LCLR(x)      do {*(x) = 0;} while(0)
/*x.append(val)*/
#define CCPM_LAPP(x, val) do {(x)[1 + (x)[0]++] = val;} while(0)


/*===========================================================================*/
#define CCPM_FAKE (0xffff)

/*===========================================================================*/
ccpmResultEn ccpm_check_act_ids(uint16_t * act_id)
{
    CCPM_CHECK_RETURN(act_id, CCPM_EINVAL);

    uint16_t n_act = CCPM_LLEN(act_id);

    for (uint16_t i = 0; i < n_act; i++)
    {
        for (uint16_t j = i + 1; j < n_act; j++)
        {
            CCPM_CHECK_RETURN(CCPM_LGET(act_id, i) != CCPM_LGET(act_id, j), CCPM_EINVAL);
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
static inline bool _ccpm_lookup_act_pos(uint16_t * link, uint16_t * act_id)
{
    uint16_t n_act = CCPM_LLEN(act_id);
    for (uint16_t i = 0; i < n_act; i++)
    {
        if (0 == *link - CCPM_LGET(act_id, i))
        {
            *link = i;
            return true;
        }
    }
    return false;
}

ccpmResultEn ccpm_links_prepare(uint16_t * act_id, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk)
{
    CCPM_CHECK_RETURN(act_id,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Translate work indexes to work array positions...\n");
    for (uint16_t i = 0; i < n_lnk; i++)
    {
        CCPM_LOG_PRINTF("%d: (%d,%d)->", (int)i, (int)lnk_src[i], (int)lnk_dst[i]);

        bool found_src = _ccpm_lookup_act_pos(lnk_src + i, act_id);
        bool found_dst = _ccpm_lookup_act_pos(lnk_dst + i, act_id);

        CCPM_LOG_PRINTF("[%d,%d]\n", (int)lnk_src[i], (int)lnk_dst[i]);
        CCPM_CHECK_RETURN(found_src && found_dst, CCPM_EINVAL);
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_populate_dep_info(uint16_t n_max,
                                    uint16_t n_lnk, uint16_t * lnk_src, uint16_t * lnk_dst,
                                    uint16_t * dep, bool * dep_map)
{
    uint16_t i;
    uint16_t j;

    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(dep,     CCPM_EINVAL);
    CCPM_CHECK_RETURN(dep_map, CCPM_EINVAL);

    for (i = 0; i < n_max; i++)
    {
        /*dep[i] = []*/
        CCPM_LCLR(dep + n_max * i);
        for (j = 0; j < n_max; j++)
        {
            dep_map[n_max * i + j] = false;
        }
    }

    CCPM_LOG_PRINTF("Populate dependencies data...\n");
    for (uint16_t l = 0; l < n_lnk; l++)
    {
        i = lnk_src[l];
        j = lnk_dst[l];

        CCPM_CHECK_RETURN((i < n_max) && (j < n_max), CCPM_EINVAL);

        /*Populate dependency maps*/
        /*dep_map[j, i] = True*/
        dep_map[n_max * j + i] = true;

        /*Append dependencies*/
        /*dep[j].append(i)*/
        CCPM_LAPP(dep + n_max * j, i);
    }

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_build_full_deps(uint16_t   n_act,    int16_t n_max,
                                  uint16_t * full_dep, bool *  full_map)
{
    uint16_t i;
    uint16_t j;
    uint16_t k;
    uint16_t l;
    uint16_t m;
    uint16_t p;
    uint16_t q;

    CCPM_CHECK_RETURN(full_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Building full dependency arrays and maps:\n");
    for (i = 0; i < n_act; i++)
    {
        CCPM_LOG_PRINTF("%5d:[", (int)i);
        /*len(full_dep[i])*/
        p = CCPM_LLEN(full_dep + n_max * i);
        for (j = 0; j < p; j++)
        {
            /*k = full_dep[i][j]*/
            k = CCPM_LGET(full_dep + n_max * i, j);
            /*len(full_dep[k])*/
            q = CCPM_LLEN(full_dep + n_max * k);
            for (l = 0; l < q; l++)
            {
                /*m = full_dep[k][l]*/
                m = CCPM_LGET(full_dep + n_max * k, l);
                if (!full_map[n_max * i + m])
                {
                    /*Add a dependency to full map*/
                    /*full_map[i,m] = True*/
                    full_map[n_max * i + m] = true;

                    /*Loop detection must be here for segfault protection*/
                    CCPM_CHECK_RETURN(i != m, CCPM_ELOOP);

                    /*Append a dependency*/
                    /*full_dep[i].append(m)*/
                    CCPM_LAPP(full_dep + n_max * i, m);
                    CCPM_LOG_PRINTF("%5d ", (int)m);
                }
            }
        }
        CCPM_LOG_PRINTF("]\n");
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_optimize_deps(uint16_t   n_act,    int16_t    n_max,
                                uint16_t * act_pos,  uint16_t * full_n,
                                uint16_t * dep,      bool     * map,
                                uint16_t * tmp)
{
    uint16_t i;
    uint16_t j;
    uint16_t k;
    uint16_t l;
    uint16_t m;
    uint16_t p;
    uint16_t q;

    /*
    This sorting gives as some cool features:
    1. In place preprocessing of dependencies by ccpm_optimize_deps.
    2. No need to look back: all current work dependencies after
       ccpm_optimize_deps call are behind its position.
    */
    for (i = 0; i < n_max; i++)
    {
        act_pos[i] = i;
    }
    CCPM_CHECK_RETURN(CCPM_OK == ccpm_sort(tmp, act_pos, full_n, n_act), CCPM_EUNK);

    CCPM_LOG_PRINTF("Removing redundant dependencies\n");
    /*
    We are going backward here so if dependency processing is done inplace
    then previous works have full dependencies untouched for any processed work.
    */
    for (p = n_act; p > 0; p--)
    {
        i = act_pos[p - 1];
        /*len(full_dep[i])*/
        q = CCPM_LLEN(dep + n_max * i);
        for (l = 0; l < q; l++)
        {
            /*j = dep[i][l]*/
            j = CCPM_LGET(dep + n_max * i, l);
            for (m = 0; m < q; m++)
            {
                /*k = dep[i][m]*/
                k = CCPM_LGET(dep + n_max * i, m);
                if (k == j)
                {
                    continue;
                }
                if (map[n_max * k + j])
                {
                    map[n_max * i + j] = false;
                }
            }
        }
    }

    /*Populate optimized dependency arrays*/
    for (i = 0; i < n_act; i++)
    {
        /*dep[i] = []*/
        CCPM_LCLR(dep + n_max * i);
        for (j = 0; j < n_max; j++)
        {
            if (map[n_max * i + j])
            {
                /*Append a dependency*/
                /*dep[i].append(j)*/
                CCPM_LAPP(dep + n_max * i, j);
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
#define _CCPM_SG_LIM 0xffff

ccpmResultEn ccpm_make_aoa(uint16_t * act_id, uint16_t * act_src, uint16_t * act_dst, uint16_t * n_dum, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t * n_lnk)
{
    uint16_t i;
    ccpmResultEn ret = CCPM_OK;
    const uint16_t _n_lnk = *n_lnk;

    CCPM_CHECK_RETURN(act_id,  CCPM_EINVAL);

    uint16_t n_act = CCPM_LLEN(act_id);
    CCPM_CHECK_RETURN(n_act,   CCPM_EINVAL);

    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);

    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_lnk,   CCPM_EINVAL);

    CCPM_TRY_RETURN(ccpm_check_act_ids(act_id));
    CCPM_TRY_RETURN(ccpm_check_links(lnk_src, lnk_dst, _n_lnk));

    uint16_t n_max = n_act + ((_n_lnk > n_act) ? _n_lnk : n_act);

    CCPM_LOG_PRINTF("n_act: %5d\nn_max: %5d\n",  (int)n_act, (int)n_max);

    CCPM_MEM_INIT();

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,act_pos      ,n_max        ); /*Works positions in sorted lists*/

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,full_ndep    ,n_max        ); /*Number of dependencies*/
    CCPM_MEM_ALLOC(uint16_t   ,full_dep     ,n_max * n_max); /*Array of dependencies*/
    CCPM_MEM_ALLOC(bool       ,full_dep_map ,n_max * n_max); /*Work dependency map*/

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,min_ndep     ,n_max        ); /*Number of dependencies*/
    CCPM_MEM_ALLOC(uint16_t   ,min_dep      ,n_max * n_max); /*Array of dependencies*/
    CCPM_MEM_ALLOC(bool       ,min_dep_map  ,n_max * n_max); /*Work dependency map*/

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,tmp          ,n_max        );

    /*Prepare links for computing dependency info*/
    CCPM_TRY_GOTO_END(ccpm_links_prepare(act_id, lnk_src, lnk_dst, _n_lnk));

    /*Compute dependency info as is*/
    CCPM_TRY_GOTO_END(ccpm_populate_dep_info(n_max, _n_lnk, lnk_src, lnk_dst, full_dep, full_dep_map));
    CCPM_PRINT_DEPS(n_act, n_max, full_dep, full_dep_map);

    /*Compute full dependency info*/
    CCPM_TRY_GOTO_END(ccpm_build_full_deps(n_act, n_max, full_dep, full_dep_map));
    CCPM_PRINT_DEPS(n_act, n_max, full_dep, full_dep_map);

    for (i = 0; i < n_max; i++)
    {
        full_ndep[i] = CCPM_LLEN(full_dep + n_max * i);
    }

    memcpy(min_dep,         full_dep, n_max * n_max * sizeof(uint16_t));
    memcpy(min_dep_map, full_dep_map, n_max * n_max * sizeof(bool)    );

    CCPM_TRY_GOTO_END(ccpm_optimize_deps(n_act, n_max, act_pos, full_ndep, min_dep, min_dep_map, tmp));
    CCPM_PRINT_DEPS(n_act, n_max, min_dep, min_dep_map);
end:
    CCPM_MEM_FREE_ALL();
    return ret;
}


ccpmResultEn ccpm_make_full_map(uint16_t * act_id, uint16_t n_act, \
                                uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk,
                                bool * full_dep_map)
{
    ccpmResultEn ret = CCPM_OK;
    CCPM_MEM_INIT();
end:
    CCPM_MEM_FREE_ALL();
    return ret;
}
