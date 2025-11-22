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
            CCPM_LOG_PRINTF("%5d ", (int)CCPM_LITEM(dep + n_max * i, j)); \
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
#define  CCPM_LLEN(x)      (*(x))
/*x[i]*/
#define CCPM_LITEM(x, i)   (*((x) + 1 + i))
/*x = []*/
#define  CCPM_LCLR(x)      do {*(x) = 0;} while(0)
/*x.append(val)*/
#define  CCPM_LAPP(x, val) do {(x)[1 + (x)[0]++] = val;} while(0)


/*===========================================================================*/
#define CCPM_FAKE (0xffff)

/*===========================================================================*/
ccpmResultEn ccpm_check_act_idss(uint16_t * act_ids)
{
    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);

    uint16_t n_act = CCPM_LLEN(act_ids);

    for (uint16_t i = 0; i < n_act; i++)
    {
        for (uint16_t j = i + 1; j < n_act; j++)
        {
            CCPM_CHECK_RETURN(CCPM_LITEM(act_ids, i) != CCPM_LITEM(act_ids, j), CCPM_EINVAL);
        }
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_check_links(uint16_t * lnk_src, uint16_t * lnk_dst, size_t n_lnk)
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
static inline bool _ccpm_lookup_act_pos(uint16_t * link, uint16_t * act_ids)
{
    uint16_t n_act = CCPM_LLEN(act_ids);
    for (uint16_t i = 0; i < n_act; i++)
    {
        if (0 == *link - CCPM_LITEM(act_ids, i))
        {
            *link = i;
            return true;
        }
    }
    return false;
}

ccpmResultEn ccpm_links_prepare(uint16_t * act_ids, uint16_t * lnk_src, uint16_t * lnk_dst, size_t n_lnk)
{
    CCPM_CHECK_RETURN(act_ids,  CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Translate work indexes to work array positions...\n");
    for (uint16_t i = 0; i < n_lnk; i++)
    {
        CCPM_LOG_PRINTF("%d: (%d,%d)->", (int)i, (int)lnk_src[i], (int)lnk_dst[i]);

        bool found_src = _ccpm_lookup_act_pos(lnk_src + i, act_ids);
        bool found_dst = _ccpm_lookup_act_pos(lnk_dst + i, act_ids);

        CCPM_LOG_PRINTF("[%d,%d]\n", (int)lnk_src[i], (int)lnk_dst[i]);
        CCPM_CHECK_RETURN(found_src && found_dst, CCPM_EINVAL);
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_populate_dep_info(size_t   n_max,
                                    size_t   n_lnk, uint16_t * lnk_src, uint16_t * lnk_dst,
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
ccpmResultEn ccpm_build_full_deps(size_t     n_act,    size_t  n_max,
                                  uint16_t * full_act_dep, bool *  full_map)
{
    uint16_t i;
    uint16_t j;
    uint16_t k;
    uint16_t l;
    uint16_t m;
    uint16_t p;
    uint16_t q;

    CCPM_CHECK_RETURN(full_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_act_dep, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Building full dependency arrays and maps:\n");
    for (i = 0; i < n_act; i++)
    {
        CCPM_LOG_PRINTF("%5d:[", (int)i);
        /*len(full_act_dep[i])*/
        p = CCPM_LLEN(full_act_dep + n_max * i);
        for (j = 0; j < p; j++)
        {
            /*k = full_act_dep[i][j]*/
            k = CCPM_LITEM(full_act_dep + n_max * i, j);
            /*len(full_act_dep[k])*/
            q = CCPM_LLEN(full_act_dep + n_max * k);
            for (l = 0; l < q; l++)
            {
                /*m = full_act_dep[k][l]*/
                m = CCPM_LITEM(full_act_dep + n_max * k, l);
                if (!full_map[n_max * i + m])
                {
                    /*Add a dependency to full map*/
                    /*full_map[i,m] = True*/
                    full_map[n_max * i + m] = true;

                    /*Loop detection must be here for segfault protection*/
                    CCPM_CHECK_RETURN(i != m, CCPM_ELOOP);

                    /*Append a dependency*/
                    /*full_act_dep[i].append(m)*/
                    CCPM_LAPP(full_act_dep + n_max * i, m);
                    CCPM_LOG_PRINTF("%5d ", (int)m);
                }
            }
        }
        CCPM_LOG_PRINTF("]\n");
    }
    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_optimize_deps(size_t     n_act,    size_t     n_max,
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
    CCPM_LCLR(act_pos);
    for (i = 1; i < n_act; i++)
    {
        CCPM_LAPP(act_pos, i);
    }
    CCPM_CHECK_RETURN(CCPM_OK == ccpm_sort(tmp, act_pos + 1, full_n, n_act), CCPM_EUNK);

    CCPM_LOG_PRINTF("Removing redundant dependencies\n");
    /*
    We are going backward here so if dependency processing is done inplace
    then previous works have full dependencies untouched for any processed work.
    */
    for (p = n_act; p > 0; p--)
    {
        i = CCPM_LITEM(act_pos, p - 1);
        /*len(full_act_dep[i])*/
        q = CCPM_LLEN(dep + n_max * i);
        for (l = 0; l < q; l++)
        {
            /*j = dep[i][l]*/
            j = CCPM_LITEM(dep + n_max * i, l);
            for (m = 0; m < q; m++)
            {
                /*k = dep[i][m]*/
                k = CCPM_LITEM(dep + n_max * i, m);
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
ccpmResultEn ccpm_handle_deps(uint16_t * min_deps, uint16_t target,
                              size_t n_cur, size_t n_max,
                              uint16_t *  min_act_dep, bool *  min_dep_map,
                              uint16_t * full_act_dep, bool * full_dep_map)
{
    uint16_t i;
    uint16_t d;

    CCPM_CHECK_RETURN(min_deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(target < n_max, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_cur < n_max, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Handling dependencies for target %d with dummy %d\n",
                   (int)target, (int)n_cur);

    /* Append to target predeceptors in full dependencies */
    full_dep_map[n_max * target + n_cur] = true;
    CCPM_LAPP(full_act_dep + n_max * target, n_cur);

    /* Replace target min dependencies with dummy action */
    for (i = 0; i < CCPM_LLEN(min_deps); i++)
    {
        d = CCPM_LITEM(min_deps, i);

        /* Remove direct dependency on d */
        min_dep_map[n_max * target + d] = false;

        /* Add dependency on dummy */
        min_dep_map[n_max * target + n_cur] = true;
    }

    /* Recompute target min dependencies */
    CCPM_LCLR(min_act_dep + n_max * target);
    for (i = 0; i < n_max; i++)
    {
        if (min_dep_map[n_max * target + i])
        {
            CCPM_LAPP(min_act_dep + n_max * target, i);
        }
    }

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_add_a_dummy(uint16_t * min_deps, uint16_t * deps, bool * dep_map,
                              size_t n_cur, size_t n_max,
                              uint16_t * act_ids, uint16_t * act_pos,
                              uint16_t *  min_act_dep, bool *  min_dep_map,
                              uint16_t * full_act_dep, bool * full_dep_map)
{
    uint16_t i;
    uint16_t d;

    CCPM_CHECK_RETURN(min_deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_pos, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_cur < n_max, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Adding dummy activity at position %d\n", (int)n_cur);

    /* Add fake activity ID to act_ids list */
    CCPM_LAPP(act_ids, CCPM_FAKE);

    /* Add current position to act_pos list */
    CCPM_LAPP(act_pos, n_cur);

    /* Set dummy minimal dependencies */
    CCPM_LCLR(min_act_dep + n_max * n_cur);
    for (i = 0; i < CCPM_LLEN(min_deps); i++)
    {
        d = CCPM_LITEM(min_deps, i);
        CCPM_LAPP(min_act_dep + n_max * n_cur, d); /*Copy deps to new dummy*/
        min_dep_map[n_max * n_cur + d] = true;     /*Set dep map*/
    }

    /* Set dummy full dependencies */
    CCPM_LCLR(full_act_dep + n_max * n_cur);
    for (i = 0; i < CCPM_LLEN(deps); i++)
    {
        d = CCPM_LITEM(deps, i);
        CCPM_LAPP(full_act_dep + n_max * n_cur, d);
    }

    /* Also copy the dependency map for the dummy */
    for (i = 0; i < n_max; i++)
    {
        full_dep_map[n_max * n_cur + i] = dep_map[i];
    }

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_full_act_deps(uint16_t * min_deps,
                                uint16_t * deps, bool * dep_map,
                                size_t n_max,
                                uint16_t * full_act_dep, bool * full_dep_map)
{
    uint16_t i, j, k;

    CCPM_CHECK_RETURN(min_deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep_map, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Building full dependencies for given minimal dependencies\n");

    /* Copy minimal dependencies to deps list */
    CCPM_LCLR(deps);
    for (i = 0; i < CCPM_LLEN(min_deps); i++)
    {
        j = CCPM_LITEM(min_deps, i);
        CCPM_LAPP(deps, j);
    }

    /* Initialize dependency map */
    for (i = 0; i < n_max; i++)
    {
        dep_map[i] = false;
    }

    /* Build transitive closure of dependencies */
    for (i = 0; CCPM_LLEN(deps); i++)
    {
        j = CCPM_LITEM(deps, i);

        /* Process all dependencies of j */
        for (k = 0; k < CCPM_LLEN(full_act_dep + n_max * j); k++)
        {
            uint16_t d = CCPM_LITEM(full_act_dep + n_max * j, k);

            if (dep_map[d])
            {
                continue;
            }

            dep_map[d] = true;

            /* Check for loop */
            CCPM_CHECK_RETURN(j != d, CCPM_ELOOP);

            CCPM_LAPP(deps, d);
        }
    }

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_process_nested_deps(size_t n_act, size_t n_max,
                                      uint16_t * act_pos,
                                      uint16_t * min_act_dep, bool * min_dep_map,
                                      uint16_t * full_act_dep, bool * full_dep_map,
                                      uint16_t * act_ids,
                                      size_t * n_cur,
                                      uint16_t * min_com_deps,
                                      uint16_t * tmp_deps, bool * tmp_dep_map)
{
    uint16_t p, q, i, j, k;
    uint16_t lcd, lmcd;
    ccpmResultEn ret = CCPM_OK;

    CCPM_CHECK_RETURN(act_pos, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_cur, CCPM_EINVAL);
    CCPM_CHECK_RETURN(tmp_deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(tmp_dep_map, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Processing nested dependencies\n");

    for (p = 0; p < n_act; p++)
    {
        i = CCPM_LITEM(act_pos, p);

        /* Skip activities without dependencies */
        if (0 == CCPM_LLEN(min_act_dep + n_max * i))
        {
            continue;
        }

        /* Search for nested list */
        CCPM_LCLR(min_com_deps);

        for (q = p + 1; q < n_act; q++)
        {
            j = CCPM_LITEM(act_pos, q);

            /* Skip activities without dependencies */
            if (0 == CCPM_LLEN(min_act_dep + n_max * j))
            {
                continue;
            }

            /* Find common dependencies between i and j */
            CCPM_LCLR(min_com_deps);
            for (k = 0; k < CCPM_LLEN(min_act_dep + n_max * i); k++)
            {
                uint16_t d = CCPM_LITEM(min_act_dep + n_max * i, k);
                if (min_dep_map[n_max * j + d])
                {
                    CCPM_LAPP(min_com_deps, d);
                }
            }

            lcd = CCPM_LLEN(min_com_deps);
            uint16_t len_i = CCPM_LLEN(min_act_dep + n_max * i);
            uint16_t len_j = CCPM_LLEN(min_act_dep + n_max * j);

            if ((len_i == lcd || len_j == lcd) && (len_i != len_j))
            {
                /* Nested lists found, will reduce nested lists */
                break;
            }
        }

        if (0 == CCPM_LLEN(min_com_deps))
        {
            /* No nested lists found, continue */
            continue;
        }

        /* Found nested lists, reduce them */
        lmcd = CCPM_LLEN(min_com_deps);

        /* Build full dependencies for common deps */
        CCPM_TRY_RETURN(ccpm_full_act_deps(min_com_deps, tmp_deps, tmp_dep_map,
                                          n_max, full_act_dep, full_dep_map));

        /* Process all activities that have these common dependencies */
        for (q = p + 1; q < n_act; q++)
        {
            j = CCPM_LITEM(act_pos, q);

            uint16_t len_j = CCPM_LLEN(min_act_dep + n_max * j);
            if (0 == len_j || len_j == lmcd)
            {
                /* Skip empty, equal, or non-nested lists */
                continue;
            }

            /* Check if j has all common dependencies */
            uint16_t com_count = 0;
            for (k = 0; k < lmcd; k++)
            {
                uint16_t d = CCPM_LITEM(min_com_deps, k);
                if (min_dep_map[n_max * j + d])
                {
                    com_count++;
                }
            }

            if (com_count != lmcd)
            {
                /* Skip non-nested lists */
                continue;
            }

            /* Reduce nested lists */
            CCPM_TRY_RETURN(ccpm_handle_deps(min_com_deps, j, *n_cur, n_max,
                                            min_act_dep, min_dep_map,
                                            full_act_dep, full_dep_map));

            CCPM_TRY_RETURN(ccpm_add_a_dummy(min_com_deps, tmp_deps, tmp_dep_map,
                                            *n_cur, n_max,
                                            act_ids, act_pos,
                                            min_act_dep, min_dep_map,
                                            full_act_dep, full_dep_map));

            (*n_cur)++;
        }
    }

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_process_overlapping_deps(size_t n_max,
                                          uint16_t * act_pos,
                                          uint16_t * min_act_dep, bool * min_dep_map,
                                          uint16_t * full_act_dep, bool * full_dep_map,
                                          uint16_t * act_ids,
                                          size_t * n_cur,
                                          uint16_t * min_com_deps,
                                          uint16_t * tmp_deps, bool * tmp_dep_map)
{
    uint16_t p, q, i, j, k;
    ccpmResultEn ret = CCPM_OK;

    CCPM_CHECK_RETURN(act_pos, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_cur, CCPM_EINVAL);
    CCPM_CHECK_RETURN(tmp_deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(tmp_dep_map, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Processing overlapping dependencies\n");

    size_t n_last = *n_cur;

    for (p = 0; p < n_last; p++)
    {
        i = CCPM_LITEM(act_pos, p);

        /* Skip activities without dependencies */
        if (0 == CCPM_LLEN(min_act_dep + n_max * i))
        {
            continue;
        }

        /* Search for overlapping lists */
        CCPM_LCLR(min_com_deps);

        for (q = 0; q < n_last; q++)
        {
            j = CCPM_LITEM(act_pos, q);

            /* Skip activities without dependencies */
            if (0 == CCPM_LLEN(min_act_dep + n_max * j))
            {
                continue;
            }

            /* Find common dependencies between i and j */
            CCPM_LCLR(min_com_deps);
            for (k = 0; k < CCPM_LLEN(min_act_dep + n_max * i); k++)
            {
                uint16_t d = CCPM_LITEM(min_act_dep + n_max * i, k);
                if (min_dep_map[n_max * j + d])
                {
                    CCPM_LAPP(min_com_deps, d);
                }
            }

            uint16_t lmcd = CCPM_LLEN(min_com_deps);
            uint16_t len_i = CCPM_LLEN(min_act_dep + n_max * i);
            uint16_t len_j = CCPM_LLEN(min_act_dep + n_max * j);

            if (lmcd > 0 && len_i != lmcd && len_j != lmcd)
            {
                /* Found overlapping lists */
                break;
            }
        }

        if (0 == CCPM_LLEN(min_com_deps))
        {
            n_last = *n_cur;
            continue;
        }

        uint16_t lmcd = CCPM_LLEN(min_com_deps);

        /* Build full dependencies for common deps */
        CCPM_TRY_RETURN(ccpm_full_act_deps(min_com_deps, tmp_deps, tmp_dep_map,
                                          n_max, full_act_dep, full_dep_map));

        /* Process all activities that have these common dependencies */
        for (q = 0; q < n_last; q++)
        {
            j = CCPM_LITEM(act_pos, q);

            /* Skip activities without dependencies */
            if (0 == CCPM_LLEN(min_act_dep + n_max * j))
            {
                continue;
            }

            /* Check if j has all common dependencies */
            uint16_t com_count = 0;
            for (k = 0; k < lmcd; k++)
            {
                uint16_t d = CCPM_LITEM(min_com_deps, k);
                if (min_dep_map[n_max * j + d])
                {
                    com_count++;
                }
            }

            if (com_count == lmcd && CCPM_LLEN(min_act_dep + n_max * j) != lmcd)
            {
                /* Reduce overlapping dependencies */
                CCPM_TRY_RETURN(ccpm_handle_deps(min_com_deps, j, *n_cur, n_max,
                                                min_act_dep, min_dep_map,
                                                full_act_dep, full_dep_map));

                CCPM_TRY_RETURN(ccpm_add_a_dummy(min_com_deps, tmp_deps, tmp_dep_map,
                                                *n_cur, n_max,
                                                act_ids, act_pos,
                                                min_act_dep, min_dep_map,
                                                full_act_dep, full_dep_map));

                (*n_cur)++;
            }
        }

        n_last = *n_cur;
    }

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_build_network(size_t n_max, size_t * n_cur,
                               uint16_t * act_ids, uint16_t * act_pos,
                               uint16_t * min_act_dep, bool * min_dep_map,
                               uint16_t * full_act_dep, bool * full_dep_map,
                               uint16_t * act_src, uint16_t * act_dst,
                               uint16_t * started, uint16_t * num_dep,
                               uint16_t * events, uint16_t * chk,
                               uint16_t * start)
{
    uint16_t i, j, k;
    uint16_t evt = 1;
    uint16_t dum = *n_cur;

    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_pos, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(min_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_act_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(full_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(started, CCPM_EINVAL);
    CCPM_CHECK_RETURN(num_dep, CCPM_EINVAL);
    CCPM_CHECK_RETURN(events, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Building network with %d activities\n", (int)dum);

    /* Initialize arrays as lists */
    CCPM_LCLR(started);
    CCPM_LCLR(num_dep);
    CCPM_LCLR(act_src);
    CCPM_LCLR(act_dst);
    CCPM_LCLR(events);
    CCPM_LCLR(chk);

    /* Initialize started and num_dep lists */
    for (i = 0; i < dum; i++)
    {
        CCPM_LAPP(started, false);
        CCPM_LAPP(num_dep, CCPM_LLEN(min_act_dep + n_max * i));
        CCPM_LAPP(act_src, 0);
        CCPM_LAPP(act_dst, 0);
    }

    /* Find initial activities without dependencies */
    CCPM_LCLR(chk);

    for (i = 0; i < dum; i++)
    {
        if ((0 == CCPM_LITEM(num_dep, i)) && (!CCPM_LITEM(started, i)))
        {
            /* started[i] = true */
            CCPM_LITEM(started, i) = true;
            /* act_src[i] = evt */
            CCPM_LITEM(act_src, i) = evt;
            CCPM_LAPP(chk, i);
        }
    }

    /* Add initial event */
    CCPM_LAPP(events, evt);
    evt++;

    /* Process activities in order */
    for (i = 0; i < CCPM_LLEN(chk); i++)
    {
        uint16_t current_act = CCPM_LITEM(chk, i);

        /* Decrement dependency counters for activities that depend on current activity */
        for (j = 0; j < dum; j++)
        {
            if (min_dep_map[n_max * j + current_act])
            {
                /* num_dep[j]-- */
                CCPM_LITEM(num_dep, j) = CCPM_LITEM(num_dep, j) - 1;
            }
        }

        /* Find newly started activities */
        CCPM_LCLR(start);

        for (j = 0; j < dum; j++)
        {
            if ((0 == CCPM_LITEM(num_dep, j)) && (!CCPM_LITEM(started, j)))
            {
                CCPM_LITEM(started, j) = true;
                CCPM_LITEM(act_src, j) = evt;
                CCPM_LAPP(start, j);
            }
        }

        /* Process newly started activities */
        if (CCPM_LLEN(start) > 0)
        {
            /* For the first newly started activity, set destinations for its dependencies */
            uint16_t first_act = CCPM_LITEM(start, 0);
            for (k = 0; k < CCPM_LLEN(min_act_dep + n_max * first_act); k++)
            {
                uint16_t dep_act = CCPM_LITEM(min_act_dep + n_max * first_act, k);

                if (CCPM_LITEM(act_dst, dep_act))
                {
                    /* Need to add a dummy activity */
                    CCPM_LAPP(act_pos, dum);
                    CCPM_LAPP(act_ids, CCPM_FAKE);

                    /* Extend lists for new dummy activity */
                    CCPM_LAPP(started, true);
                    CCPM_LAPP(num_dep, 0);
                    CCPM_LAPP(act_src, CCPM_LITEM(act_dst, dep_act));
                    CCPM_LAPP(act_dst, evt);

                    dum++;
                }
                else
                {
                    CCPM_LITEM(act_dst, dep_act) = evt;
                }
            }

            /* Add new event */
            CCPM_LAPP(events, evt);
            evt++;
        }

        /* Add newly started activities to check list */
        for (j = 0; j < CCPM_LLEN(start); j++)
        {
            CCPM_LAPP(chk, CCPM_LITEM(start, j));
        }
    }

    /* Set destination for last started  activities*/
    for (i = 0; i < dum; i++)
    {
        if (0 == CCPM_LITEM(act_dst, i))
        {
            CCPM_LITEM(act_dst, i) = evt;
        }
    }

    /* Add final event */
    CCPM_LAPP(events, evt);

    *n_cur = dum;

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_do_glue(size_t n_cur,
                          uint16_t * act_src, uint16_t * act_dst,
                          uint16_t * events)
{
    uint16_t k;

    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(events, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Applying event glueing to all activities\n");

    /* Apply event glueing to all activities */
    for (k = 0; k < n_cur; k++)
    {
        /* Skip redundant activities */
        if (CCPM_FAKE == CCPM_LITEM(act_src, k) || CCPM_FAKE == CCPM_LITEM(act_dst, k))
        {
            continue;
        }

        /* Update source and destination events */
        uint16_t src_evt = CCPM_LITEM(act_src, k) - 1;
        uint16_t dst_evt = CCPM_LITEM(act_dst, k) - 1;

        CCPM_LITEM(act_src, k) = CCPM_LITEM(events, src_evt);
        CCPM_LITEM(act_dst, k) = CCPM_LITEM(events, dst_evt);
    }

    return CCPM_OK;
}

/*===========================================================================*/
ccpmResultEn ccpm_optimize_network_stage_1(size_t n_cur, size_t n_max,
                                          uint16_t * act_ids, uint16_t * act_src, uint16_t * act_dst,
                                          uint16_t * events, bool * evt_dep_map,
                                          uint16_t * evt_deps, uint16_t * evt_dins, bool * evt_real)
{
    ccpmResultEn ret = CCPM_OK;

    uint16_t i, j, k;
    uint16_t num_events = CCPM_LLEN(events);

    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(events, CCPM_EINVAL);
    CCPM_CHECK_RETURN(evt_dep_map, CCPM_EINVAL);
    CCPM_CHECK_RETURN(evt_deps, CCPM_EINVAL);
    CCPM_CHECK_RETURN(evt_dins, CCPM_EINVAL);
    CCPM_CHECK_RETURN(evt_real, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Optimizing network stage 1\n");

    /* Initialize event data structures */
    for (i = 0; i < num_events; i++)
    {
        CCPM_LCLR(evt_deps + 2 * n_max * i);
        CCPM_LCLR(evt_dins + 2 * n_max * i);
        evt_real[i] = false;
    }

    /* Clear full_dep_map for reuse */
    for (i = 0; i < 4 * n_max * n_max; i++)
    {
        evt_dep_map[i] = false;
    }

    /* Populate event dependencies and inputs */
    for (k = 0; k < n_cur; k++)
    {
        uint16_t src_evt = CCPM_LITEM(act_src, k) - 1;
        uint16_t dst_evt = CCPM_LITEM(act_dst, k) - 1;

        /*Skip events that were marked as fake*/
        if (CCPM_FAKE != CCPM_LITEM(act_ids, k))
        {
            evt_real[dst_evt] = true;
            continue;
        }

        /* For dummy activities, record dependencies */
        CCPM_LAPP(evt_dins + 2 * n_max * dst_evt, k);
        CCPM_LAPP(evt_deps + 2 * n_max * dst_evt, src_evt);
        evt_dep_map[2 * n_max * dst_evt + src_evt] = true;
    }

    /*
    If some events have only dummy inputs and have equal dependencies
    (earlier events) then we can "glue" them together
    */
    for (i = 0; i < num_events; i++)
    {
        /*Skip events with real inputs*/
        if (evt_real[i])
        {
            continue;
        }

        /*Skip events without dummy inputs*/
        if (0 == CCPM_LLEN(evt_deps + 2 * n_max * i))
        {
            continue;
        }

        for (j = i + 1; j < num_events; j++)
        {
            /*Skip events with real inputs*/
            if (evt_real[j])
            {
                continue;
            }

            /*Skip events without dummy inputs*/
            if (CCPM_LLEN(evt_deps + 2 * n_max * i) < 2)
            {
                continue;
            }

            /*Skip events with unequal dependencies lists*/
            if (CCPM_LLEN(evt_deps + 2 * n_max * i) != CCPM_LLEN(evt_deps + 2 * n_max * j))
            {
                continue;
            }

            /* Check if events have same dependencies */
            uint16_t match_count = 0;
            for (k = 0; k < CCPM_LLEN(evt_deps + 2 * n_max * i); k++)
            {
                uint16_t dep = CCPM_LITEM(evt_deps + 2 * n_max * i, k);
                if (evt_dep_map[2 * n_max * j + dep])
                {
                    match_count++;
                }
            }

            /*
            Will redirect _act_src later
            (events[i] != (i + 1)) is feature of redundant event
            */
            if (match_count == CCPM_LLEN(evt_deps + 2 * n_max * i))
            {
                /* Glue events: redirect j to i */
                CCPM_LITEM(events, j) = CCPM_LITEM(events, i);

                /* Mark dummy activities for removal */
                for (k = 0; k < CCPM_LLEN(evt_dins + 2 * n_max * j); k++)
                {
                    uint16_t dummy_act = CCPM_LITEM(evt_dins + 2 * n_max * j, k);
                    CCPM_LITEM(act_src, dummy_act) = CCPM_FAKE;
                    CCPM_LITEM(act_dst, dummy_act) = CCPM_FAKE;
                }
            }
        }
    }

    /*
    Work with redundant events and dummies (part 2)
    "Glue" events each of which has only one input which is dummy
    to their predeceptors
    */
    for (i = 0; i < num_events; i++)
    {
        /*Skip events with real inputs*/
        if (evt_real[i])
        {
            continue;
        }

        if (1 == CCPM_LLEN(evt_deps + 2 * n_max * i))
        {
            uint16_t dummy_act = CCPM_LITEM(evt_dins + 2 * n_max * i, 0);
            CCPM_LITEM(events, i) = CCPM_LITEM(act_src, dummy_act);

            /* Mark dummy activity for removal */
            CCPM_LITEM(act_src, dummy_act) = CCPM_FAKE;
            CCPM_LITEM(act_dst, dummy_act) = CCPM_FAKE;
        }
    }

    /* Apply event glueing to all activities */
    CCPM_TRY_RETURN(ccpm_do_glue(n_cur, act_src, act_dst, events));

    return ret;
}
/*===========================================================================*/
ccpmResultEn ccpm_optimize_network_stage_2(size_t n_cur, size_t n_max,
                                          uint16_t * act_ids, uint16_t * act_src, uint16_t * act_dst,
                                          uint16_t * events,
                                          uint16_t * evt_douts, uint16_t * evt_nout)
{
    ccpmResultEn ret = CCPM_OK;
    uint16_t i, k;
    uint16_t num_events = CCPM_LLEN(events);

    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(events, CCPM_EINVAL);
    CCPM_CHECK_RETURN(evt_douts, CCPM_EINVAL);
    CCPM_CHECK_RETURN(evt_nout, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Optimizing network stage 2\n");

    /* Initialize event data structures */
    for (i = 0; i < num_events; i++)
    {
        CCPM_LCLR(evt_douts + 2 * n_max * i);
        evt_nout[i] = 0;
    }

    /* Count outputs and collect dummy outputs for each event */
    for (k = 0; k < n_cur; k++)
    {
        /* Skip redundant activities */
        if (CCPM_FAKE == CCPM_LITEM(act_src, k) || CCPM_FAKE == CCPM_LITEM(act_dst, k))
        {
            continue;
        }

        uint16_t src_evt = CCPM_LITEM(act_src, k) - 1;
        evt_nout[src_evt]++;

        /* If this is a dummy activity, add to dummy outputs */
        if (CCPM_FAKE == CCPM_LITEM(act_ids, k))
        {
            CCPM_LAPP(evt_douts + 2 * n_max * src_evt, k);
        }
    }

    /* Optimize: glue events with single dummy output to their successors */
    for (i = 0; i < num_events; i++)
    {
        /* Skip if event has more than one output */
        if (evt_nout[i] > 1)
        {
            continue;
        }

        /* Skip if event has no dummy outputs */
        if (0 == CCPM_LLEN(evt_douts + 2 * n_max * i))
        {
            continue;
        }

        /* Get the dummy output activity */
        uint16_t dummy_act = CCPM_LITEM(evt_douts + 2 * n_max * i, 0);

        /* Glue event to its successor */
        CCPM_LITEM(events, i) = CCPM_LITEM(act_dst, dummy_act);

        /* Mark dummy activity for removal */
        CCPM_LITEM(act_src, dummy_act) = CCPM_FAKE;
        CCPM_LITEM(act_dst, dummy_act) = CCPM_FAKE;
    }

    /* Apply event glueing to all activities */
    CCPM_TRY_RETURN(ccpm_do_glue(n_cur, act_src, act_dst, events));

    return ret;
}

/*===========================================================================*/
ccpmResultEn ccpm_add_needed_dummies(size_t * n_cur,
                                    uint16_t * act_ids, uint16_t * act_pos,
                                    uint16_t * act_src, uint16_t * act_dst,
                                    uint16_t * to_do, uint16_t * events,
                                    size_t * n_events,
                                    uint16_t * sort_values, uint16_t * tmp)
{
    ccpmResultEn ret = CCPM_OK;
    uint16_t i, j;
    uint16_t d = *n_cur;
    uint16_t evt = CCPM_LITEM(events, CCPM_LLEN(events) - 1); // Last event

    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_pos, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(to_do, CCPM_EINVAL);
    CCPM_CHECK_RETURN(events, CCPM_EINVAL);
    CCPM_CHECK_RETURN(n_events, CCPM_EINVAL);
    CCPM_CHECK_RETURN(sort_values, CCPM_EINVAL);
    CCPM_CHECK_RETURN(tmp, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Adding needed dummies\n");

    /* Sort act_pos by act_dst and then by act_src */
    /* First, sort by act_dst */
    for (i = 0; i < CCPM_LLEN(act_pos); i++)
    {
        uint16_t idx = CCPM_LITEM(act_pos, i);
        sort_values[i] = CCPM_LITEM(act_dst, idx);
    }

    CCPM_TRY_RETURN(ccpm_sort(tmp, act_pos + 1, sort_values, CCPM_LLEN(act_pos)));

    /* Then, sort by act_src */
    for (i = 0; i < CCPM_LLEN(act_pos); i++)
    {
        uint16_t idx = CCPM_LITEM(act_pos, i);
        sort_values[i] = CCPM_LITEM(act_src, idx);
    }

    CCPM_TRY_RETURN(ccpm_sort(tmp, act_pos + 1, sort_values, CCPM_LLEN(act_pos)));

    /* Process activities to add needed dummies */
    for (i = 0; i < d; i++)
    {
        uint16_t act_i = CCPM_LITEM(act_pos, i);

        /* Skip redundant activities */
        if (CCPM_FAKE == CCPM_LITEM(act_src, act_i) || CCPM_FAKE == CCPM_LITEM(act_dst, act_i))
        {
            continue;
        }

        /* Skip done activities */
        if (!CCPM_LITEM(to_do, act_i))
        {
            continue;
        }

        for (j = i + 1; j < d; j++)
        {
            uint16_t act_j = CCPM_LITEM(act_pos, j);

            /* Skip redundant activities */
            if (CCPM_FAKE == CCPM_LITEM(act_src, act_j) || CCPM_FAKE == CCPM_LITEM(act_dst, act_j))
            {
                continue;
            }

            /* Check if activities have same source and destination */
            if ((CCPM_LITEM(act_dst, act_i) == CCPM_LITEM(act_dst, act_j)) &&
                (CCPM_LITEM(act_src, act_i) == CCPM_LITEM(act_src, act_j)))
            {
                /* Mark second activity as not to_do */
                CCPM_LITEM(to_do, act_j) = false;

                /* Create new event */
                evt++;
                CCPM_LITEM(act_dst, act_j) = evt;

                /* Add dummy activity */
                CCPM_LAPP(act_pos, d);
                CCPM_LAPP(act_ids, CCPM_FAKE);
                CCPM_LAPP(act_src, evt);
                CCPM_LAPP(act_dst, CCPM_LITEM(act_dst, act_i));
                CCPM_LAPP(to_do, false);

                /* Add new event to events list */
                CCPM_LAPP(events, evt);
                (*n_events)++;

                (*n_cur)++;
            }
        }
    }
    return ret;
}

/*===========================================================================*/
ccpmResultEn ccpm_finalize_network(size_t n_cur,
                                  uint16_t * act_ids, uint16_t * act_pos,
                                  uint16_t * act_src, uint16_t * act_dst,
                                  uint16_t * events,
                                  uint16_t * final_act_ids, uint16_t * final_act_src, uint16_t * final_act_dst,
                                  uint16_t * tmp)
{
    ccpmResultEn ret = CCPM_OK;

    uint16_t i;
    uint16_t num_events = CCPM_LLEN(events);
    uint16_t evt = 1;

    CCPM_CHECK_RETURN(act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_pos, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(events, CCPM_EINVAL);
    CCPM_CHECK_RETURN(final_act_ids, CCPM_EINVAL);
    CCPM_CHECK_RETURN(final_act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(final_act_dst, CCPM_EINVAL);
    CCPM_CHECK_RETURN(tmp, CCPM_EINVAL);

    CCPM_LOG_PRINTF("Finalizing network\n");

    /* Initialize final output lists */
    CCPM_LCLR(final_act_ids);
    CCPM_LCLR(final_act_src);
    CCPM_LCLR(final_act_dst);

    /*Renumerate events*/
    for (i = 0; i < num_events; i++)
    {
        if (CCPM_LITEM(events, i) != (i + 1))
        {
            CCPM_LITEM(events, i) = CCPM_FAKE;
        }
        else
        {
            CCPM_LITEM(events, i) = evt;
            evt++;
        }
    }

    /* Update activity source and destination events */
    for (i = 0; i < n_cur; i++)
    {
        /* Skip redundant activities */
        if (CCPM_FAKE == CCPM_LITEM(act_src, i) || CCPM_FAKE == CCPM_LITEM(act_dst, i))
        {
            continue;
        }

        uint16_t src_evt = CCPM_LITEM(act_src, i) - 1;
        uint16_t dst_evt = CCPM_LITEM(act_dst, i) - 1;

        /* Skip if events are invalid */
        if (CCPM_FAKE == CCPM_LITEM(events, src_evt) || CCPM_FAKE == CCPM_LITEM(events, dst_evt))
        {
            continue;
        }

        /* Update source and destination */
        CCPM_LITEM(act_src, i) = CCPM_LITEM(events, src_evt);
        CCPM_LITEM(act_dst, i) = CCPM_LITEM(events, dst_evt);
    }

    /* Sort act_pos by act_ids for final output */
    CCPM_TRY_RETURN(ccpm_sort(tmp, act_pos + 1, act_ids, CCPM_LLEN(act_pos)));

    /* Build final output lists */
    for (i = 0; i < CCPM_LLEN(act_pos); i++)
    {
        uint16_t idx = CCPM_LITEM(act_pos, i);

        /* Skip if activity is fake or has invalid events */
        if (CCPM_FAKE == CCPM_LITEM(act_ids, idx) ||
            CCPM_FAKE == CCPM_LITEM(act_src, idx) ||
            CCPM_FAKE == CCPM_LITEM(act_dst, idx))
        {
            continue;
        }

        /* Add to final output */
        CCPM_LAPP(final_act_ids, CCPM_LITEM(act_ids, idx));
        CCPM_LAPP(final_act_src, CCPM_LITEM(act_src, idx));
        CCPM_LAPP(final_act_dst, CCPM_LITEM(act_dst, idx));
    }

    return ret;
}

/*===========================================================================*/
/*to DeepSeek: All allocation must in the function below*/
ccpmResultEn ccpm_make_aoa(uint16_t * act_ids, uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk, uint16_t * act_src, uint16_t * act_dst)
{
    ccpmResultEn ret = CCPM_OK;

    uint16_t i;

    const size_t _n_lnk = n_lnk;

    CCPM_CHECK_RETURN(act_ids,  CCPM_EINVAL);

    size_t n_act = CCPM_LLEN(act_ids);
    CCPM_CHECK_RETURN(n_act,   CCPM_EINVAL);

    CCPM_CHECK_RETURN(act_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(act_dst, CCPM_EINVAL);

    CCPM_CHECK_RETURN(lnk_src, CCPM_EINVAL);
    CCPM_CHECK_RETURN(lnk_dst, CCPM_EINVAL);

    CCPM_TRY_RETURN(ccpm_check_act_idss(act_ids));
    CCPM_TRY_RETURN(ccpm_check_links(lnk_src, lnk_dst, _n_lnk));

    size_t n_max = n_act + ((_n_lnk > n_act) ? _n_lnk : n_act);

    CCPM_LOG_PRINTF("n_act: %5d\nn_max: %5d\n",  (int)n_act, (int)n_max);

    CCPM_MEM_INIT();

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,_act_ids      ,n_max + 1   );
    CCPM_MEM_ALLOC(uint16_t   ,_act_pos      ,n_max + 1   ); /*Works positions in sorted lists*/
    CCPM_MEM_ALLOC(uint16_t   ,_act_src      ,n_max + 1   );
    CCPM_MEM_ALLOC(uint16_t   ,_act_dst      ,n_max + 1   );

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,_full_act_ndep ,n_max        ); /*Number of dependencies*/
    CCPM_MEM_ALLOC(uint16_t   ,_full_act_dep  ,n_max * n_max); /*Array of dependencies*/
    CCPM_MEM_ALLOC(bool       ,_full_dep_map  ,n_max * n_max); /*Work dependency map*/

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,_min_act_ndep ,n_max        ); /*Number of dependencies*/
    CCPM_MEM_ALLOC(uint16_t   ,_min_act_dep  ,n_max * n_max); /*Array of dependencies*/
    CCPM_MEM_ALLOC(bool       ,_min_dep_map  ,n_max * n_max); /*Work dependency map*/

    /*=======================================================================*/
    CCPM_MEM_ALLOC(uint16_t   ,_tmp          ,2 * n_max + 1);

    /*=======================================================================*/
    /* Additional allocations for new functions */
    CCPM_MEM_ALLOC(uint16_t   ,_started     ,n_max + 1   );
    CCPM_MEM_ALLOC(uint16_t   ,_num_dep     ,n_max + 1   );
    CCPM_MEM_ALLOC(uint16_t   ,_events      ,n_max + 1   );
    CCPM_MEM_ALLOC(uint16_t   ,_chk         ,n_max + 1   );

    /* For nested and overlapping dependencies */
    CCPM_MEM_ALLOC(uint16_t   ,_min_com_deps ,n_max       );
    CCPM_MEM_ALLOC(uint16_t   ,_tmp_deps     ,n_max + 1   );
    CCPM_MEM_ALLOC(bool       ,_tmp_dep_map  ,n_max       );

    /* For optimize_network_stage_1 */
    CCPM_MEM_ALLOC(bool       ,_evt_dep_map ,4 * n_max * n_max);
    CCPM_MEM_ALLOC(uint16_t   ,_evt_deps    ,4 * n_max * n_max);
    CCPM_MEM_ALLOC(uint16_t   ,_evt_dins    ,4 * n_max * n_max);
    CCPM_MEM_ALLOC(bool       ,_evt_real    ,2 * n_max        );

    /* For optimize_network_stage_2 */
    CCPM_MEM_ALLOC(uint16_t   ,_evt_douts   ,4 * n_max * n_max);
    CCPM_MEM_ALLOC(uint16_t   ,_evt_nout    ,2 * n_max        );

    /* For add_needed_dummies */
    CCPM_MEM_ALLOC(uint16_t   ,_sorted_activities, 2 * n_max + 1);
    CCPM_MEM_ALLOC(uint16_t   ,_sort_values      , 2 * n_max + 1);

    /*=======================================================================*/
    size_t n_cur = n_act;
    size_t n_events = 0;

    for (i = 0; i <= CCPM_LLEN(act_ids); i++)
    {
        _act_ids[i] = act_ids[i];
    }

    /*Prepare links for computing dependency info*/
    CCPM_TRY_GOTO_END(ccpm_links_prepare(_act_ids, lnk_src, lnk_dst, _n_lnk));

    /*Compute dependency info as is*/
    CCPM_TRY_GOTO_END(ccpm_populate_dep_info(n_max, _n_lnk, lnk_src, lnk_dst, _full_act_dep, _full_dep_map));
    CCPM_PRINT_DEPS(n_act, n_max, _full_act_dep, _full_dep_map);

    /*Compute full dependency info*/
    CCPM_TRY_GOTO_END(ccpm_build_full_deps(n_act, n_max, _full_act_dep, _full_dep_map));
    CCPM_PRINT_DEPS(n_act, n_max, _full_act_dep, _full_dep_map);

    for (i = 0; i < n_max; i++)
    {
        _full_act_ndep[i] = CCPM_LLEN(_full_act_dep + n_max * i);
    }

    memcpy(_min_act_dep, _full_act_dep, n_max * n_max * sizeof(uint16_t));
    memcpy(_min_dep_map, _full_dep_map, n_max * n_max * sizeof(bool)    );

    CCPM_TRY_GOTO_END(ccpm_optimize_deps(n_act, n_max, _act_pos, _full_act_ndep, _min_act_dep, _min_dep_map, _tmp));
    CCPM_PRINT_DEPS(n_act, n_max, _min_act_dep, _min_dep_map);

    /* Process nested dependencies */
    CCPM_TRY_GOTO_END(ccpm_process_nested_deps(n_act, n_max, _act_pos,
                                              _min_act_dep, _min_dep_map,
                                              _full_act_dep, _full_dep_map,
                                              _act_ids, &n_cur,
                                              _min_com_deps,
                                              _tmp_deps, _tmp_dep_map));

    CCPM_PRINT_DEPS(n_cur, n_max, _min_act_dep, _min_dep_map);

    /* Process overlapping dependencies */
    CCPM_TRY_GOTO_END(ccpm_process_overlapping_deps(n_max, _act_pos,
                                                   _min_act_dep, _min_dep_map,
                                                   _full_act_dep, _full_dep_map,
                                                   _act_ids, &n_cur,
                                                   _min_com_deps,
                                                   _tmp_deps, _tmp_dep_map));
    CCPM_PRINT_DEPS(n_cur, n_max, _min_act_dep, _min_dep_map);

    /* Build network */
    CCPM_TRY_GOTO_END(ccpm_build_network(n_max, &n_cur, _act_ids, _act_pos,
                                        _min_act_dep, _min_dep_map,
                                        _full_act_dep, _full_dep_map,
                                        _act_src, _act_dst,
                                        _started, _num_dep,
                                        _events, _chk, _tmp_deps));

    /* Optimize network stage 1 */
    CCPM_TRY_GOTO_END(ccpm_optimize_network_stage_1(n_cur, n_max, _act_ids, _act_src, _act_dst,
                                                   _events, _evt_dep_map,
                                                   _evt_deps, _evt_dins, _evt_real));

    /* Optimize network stage 2 */
    CCPM_TRY_GOTO_END(ccpm_optimize_network_stage_2(n_cur, n_max, _act_ids, _act_src, _act_dst,
                                                   _events, _evt_douts, _evt_nout));

    /* Add needed dummies */
    n_events = CCPM_LLEN(_events);
    CCPM_TRY_GOTO_END(ccpm_add_needed_dummies(&n_cur, _act_ids, _act_pos,
                                             _act_src, _act_dst, _started, _events, &n_events,
                                             _sort_values, _tmp));

    /* Finalize network */
    CCPM_TRY_GOTO_END(ccpm_finalize_network(n_cur, _act_ids, _act_pos,
                                           _act_src, _act_dst, _events,
                                           act_ids, act_src, act_dst,
                                           _tmp));

end:
    CCPM_MEM_FREE_ALL();
    return ret;
}

//ccpmResultEn ccpm_make_full_map(uint16_t * act_ids, uint16_t n_act, \
//                                uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t n_lnk,
//                                bool * full_dep_map)
//{
//    ccpmResultEn ret = CCPM_OK;
//    CCPM_MEM_INIT();
//end:
//    CCPM_MEM_FREE_ALL();
//    return ret;
//}
