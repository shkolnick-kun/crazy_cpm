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
#ifndef CCMP_H
#define CCMP_H

/*===========================================================================*/
#include <stdint.h>
#include <stdbool.h>

/*===========================================================================*/
#define CCPM_ARRAY_SZ(a) (sizeof(a) / sizeof(*a))

/*===========================================================================*/
typedef enum {
    CCMP_OK = 0,
    CCMP_EINVAL,
    CCMP_EAPPEND
}ccpmResultEn;

/*===========================================================================*/
typedef struct {
    uint64_t start;
    uint64_t finish;
} ccmpDatesSt;

/*===========================================================================*/
typedef struct {
    uint16_t src;
    uint16_t dst;
    ccmpDatesSt     early;
    ccmpDatesSt     late;
    uint64_t        duration;
    uint64_t        reserve;
} ccpmWorkSt;

#define CCPM_DEP_BUF(id, ...) \
static const uint16_t _ccpm_dep_buf##id[] = {__VA_ARGS__};

/*===========================================================================*/
typedef struct {
    uint16_t *id;
    uint16_t n;
    uint16_t nmax;
} ccpmIdxSt;

static inline ccpmResultEn ccpm_idx_init(ccpmIdxSt * self, uint16_t * mem, uint16_t lim)
{
    if (!self)
    {
        return CCMP_EINVAL;
    }

    if (!mem)
    {
        return CCMP_EINVAL;
    }
    self->id   = mem;
    self->n    = 0;
    self->nmax = lim;
}

static inline ccpmResultEn ccpm_idx_append(ccpmIdxSt * self, uint16_t i)
{
    if (self->n >= self->nmax)
    {
        return CCMP_EAPPEND;
    }
    self->id[self->n++] = i;
    return CCMP_OK;
}

/*===========================================================================*/
void ccpm_sort(uint16_t * tmp, uint16_t * key, uint16_t * val, uint16_t n);

#endif // CCMP_H
