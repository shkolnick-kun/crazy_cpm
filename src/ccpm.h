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
#define CCPM_CAT(a,b) a##b
#define CCPM_CAT2(a,b) CCPM_CAT(a,b)
#define CCPM_ARRAY_SZ(a) (sizeof(a) / sizeof(*a))

/*===========================================================================*/
typedef enum {
    CCPM_OK = 0,
    CCPM_EINVAL,
    CCPM_ENOMEM,
    CCPM_ELOOP,
    CCPM_EUNK
}ccpmResultEn;

#define CCPM_DEP_BUF(id, ...) \
static const uint16_t _ccpm_dep_buf##id[] = {__VA_ARGS__};

/*===========================================================================*/
ccpmResultEn ccpm_sort(uint16_t * tmp, uint16_t * key, uint16_t * val, uint16_t n);

/*===========================================================================*/
ccpmResultEn ccpm_make_aoa(uint16_t * act_id, uint16_t * act_src, uint16_t * act_dst, \
                           uint16_t n_act, uint16_t * n_dum, \
                           uint16_t * lnk_src, uint16_t * lnk_dst, uint16_t * n_lnk);

#endif // CCMP_H
