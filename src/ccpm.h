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
    CCPM_ELIM,
    CCPM_EUNK
}ccpmResultEn;

#define CCPM_DEP_BUF(id, ...) \
static const uint16_t _ccpm_dep_buf##id[] = {__VA_ARGS__};

/*===========================================================================*/
ccpmResultEn ccpm_sort(uint32_t * tmp, uint32_t * key, uint32_t * val, size_t n);

/*===========================================================================*/
ccpmResultEn ccpm_make_aoa(uint16_t * act_ids,
                           uint16_t * lnk_src, uint16_t * lnk_dst, size_t n_lnk,
                           uint16_t * act_src, uint16_t * act_dst);

/*===========================================================================*/
ccpmResultEn ccpm_make_full_map(uint16_t * act_ids,
                                uint16_t * lnk_src, uint16_t * lnk_dst,
                                size_t n_lnk, size_t n_max,
                                uint32_t * full_act_dep, uint8_t * full_dep_map);
#endif // CCMP_H
