#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
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
"""
import numpy as np
import pandas as pd

import _ccpm

def create(act_id, act_time, lnk_id, lnk_src, lnk_dst):
    """
    Построение сетевой модели
    """
    status, net_src, net_dst, lnk_src, lnk_dst = _ccpm.compute_aoa(act_id, lnk_src, lnk_dst)
    if 0 != status:
        raise RuntimeError("Couldn't compute AoA network model!")

    #Оптимизированные связи
    prj_lnk = pd.DataFrame(data={'src': lnk_src, 'dst': lnk_dst},
                           index=lnk_id[:len(lnk_src)].astype(int))

    #Работы
    #В net_src и net_dst у нас сначала идут реальные работы, а потом - фиктивные
    nd = len(net_src) - len(act_id) #Количество фиктивных работ
    d_start = np.max(act_id) + 1

    act = pd.DataFrame(data={'time':act_time}, index=act_id.astype(int))
    dum = pd.DataFrame(data={'time':[0] * nd},
                       index=list(range(d_start, d_start + nd)))

    net_act = pd.concat([act, dum])
    net_act['src'] = net_src
    net_act['dst'] = net_dst
    net_act.sort_values(by=['src','dst'], inplace=True)

    #Вехи
    ne = np.max(net_dst)
    net_evt = pd.DataFrame(data={'early':[0]*ne,
                                 'late':[-1]*ne,
                                 'reserve':[-1]*ne,
                                 'layer':[-1]*ne},
                           index=list(range(1, ne + 1)))
    return prj_lnk, net_evt, net_act

def _cpm_compute(net_evt, net_act, target=None):
    if 'early' == target:
        act_base     = 'estart'
        act_new      = 'efinish'
        act_next     = 'dst'
        fwd          = 'finish'
        rev          = 'start'
        delta        = lambda j : net_act.time.at[j]
        choise       = max
    elif 'layer' == target:
        act_base     = None
        act_new      = None
        act_next     = 'dst'
        fwd          = 'finish'
        rev          = 'start'
        delta        = lambda j : 1
        choise       = max
    elif 'late' == target:
        act_base     = 'lfinish'
        act_new      = 'lstart'
        act_next     = 'src'
        fwd          = 'start'
        rev          = 'finish'
        delta        = lambda j : - net_act.time.at[j]
        choise       = min
    else:
        raise ValueError("Unknown 'target' value!!!")

    net_evt['n_dep'] = net_evt[rev].apply(len)

    if 'layer' != target:
        net_act[act_base] = -1
        net_act[act_new]  = -1

    i = 0
    evt = list(net_evt[net_evt.n_dep == 0].index)
    while True:

        base_val = net_evt[target].at[evt[i]]
        for j in net_evt[fwd].at[evt[i]]:
            if act_base:
                net_act[act_base].at[j] = base_val

            new_val = base_val + delta(j)
            if act_new:
                net_act[act_new].at[j] = new_val

            nxt = net_act[act_next].at[j]
            net_evt[target].at[nxt] = choise(net_evt[target].at[nxt], new_val)
            net_evt['n_dep'].at[nxt] -= 1

            if 0 >= net_evt['n_dep'].at[nxt]:
                evt.append(nxt)

        #Пост условие
        i += 1
        if i >= len(evt):
            break

    net_evt.drop(labels=['n_dep'], axis=1, inplace=True)

def do_cpm(net_evt, net_act):
    net_evt['start' ] = [list(net_act[net_act.dst == i].index) for i in net_evt.index]
    net_evt['finish'] = [list(net_act[net_act.src == i].index) for i in net_evt.index]

    _cpm_compute(net_evt, net_act, 'early')

    net_evt['late'] = np.max(net_evt['early'].values)
    _cpm_compute(net_evt, net_act, 'late')

    net_act['reserve'] = net_act.lstart - net_act.estart
    net_evt['reserve'] = net_evt.late - net_evt.early

    net_evt['layer'] = 0
    _cpm_compute(net_evt, net_act, 'layer')

    net_evt.drop(labels=['start', 'finish'], axis=1, inplace=True)

