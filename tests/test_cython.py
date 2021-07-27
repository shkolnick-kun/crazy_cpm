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
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import pyximport
import scipy.stats
import sys
import time

sys.path.insert(0,'../src')

pyximport.install(
    build_dir='ctest/obj',
    pyimport=True,
    reload_support=True,
    language_level=3,
    setup_args={
        'include_dirs': [np.get_include(), '../src'],
        }
    )

import ccpmpy
from ccpmpy import ccpm_compute_aoa

if __name__ == '__main__':
    wrk = pd.DataFrame(data = {
        'dep':[
            [5,19],
            [1,4,16,17,10,12,14],
            [2,18,6,7,20],
            [5,19],
            [],
            [1,4,16,17,10,12,14],
            [4,16,17,10,12,14],
            [6,7,20],
            [2,18,6,7,20],
            [5,19,11,13],
            [],
            [5,19,11,13],
            [],
            [5,19,11,13,15],
            [],
            [5,19],
            [5,19],
            [1,4,16,17,10,12,14],
            [],
            [5,19,11,13,15],
            [3,8,9],
            ],
        'time':[1,2,3,4,14,4,7,6,2,3,11,1,1,1,7,4,2,5,2,9,3]
        }, index = list(range(1,22)))

    wrk_id = np.array(wrk.index)

    #Make links dataframe
    src = []
    dst = []
    for d in wrk.index:
        wrk.dep.at[d].sort()
        for s in wrk.dep[d]:
            src.append(s)
            dst.append(d)
    src = np.array(src)
    dst = np.array(dst)

    #Строим сетевую модель
    status, wrk_src, wrk_dst, lnk_src, lnk_dst = ccpm_compute_aoa(wrk_id, src, dst)

    print(status)
    wrk["src"] = wrk_src
    wrk["dst"] = wrk_dst

    wrk.sort_values(by='src', inplace=True)
    print(wrk)

    nd = len(lnk_src)
    d_start = max(list(wrk.index)) + 1
    print(list(range(d_start, d_start + nd)))
    dummys = pd.DataFrame(data={'dep':[None] * nd, 'time':[0] * nd,
                                'src': list(lnk_src), 'dst':list(lnk_dst)},
                          index=list(range(d_start, d_start + nd)))
    dummys.sort_values(by='src', inplace=True)
    print(dummys)

    #Все работы
    net_wrk = pd.concat([wrk, dummys])
    net_wrk.sort_values(by=['src','dst'], inplace=True)
    nn = len(net_wrk)
    net_wrk['estart']   = [-1]*nn
    net_wrk['lstart']   = [-1]*nn
    net_wrk['efinish']  = [-1]*nn
    net_wrk['lfinish']  = [-1]*nn
    net_wrk['reserve']  = [-1]*nn
    print(net_wrk)

    #Вехи
    ne = np.max(net_wrk.dst.values)
    net_evt = pd.DataFrame(data={'early'  :[-1] * ne,
                                 'late'   :[-1] * ne,
                                 'reserve':[-1] * ne,
                                 'start'  :[[] for i in range(ne)],
                                 'finish' :[[] for i in range(ne)],
                                 'fwd'    :[0] * ne,
                                 'rev'    :[0] * ne,},
                           index=list(range(1, ne + 1)))

    for i in net_wrk.index:
        s = net_wrk.src[i]
        net_evt.start.at[s].append(i)
        net_evt.fwd.at[s] += 1

        d = net_wrk.dst[i]
        net_evt.finish.at[d].append(i)
        net_evt.rev.at[d] += 1
    print(net_evt)

    #Расчет критического пути
    #Прямой ход; Берем работы из start, при обнулении rev пополняем список вех
    #Обратный ход: Берем работы из finish, при обнулении fwd пополняем список вех


