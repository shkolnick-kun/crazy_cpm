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
        'include_dirs': [np.get_include(), '../src']
        }
    )

import ccpmpy
from ccpmpy import ccpm_compute_aoa

if __name__ == '__main__':
    # wrk = pd.DataFrame(data = {'dep':[
    #     [5,19],
    #     [1,4,16,17,10,12,14],
    #     [2,18,6,7,20],
    #     [5,19],
    #     [],
    #     [1,4,16,17,10,12,14],
    #     [4,16,17,10,12,14],
    #     [6,7,20],
    #     [2,18,6,7,20],
    #     [5,19,11,13],
    #     [],
    #     [5,19,11,13],
    #     [],
    #     [5,19,11,13,15],
    #     [],
    #     [5,19],
    #     [5,19],
    #     [1,4,16,17,10,12,14],
    #     [],
    #     [5,19,11,13,15],
    #     [3,8,9],
    #     ]}, index = list(range(1,22)))

    # wrk_id = np.array(wrk.index)

    # #Make links dataframe
    # src = []
    # dst = []
    # for d in wrk.index:
    #     wrk.dep.at[d].sort()
    #     for s in wrk.dep[d]:
    #         src.append(s)
    #         dst.append(d)
    # src = np.array(src)
    # dst = np.array(dst)

    # status, wrk_src, wrk_dst, lnk_src, lnk_dst = ccpm_compute_aoa(wrk_id, src, dst)

    # print(status)
    # wrk["src"] = wrk_src
    # wrk["dst"] = wrk_dst

    # wrk.sort_values(by='src', inplace=True)
    # print(wrk)

    # dummys = pd.DataFrame(data={'src': list(lnk_src), 'dst':list(lnk_dst)})
    # dummys.sort_values(by='src', inplace=True)
    # print(dummys)

    n_planed = []
    n_effective = []
    ii = []
    tm  = []

    for n in range(1000, 11000, 1000):

        for i in range(10):

            print(i)

            g=nx.gnp_random_graph(n, 0.001, directed=True)
            dag = nx.DiGraph([(u,v) for (u,v) in g.edges() if u<v])
            print(nx.is_directed_acyclic_graph(dag))

            #print(dag.edges())

            wrk_id = np.array(dag.nodes())
            src, dst = tuple(zip(*dag.edges()))
            src = np.array(src)
            dst = np.array(dst)
            print(len(wrk_id))
            print(len(src))

            start = time.time()
            status, wrk_src, wrk_dst, lnk_src, lnk_dst = ccpm_compute_aoa(wrk_id, src, dst)
            end = time.time()

            print(status)
            print(len(wrk_src))
            print(len(lnk_src))
            print(end-start)
            #wrk = pd.DataFrame(data={'src': list(wrk_src), 'dst':list(wrk_dst)}, index=wrk_id)

            #wrk.sort_values(by='src', inplace=True)
            #print(wrk)

            #dummys = pd.DataFrame(data={'src': list(lnk_src), 'dst':list(lnk_dst)})
            #dummys.sort_values(by='src', inplace=True)
            #print(dummys)

            n_planed.append(n)
            n_effective.append(len(wrk_id))
            tm.append(end-start)
            ii.append(i)

    bm = pd.DataFrame({'n_planed': n_planed,
                       'n_effective':n_effective,
                       'i': ii,
                       'time':tm})
    bm.to_csv('../doc/benchmark3.csv')

    plt.scatter(n_effective, tm, s=0.1)
    plt.show()


