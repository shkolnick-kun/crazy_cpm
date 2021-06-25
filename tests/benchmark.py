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
    n_planed = []
    n_effective = []
    n_links = []
    tm = []
    ii = []

    for n in range(1000, 11000, 1000):

        for i in range(2,13):

            print(i)

            g = nx.gnp_random_graph(n, i/(n - 1), directed=True)
            dag = nx.DiGraph([(u,v) for (u,v) in g.edges() if u<v])
            print(nx.is_directed_acyclic_graph(dag))


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
            n_links.append(len(src))
            ii.append(i)
            tm.append(end-start)

    bm = pd.DataFrame({'n_planed': n_planed,
                       'n_effective':n_effective,
                       'n_links': n_links,
                       'ii': ii,
                       'time':tm})
    bm.to_csv('../doc/benchmark3.csv')

    plt.plot(bm.n_effective, bm.time, s=0.1)
    plt.show()


