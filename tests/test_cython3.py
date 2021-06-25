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
    wrk = pd.DataFrame(data = {'dep':[
        [],
        [],
        [],
        [],
        [1,2,3],
        [2,3],
        [3,4],
        [1,6,7],
        [5,6,7],
        [3,6,7],
        [6,8,9],
        [7,8,9,10],
        ]}, index = list(range(1,13)))

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

    status, wrk_src, wrk_dst, lnk_src, lnk_dst = ccpm_compute_aoa(wrk_id, src, dst)

    print(status)
    wrk["src"] = wrk_src
    wrk["dst"] = wrk_dst

    wrk.sort_values(by='src', inplace=True)
    print(wrk)

    dummys = pd.DataFrame(data={'src': list(lnk_src), 'dst':list(lnk_dst)})
    dummys.sort_values(by='src', inplace=True)
    print(dummys)



