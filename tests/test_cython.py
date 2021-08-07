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
import sys
import time
import pandas as pd
import scipy.stats


from ccpm import net as net


if __name__ == '__main__':
    in_wrk = pd.read_excel('data/input/project_template.xlsx', sheet_name='WBS', engine='openpyxl')
    #in_lnk = pd.read_excel('data/input/project_template.xlsx', sheet_name='Links', engine='openpyxl')

    in_lnk = pd.read_excel('data/input/project_template.xlsx', sheet_name='Links', engine='openpyxl').T
    in_lnk.columns = in_lnk.iloc[0]
    in_lnk.drop(index=['Name','N'],inplace=True)

    #[i for i in in_lnk[5] if not np.isnan(i)]
    src = []
    dst = []
    for d in in_lnk.columns:
        for s in in_lnk[d]:
            if not np.isnan(s):
                dst.append(int(d))
                src.append(int(s))



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
        'time':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        }, index = list(range(1,22)))

    #Make links dataframe
    src = []
    dst = []
    for d in wrk.index:
        wrk.dep.at[d].sort()
        for s in wrk.dep[d]:
            src.append(s)
            dst.append(d)


    #Из условной базы данных мы получим индексы работ с длительностями и связи
    act_time = wrk['time'].values
    act_id   = np.array(wrk.index, dtype=np.uint16)
    lnk_src  = np.array(src, dtype=np.uint16)
    lnk_dst  = np.array(dst, dtype=np.uint16)
    lnk_id   = np.array(list(range(len(lnk_src))), dtype=np.uint16)

    #Строим сетевую модель
    prj_lnk, net_evt, net_act = net.create(act_id, act_time, lnk_id, lnk_src, lnk_dst)

    #Считаем критический путь, резервы времени, слои графа сетевой модели
    net.do_cpm(net_evt, net_act)
