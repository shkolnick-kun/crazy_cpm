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

in_wrk = pd.read_excel('data/input/project_template.xlsx', sheet_name='WBS', engine='openpyxl')
in_wrk.dropna(inplace=True)

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

#Из условной базы данных мы получим индексы работ с длительностями и связи
act_time = in_wrk['time'].values
act_id   = np.array(in_wrk['N'], dtype=np.uint16)
lnk_src  = np.array(src, dtype=np.uint16)
lnk_dst  = np.array(dst, dtype=np.uint16)
lnk_id   = np.array(list(range(len(lnk_src))), dtype=np.uint16)

#Строим сетевую модель
prj_lnk, net_evt, net_act = net.create(act_id, act_time, lnk_id, lnk_src, lnk_dst)

#Считаем критический путь, резервы времени, слои графа сетевой модели
net.do_cpm(net_evt, net_act)

#Создаем граф для визуализации сетевой модели проекта
viz_nodes, viz_edges = net.create_viz_graph(net_act, net_evt)

#with pd.ExcelWriter('data/output/project_template.xlsx', mode='a', engine='openpyxl') as writer:
#    net_act.sort_index(inplace=True)
#    net_act.to_excel(writer, sheet_name='Acvitities')
#    net_evt.to_excel(writer, sheet_name='Events')

#TODO: Это надо переписать на Си/Cython
def optimize_this(p):
    nn, = p.shape
    assert nn == len(viz_nodes)
    #Значение функции стоимости
    cost = 0.0

    #Держим дистанцию между нодами
    for i in range(1, nn - 1):
        for j in range(i + 1, nn - 1):
            if viz_nodes.layer.iat[i] == viz_nodes.layer.iat[j]:
                cost += 1.0 / (abs(p[i] - p[j]) + 0.1)
            else:
                break

    #Считаем длину пути и количество пересечений связей
    viz_nodes['y'] = p
    ne = len(viz_edges)
    n_cross = 0
    l_graph = 0.0

    for i in range(ne):
        si = viz_edges.src.iat[i]
        di = viz_edges.dst.iat[i]
        l_graph += viz_edges.w.iat[i] * np.sqrt(1.0 + np.square(viz_nodes.y.at[di] - viz_nodes.y.at[si]))

        for j in range(i + 1, ne):
            sj = viz_edges.src.iat[j]
            dj = viz_edges.dst.iat[j]

            #Тут нет пересечений
            if si == sj or di == dj:
                continue

            #Поиск пересечений
            if viz_nodes.layer.at[si] == viz_nodes.layer.at[sj]:
                if viz_nodes.y.at[si] > viz_nodes.y.at[sj]:
                    if viz_nodes.y.at[di] <= viz_nodes.y.at[dj]:
                        n_cross += 1
                else:
                    if viz_nodes.y.at[di] >= viz_nodes.y.at[dj]:
                        n_cross += 1

    cost += l_graph * (n_cross + 1.0)
    return cost

NN = len(viz_nodes)

# from sko.GA import GA
# ga = GA(func=optimize_this, n_dim=NN, size_pop=50, max_iter=100, prob_mut=0.001,
#         lb=np.zeros((NN,)), ub=np.ones((NN,)), precision=1e-7)

# best_x, best_y = ga.run()
# print('best_x:', best_x, '\n', 'best_y:', best_y)

from sko.DE import DE


de = DE(func=optimize_this, n_dim=NN, size_pop=50, max_iter=200,
        lb=np.zeros((NN,)), ub=np.ones((NN,)))

best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

# from sko.SA import SA

# sa = SA(func=optimize_this, x0=np.random.rand(NN), T_max=1, T_min=1e-9, L=300, max_stay_counter=150,
#         lb=np.zeros((NN,)), ub=np.ones((NN,)))

# best_x, best_y = sa.run()
# print('best_x:', best_x, 'best_y', best_y)
