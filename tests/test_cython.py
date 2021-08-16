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


from sko.DE import DE
from ccpm import vizGraphLoss

loss = vizGraphLoss(np.array(viz_nodes.index), viz_nodes.layer.values,
                    viz_edges.src.values, viz_edges.dst.values, viz_edges.w.values)


NN = len(viz_nodes)
de = DE(func=loss.run, n_dim=NN, size_pop=NN*10, max_iter=NN*100,
        lb=np.zeros((NN,)), ub=np.ones((NN,)))

best_x, best_y = de.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)

viz_nodes['y'] = best_x
