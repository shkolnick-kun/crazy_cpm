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

###############################################################################
from ccpm import vizGraphLoss
loss = vizGraphLoss(np.array(viz_nodes.index), viz_nodes.layer.values,
                    viz_edges.src.values, viz_edges.dst.values, viz_edges.w.values)

from sko.DE import DE

#Этот вариант лучше и быстрее работает
NN = len(viz_nodes)
de = DE(func=loss.run, n_dim=NN, size_pop=NN*10, max_iter=NN*100,
        lb=np.zeros((NN,)), ub=np.ones((NN,)))
best_x, best_y = de.run()

#Этот вариант работает медленно и дает нестабильный результат.
#from scipy.optimize import differential_evolution
# result = differential_evolution(loss.run,
#                                 popsize=NN*2,
#                                 maxiter=NN*100,
#                                 mutation=.5,
#                                 recombination=.3,
#                                 bounds=[(0., 1.)]*NN)
#best_x, best_y = result.x, result.fun

print('best_x:', best_x, '\n', 'best_y:', best_y)

#print(viz_nodes['place'].max()+1)

#TODO: Проработать преобразование
viz_nodes['place'] = best_x * viz_nodes['place'].max()
###############################################################################
from pyvis.network import Network

net = Network(viz_nodes['place'].max() * 100 + 70,
              float(viz_nodes['layer'].max()) * 100 + 70,
              directed=True)
#------------------------------------------------------------------------------
#Метки
def get_node_label(i):
    if not viz_nodes.is_evt.at[i]:
        return ' '

    return '%d\n%d\N{EN SPACE}\N{EN SPACE}\N{EN SPACE}%d\n%d'%(i,
                                net_evt.early.at[i],
                                net_evt.late.at[i],
                                net_evt.reserve.at[i])
def get_edge_label(i):
    i = viz_edges.act.at[i]
    return str(i) + '\n' + str(net_act.time.at[i])

#------------------------------------------------------------------------------
#Заголовки
def get_node_title(i):
    if not viz_nodes.is_evt.at[i]:
        return ' '

    ret =  'Event:' + str(i)
    ret += '\N{LINE FEED}Early date:' + str(net_evt.early.at[i])
    ret += '\N{LINE FEED}Late  date:' + str(net_evt.late.at[i])
    ret += '\N{LINE FEED}Time reserve:' + str(net_evt.reserve.at[i])
    return ret

def get_edge_title(i):
    i = viz_edges.act.at[i]
    ret =  'Activity:'       + str(i)
    ret += '\N{LINE FEED}Early start:'  + str(net_act.estart.at[i])
    ret += '\N{LINE FEED}Early finish:' + str(net_act.efinish.at[i])
    ret += '\N{LINE FEED}Late start:'   + str(net_act.lstart.at[i])
    ret += '\N{LINE FEED}Late finish:'  + str(net_act.lfinish.at[i])
    ret += '\N{LINE FEED}Time reserve:' + str(net_act.reserve.at[i])
    return ret

#------------------------------------------------------------------------------
#Цвета
VIS_CL_GREEN = np.array([0x7f, 0xc2, 0x82])
VIS_CL_RED   = np.array([0xd7, 0x9c, 0x9c])

def get_node_color(i):
    if not viz_nodes.is_evt.at[i]:
        return '#808080'

    x = float(net_evt.reserve.at[i])/float(net_evt.reserve.max())
    cl = np.around(VIS_CL_GREEN * x + VIS_CL_RED * (1. - x))
    return '#%02x%02x%02x'%(int(cl[0]), int(cl[1]), int(cl[2]))

def get_edge_color(i):
    i = viz_edges.act.at[i]
    x = float(net_act.reserve.at[i])/float(net_act.reserve.max())
    cl = np.around(VIS_CL_GREEN * x + VIS_CL_RED * (1. - x))
    return '#%02x%02x%02x'%(int(cl[0]), int(cl[1]), int(cl[2]))

#------------------------------------------------------------------------------
for i in viz_nodes.index:
    net.add_node(i,
                 hidden=not viz_nodes.is_evt.at[i],
                 title=get_node_title(i),
                 x= float(viz_nodes.layer.at[i] * 100) - float(viz_nodes['layer'].max()) * 50,
                 y= viz_nodes.place.at[i] * 100 - viz_nodes['place'].max() * 50,
                 label=get_node_label(i),
                 color=get_node_color(i),
                 physics=False, shape='circle', borderWidth=2)


for i in viz_edges.index:
    net.add_edge(int(viz_edges.src.at[i]), int(viz_edges.dst.at[i]),
                 title=get_edge_title(i),
                 label=get_edge_label(i), #TOSO: Продумать, на каком из кусков связи размещвть label
                 color=get_edge_color(i),
                 width=2)
net.show('data/output/net.html')
