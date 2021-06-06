#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:41:01 2020

@author: anon
"""
#import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

#import igraph as ig

def get_network_model(wrk):
    '''
    Parameters
    ----------
    wrk : pd.DataFrame
        Work depdstency .

    Returns
    -------
    int
        Zero on success or negative error number
    list
        list real work descriptions.
    list
        list dummy work decriptions.
    list
        Event descriprions.
    '''
    if not isinstance(wrk, pd.DataFrame):
        return -1,None,None
    #
    n = len(wrk)
    #Real works src and dst events
    wrk['src']      = [-1]*n           #src event
    wrk['dst']      = wrk.src.copy() #dst event
    wrk['ndep']     = wrk.dep.apply(lambda x: len(x)) #Number depdstencies on src
    wrk['rem_dep']  = wrk.ndep.copy() #Number of unprocessed depdstencies
    wrk['dep_map']  = wrk.dep.apply(lambda x: [False] +
                                    [(lambda y: True if y in x else False)(i)
                                     for i in wrk.index])
    wrk['started']  = [False] * n  #Work was started
    wrk['straight'] = [True]*n     #Staight work
    #step 6 sort
    # There is an error in the source article!
    # We must sort the data first to ensure that any detected grpup of works
    #   have the same depdstecy list.
    #step 1
    evt_id = 1
    chk_wrk = []
    dummys = []
    #step 2
    for i in wrk.index:
        if wrk.rem_dep.at[i] == 0:
            wrk.src.at[i] = evt_id
            wrk.started.at[i] = True
            chk_wrk.append(i)
        else:
            wrk.dep.at[i].sort()

    wrk.sort_values(by='ndep', inplace=True)
    #
    print(chk_wrk)
    evt_id += 1
    #step 3
    j = 0
    while True: #O(n)
        print('Check work: ', chk_wrk[j])
        #Get groups of srced works
        gwrk  = [] #List of group of works with common predeceptors
        gpred = [] #List of lists of predeceptors for groups of works
        for i in wrk.index: #O(n^2)
            if wrk.started.at[i]:
                continue

            if wrk.dep_map.at[i][chk_wrk[j]]: #O(n^2)
                wrk.rem_dep.at[i] -= 1

            if wrk.rem_dep.at[i] == 0:
                if wrk.dep.at[i] not in gpred: #O(n^3)
                    gpred.append(wrk.dep.at[i])
                    gwrk.append([i])
                else:
                    gwrk[gpred.index(wrk.dep.at[i])].append(i)
        #

        if gpred:
            print('Groups of works:', gwrk)
            print('Predeceptors:', gpred)
            #iterate over predeceptors lists
            for k,dl in enumerate(gpred): #O(n^2)
                print('Process group: ', dl, gwrk[k])
                #Will check a group of predeceptors
                dummy_srcs = [] #A dummy work list
                dummy_map = [False]*2*n #dummy_srcs map, max len() is 2*n
                #Unfinished predeceptor subgroups
                sg_srcs = [] #List of start events
                sg_map = [False]*2*n #sg_map map, max len() is 2*n
                straight_works = [] #Works woth no dummy successors
                for i in dl: #O(n^3)
                    if wrk.dst.at[i] > 0:
                        #Check finished predeceptors
                        if wrk.straight.at[i]:
                            if not dummy_map[wrk.dst.at[i]]: #O(n^3) Fuck yeah!!!
                                dummy_map[wrk.dst.at[i]] = True
                                dummy_srcs.append(wrk.dst.at[i])

                    elif not sg_map[wrk.src.at[i]]: #O(n^3) Fuck yeah!!!
                        #Start some subgroup
                        sg_map[wrk.src.at[i]] = True
                        sg_srcs.append(wrk.src.at[i])
                        straight_works.append(i)
                    else:
                        #Continue some subgroup
                        wrk.straight.at[i] = False
                        wrk.dst.at[i] = evt_id
                        dummy_srcs.append(evt_id)
                        evt_id += 1

                print('Dummy srcs: ', dummy_srcs)

                #Finalize a group list processing
                for i in gwrk[k]: #O(n^3)
                    wrk.src.at[i] = evt_id
                    wrk.started.at[i]  = True

                for i in straight_works: #O(n^3)
                    wrk.dst.at[i] = evt_id

                dummys += list(zip(dummy_srcs, [evt_id]*len(dummy_srcs)))
                print(wrk)
                print(dummys)
                evt_id += 1
                chk_wrk += gwrk[k]
                print(chk_wrk)

        #step 20
        j += 1
        #step 19
        if j >= len(chk_wrk):
            break

    #step 21
    for i in wrk.index:
        if not wrk.started.at[i]:
            return -2,None,None

    wrk.loc[wrk.dst== -1, 'dst'] = evt_id
    wrk.drop(labels=['ndep', 'rem_dep', 'started', 'straight', 'dep_map'],
             axis=1, inplace=True)
    wrk.sort_values(by='dst', inplace=True)
    wrk.sort_values(by='src', inplace=True)

    print('Works:\n', wrk)

    #Reduce dummy works
    dummys = list(zip(*dummys))
    dummys = pd.DataFrame(data={'src': dummys[0], 'dst':dummys[1]})
    dummys.sort_values(by='src', inplace=True) #O(n^3)

    dummys['to_drop'] = [False]*len(dummys)

    for k,j in enumerate(dummys.index): #O(n)
        if dummys.to_drop.at[j]:
            continue

        src = dummys.src.at[j]
        pivot = dummys.dst.at[j]
        for i in dummys.index[k+1:]: #O(n^2)
            if dummys.src.at[i] != src:
                break

            current = dummys.dst.at[i]
            if len(dummys[(dummys.src == pivot) & (dummys.dst == current)]): #O(n^3)
                dummys.to_drop.at[i] = True

    dummys.drop(labels=dummys[dummys.to_drop == True].index, axis=0, inplace=True)
    dummys.drop(labels=['to_drop'], axis=1, inplace=True)
    dummys.reset_index(drop=True, inplace=True)

    print('Dummys:\n', dummys)

    #Done!!!
    return 0,wrk.copy(),dummys

if __name__ == '__main__':
    _wrk = pd.DataFrame(data = {'dep':[
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
        ]}, index = list(range(1,22)))

    # _wrk = pd.DataFrame(data = {'dep':[
    #     [],
    #     [],
    #     [],
    #     [1],
    #     [1,2,3],
    #     [3],
    #     [4,5],
    #     [6],
    #     [7,8]
    #     ]}, index = list(range(1,10))) #Too much dummty works!!!

    # _wrk = pd.DataFrame(data = {'dep':[
    #     [],
    #     [],
    #     [],
    #     [],
    #     [1],
    #     [1,2],
    #     [1,2,3],
    #     [1,2,3,4],
    #     [5,6,7,8],
    #     ]}, index = list(range(1,10))) #Too much dummty works!!!

    # _wrk = pd.DataFrame(data = {'dep':[
    #     [],
    #     [],
    #     [],
    #     [1,6],
    #     [4],
    #     [5],
    #     [1,2,3],
    #     [1,2,3,4],
    #     [5,6,7,8],
    #     ]}, index = list(range(1,10))) #Loop!!!

    er,w,f = get_network_model(_wrk)

    print(er)
    print(w)
    print(f)

    gdf = w[['src','dst']].copy()
    gdf['width'] = 1.0
    gdf['label'] =list(gdf.index)

    f['width'] = 0.5
    f['label'] = ''

    gdf = gdf.append(f, ignore_index=True).sort_values(by='src')


    G=nx.from_pandas_edgelist(gdf,
                              source='src', target='dst', edge_attr=['label', 'width'],
                              create_using=nx.DiGraph())

    #top = nx.bipartite.sets(G)[0]
    #Использовать multipartite_layout и адгоритм разбиения на слои с помощью висячих вершин

    pos = nx.layout.spiral_layout(G)

    nodes = nx.draw_networkx_nodes(G, pos, node_color="pink")#, node_size=500)
    edges = nx.draw_networkx_edges(G, pos,
                                    arrowstyle="->",
                                    arrowsize=10,
                                    #min_source_margin=500,
                                    #min_target_margin=500,
                                    width=[G[u][v]['width'] for u,v in G.edges()]
                                    )
    labels = nx.draw_networkx_labels(G, pos, labels=None, font_size=12)
    #elabels = nx.draw_networkx_edge_labels(G,pos)

    ax = plt.gca()
    ax.set_axis_off()
    plt.show()