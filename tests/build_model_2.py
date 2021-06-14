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
import igraph as ig

import matplotlib.pyplot as plt


def get_network_model(wrk):
    '''
    Parameters
    ----------
    wrk : pd.DataFrame
        Work dependency .

    Returns
    -------
    int
        Zero on success or negative error number
    list
        list real work descriptions.
    list
        list imaginary work decriptions.
    list
        Event descriprions.
    '''
    if not isinstance(wrk, pd.DataFrame):
        return -1,None,None,None
    #
    #Real works start and end events
    print(wrk)
    wrk_id = max(wrk.index) + 1 #Next work id
    #
    #Make links dataframe
    src = []
    dst = []
    for d in wrk.index:
        for s in wrk.dep[d]:
            src.append(s)
            dst.append(d)
    links = pd.DataFrame(data={'src':src, 'dst':dst})
    src = None
    dst = None
    lnk_id = max(links.index) + 1 #Next link id
    #
    #Check the graph
    assert ig.Graph.DataFrame(links, directed=True).is_dag()

    #
    #Create initial events
    #Project start end end events are added by default
    evt = pd.DataFrame(data={'wrkIn':[[],[]], 'wrkOut':[[],[]]}, index=[0,1])
    #Next event id
    evt_id = 2 #Next event id
    wrk['srcEvt'] = int(-1); #src evt
    wrk['dstEvt'] = int(-1); #dst evt
    wrk['isReal'] = True; #dst evt
    for i in wrk.index:
        if i not in list(links.dst):
            #This is a starting work, it has no previous works
            evt.wrkOut.at[0].append(i)
            wrk.srcEvt.at[i] = 0
        else:
            #add new src event
            new_evt = pd.DataFrame(data={'wrkIn': [[]], 'wrkOut': [[i]]}, index=[evt_id])
            evt = pd.concat([evt, new_evt])
            wrk.srcEvt.at[i] = evt_id
            evt_id += 1
        #
        if i not in list(links.src):
            #this is an ending work, it has no successors
            evt.wrkIn.at[1].append(i)
            wrk.dstEvt.at[i] = 1
        else:
            #add new dst event
            new_evt = pd.DataFrame(data={'wrkIn': [[i]], 'wrkOut': [[]]}, index=[evt_id])
            evt = pd.concat([evt, new_evt])
            wrk.dstEvt.at[i] = evt_id
            evt_id += 1
    #
    #retarget links to events
    for i in links.index:
        links.src.at[i] = wrk.dstEvt[links.src[i]]
        links.dst.at[i] = wrk.srcEvt[links.dst[i]]

    evt['toDrop'] = False
    links['toDrop'] = False

    def traverse_in_links(i):
        lnk = links[links.dst == i].sort_values(by=['src'])
        evt.lnkIn.at[i] = list(lnk.index)
        evt.evtIn.at[i] = list(lnk.src)
        if evt.lnkIn[i]:
            evt.skipIn.at[i] = False
            evt.lenIn.at[i] = len(evt.lnkIn[i])

    def traverse_links():
        evt['lnkIn'] = None
        evt['evtIn'] = None
        evt['lnkOut'] = None
        evt['evtOut'] = None
        evt['skipIn'] = True
        evt['skipOut'] = True
        evt['lenIn'] = int(0)
        for i in evt[evt.toDrop == False].index:
            traverse_in_links(i)

            lnk = links[links.src == i].sort_values(by=['dst'])
            evt.lnkOut.at[i] = list(lnk.index)
            evt.evtOut.at[i] = list(lnk.dst)
            if evt.lnkOut[i]:
                evt.skipOut.at[i] = False

    #Merge events with common link lists
    def merge_evts_by_links(d):

        if d == 'In':
            nd = 'Out' #Inverted direction
            me = 'src' #Work event prefix
        elif d == 'Out':
            nd = 'In'
            me = 'dst'
        else:
            raise ValueError('d must be \"In\" or \"Out\"')

        print(d)

        merged = 0

        evt_idx = evt[evt['skip' + d] == False].index
        n = len(evt_idx)
        for k in range(n):
            i = evt_idx[k]
            if evt.toDrop[i]:
                continue
            #
            pivot = evt.at[i, 'evt' + d]
            #
            for m in range(k + 1, n):
                j = evt_idx[m]
                if evt.toDrop[j]:
                    continue

                current = evt.at[j, 'evt' + d]

                if current == pivot:
                    # events must not have Out links here
                    assert evt.at[j, 'skip' + nd]
                    # Will drop event and its links
                    print(i,j)
                    evt.toDrop.at[j] = True
                    links.loc[evt.at[j, 'lnk' + d], 'toDrop'] = True

                    #Merge events:
                    evt.wrkIn.at[i] = evt.wrkIn[i] + evt.wrkIn[j]
                    evt.wrkOut.at[i] = evt.wrkOut[i] + evt.wrkOut[j]

                    #Correct event links in works
                    wrk.loc[evt.at[i, 'wrk' + nd], me + 'Evt'] = i

                    merged += 1

        return merged

    traverse_links()

    while True:

        merge_evts_by_links('In')
        merge_evts_by_links('Out')

        #Drop garbage
        links.drop(labels=links[links.toDrop == True].index, axis=0, inplace=True)
        evt.drop(labels=evt[evt.toDrop == True].index, axis=0, inplace=True)

        traverse_links()

        if max(list(evt.lenIn)) < 2:
            break

        print('Cut z-configurations...')

        evt_idx = evt[(evt.skipIn == False) & (evt.toDrop == False)].sort_values(by=['lenIn']).index
        n = len(evt_idx)

        for k in range(n):
            i = evt_idx[k]
            if evt.toDrop[i]:
                continue

            pivot = evt.evtIn[i]

            for e in pivot:
                m = k + 1
                #while m < n:
                for m in range(k + 1, n):
                    j = evt_idx[m]
                    if evt.toDrop[j]:
                        continue

                    current = evt.evtIn[j]
                    if e in current:
                        #z link
                        z_lnk_id = evt.lnkIn[j][current.index(e)]

                        #create a dummy work and its events
                        new_wrk = pd.DataFrame(data={'dep': [[]],
                                                      'srcEvt': [evt_id],
                                                      'dstEvt': [evt_id + 1],
                                                      'isReal': [False]},
                                                index=[wrk_id])


                        new_evt = pd.DataFrame(data={'wrkIn':   [[],                    [wrk_id]],
                                                     'wrkOut':  [[wrk_id],              []      ],
                                                     'toDrop':  [False,                 False   ],
                                                     'skipIn':  [False,                 True    ],
                                                     'skipOut': [True,                  False   ],
                                                     'lnkIn':   [[z_lnk_id],            []      ],
                                                     'lnkOut':  [[],                    [lnk_id]],
                                                     'evtIn':   [[links.src[z_lnk_id]], []      ],
                                                     'evtOut':  [[],                    [j]     ],
                                                     'lenIn':   [1,                     0       ]
                                                      },
                                                index=[evt_id, evt_id + 1])

                        new_lnk = pd.DataFrame(data={'toDrop':  [False],
                                                      'src': [evt_id + 1],
                                                      'dst': [j]},
                                                index=[lnk_id])

                        wrk = pd.concat([wrk, new_wrk])
                        evt = pd.concat([evt, new_evt])
                        links = pd.concat([links, new_lnk])
                        #edit z-link
                        links.dst.at[z_lnk_id] = evt_id
                        #edit j event
                        traverse_in_links(j)

                        evt_id += 2
                        wrk_id += 1
                        lnk_id += 1

                        print(e, i, j, z_lnk_id)

                    m += 1

    #reduce remaining links
    for i in links.index:
        #Merge events:
        s = links.src[i]
        d = links.dst[i]
        evt.wrkIn.at[d] = evt.wrkIn[d] + evt.wrkIn[s]
        evt.wrkOut.at[d] = evt.wrkOut[d] + evt.wrkOut[s]
        evt.toDrop.at[s] = True

        wrk.loc[evt.at[s, 'wrkIn'], 'dstEvt'] = d

    links = None

    evt.drop(labels=evt[evt.toDrop == True].index, axis=0, inplace=True)
    evt.drop(labels=['toDrop', 'lnkIn', 'lnkOut', 'evtIn',
                     'evtOut', 'skipIn', 'skipOut', 'lenIn'],
             axis=1, inplace=True)

    #Add dummys for identical works
    print('Adding dummys...')
    n = len(wrk.index)
    for m in range(n):

        i = wrk.index[m]
        for k in range(m + 1, n):

            j = wrk.index[k]
            if wrk.srcEvt[i] == wrk.srcEvt[j] and wrk.dstEvt[i] == wrk.dstEvt[j]:
                print(i,j)
                #
                #create a dummy work and its events
                new_wrk = pd.DataFrame(data={'dep': [[]],
                                              'srcEvt': [evt_id],
                                              'dstEvt': [wrk.dstEvt[i]],
                                              'isReal': [False]},
                                        index=[wrk_id])


                new_evt = pd.DataFrame(data={'wrkIn':   [[j]],
                                              'wrkOut':  [[wrk_id]]},
                                        index=[evt_id])
                wrk = pd.concat([wrk, new_wrk])
                evt = pd.concat([evt, new_evt])

                wrk.dstEvt.at[j] = evt_id

                evt_id += 1
                wrk_id += 1

    #enumerate events
    n = len(evt.index)
    for m in range(n):
        i = evt.index[m]
        wrk.loc[wrk.srcEvt == i, 'srcEvt'] = m
        wrk.loc[wrk.dstEvt == i, 'dstEvt'] = m

    evt.reset_index(inplace=True, drop=True)

    return wrk,evt





if __name__ == '__main__':
    #Матрица связности работ, в каждой строке замаскированы предшествующие работы
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
        [5,19,11,13,15], #[5,19,11,15] #УУУУПС!!!
        [3,8,9]
        ]}, index = list(range(1,22)))

    w,ev = get_network_model(_wrk)
    print(w)
    print(ev)

    #fig, ax = plt.subplots()
    gdf = w[['srcEvt', 'dstEvt', 'isReal']].copy()
    gdf['id'] = list(gdf.index)
    g = ig.Graph.DataFrame(gdf, directed=True)
    l = g.layout('sugiyama', hgap=1, vgap=0.5, maxiter=500)
    l.rotate(270)

    color_dict = {True: (0,255,0), False:(50,50,50)}

    visual_style = {
        'layout'      : l,
        'bbox':(0,0,1500,1000),
        'vertex_label': g.vs['name'],
        'vertex_color':'pink',
        #'edge_color':[color_dict[r] for r in g.es["isReal"]],
        'edge_width': [1 + 2 * int(r) for r in g.es["isReal"]],
        'edge_label': g.es['id'],
        'edge_curved':False
        }
    ig.plot(g, 'pot.png', **visual_style)