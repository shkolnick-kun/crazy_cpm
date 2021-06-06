#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 18:41:01 2020

@author: anon
"""
import numpy as np
import pandas as pd

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
        list dummy work decriptions.
    list
        Event descriprions.
    '''
    if not isinstance(wrk, pd.DataFrame):
        return -1,None,None,None
    #
    n = len(wrk)
    #Real works start and end events
    wrk['start'] = [-1]*n           #Start event
    wrk['end']   = wrk.start.copy() #End event
    wrk['wk0']   = wrk.dep.apply(lambda x: len(x)) #Number dependencies on start
    wrk['wki']   = wrk.wk0.copy() #Number of unprocessed dependencies
    wrk['skip']  = [False] * n   #Work processing flags
    #step 6 sort
    # There is an error in the source article!
    # We must sort the data first to ensure that any detected grpup of works
    #   have the same dependecy list.
    #step 1
    evt_id = 1
    chk_wrk = []
    dummys = []
    #step 2
    for i in wrk.index:
        if wrk.wki.at[i] == 0:
            wrk.start.at[i] = evt_id
            wrk.skip.at[i] = True
            chk_wrk.append(i)
        else:
            wrk.dep.at[i].sort()

    wrk.sort_values(by='wk0', inplace=True)
    #
    print(chk_wrk)
    evt_id += 1
    #step 3
    j = 0
    while True:
        print('Check work: ', chk_wrk[j])
        #Get groups of started works
        gwrk  = [] #List of group of works with common predeceptors
        gpred = [] #List of lists of predeceptors for groups of works
        for i in wrk.index:
            if wrk.skip.at[i]:
                continue

            if chk_wrk[j] in wrk.dep.at[i]:
                wrk.wki.at[i] -= 1

            if wrk.wki.at[i] == 0:
                if wrk.dep.at[i] not in gpred: #O(n^2)
                    gpred.append(wrk.dep.at[i])
                    gwrk.append([i])
                else:
                    gwrk[gpred.index(wrk.dep.at[i])].append(i)
        #

        if gpred:
            print('Groups of works:', gwrk)
            print('Predeceptors:', gpred)
            #iterate over predeceptors lists
            for k, dl in enumerate(gpred): #Will process a
                print('Process group: ', dl, gwrk[k])
                #Will check a group of predeceptors
                fin_dummy = -1 #A dummy start for finished subgroup of works
                #Predeceptor subgroups
                sg_starts = [] #List of tart events
                sg_works = []  #List of work lists
                for i in dl:
                    if wrk.end.at[i] > 0:
                        fin_dummy = max(fin_dummy, wrk.end.at[i])
                    elif wrk.start.at[i] not in sg_starts: #O(n)
                        sg_starts.append(wrk.start.at[i])
                        sg_works.append([i])
                    else:
                        sg_works[sg_starts.index(wrk.start.at[i])].append(i)

                if fin_dummy > 0:
                    #We have a group of finished works in predeceptor list
                    dummy_starts = [fin_dummy] #Start a dummy work
                else:
                    dummy_starts = []

                #Sort subgroups by start event
                subgroups = list(zip(sg_starts, sg_works))
                subgroups.sort()#O(n*log(n))
                print('Subgroups: ', subgroups)

                #iterate over subgroups
                strait_works = [] #Works woth no dummy successors
                for sgs, sgw in subgroups:
                    strait_works.append(sgw[0])
                    if len(sgw) > 1:
                        for i in sgw[1:]:
                            wrk.end.at[i] = evt_id
                            dummy_starts.append(evt_id)
                            evt_id += 1

                #Finalize a group list processing
                for i in gwrk[k]:
                    wrk.start.at[i] = evt_id
                    wrk.skip.at[i]  = True

                for i in strait_works:
                    wrk.end.at[i] = evt_id

                dummys += list(zip(dummy_starts, [evt_id]*len(dummy_starts)))
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
        if not wrk.skip.at[i]:
            return -2,None,None,None

    wrk.loc[wrk.end== -1, 'end'] = evt_id
    print(wrk)
    #TODO Make a DataFrame from dummy works!!!

    return -4,None,None,None





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
        [3,8,9],
        ]}, index = list(range(1,22)))

    er,w,f,ev = get_network_model(_wrk)
    print(er)
    print(w)
    print(f)
    print(ev)