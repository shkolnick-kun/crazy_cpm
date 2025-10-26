#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    CrazyCPM
    Copyright (C) 2025 anonimous

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
import graphviz
import numpy as np
import _ccpm

#==============================================================================
class _Activity:  
    def __init__(self, id, wbs_id, model, src, dst, duration=0.0):
        assert isinstance(id,       int)
        assert isinstance(wbs_id,   int)
        assert isinstance(model,    NetworkModel)
        assert isinstance(src,      _Event)
        assert isinstance(dst,      _Event)
        assert isinstance(duration, float)
        assert duration >= 0.

        self.id       = id
        self.wbs_id   = wbs_id
        self.model    = model
        self.src      = src
        self.dst      = dst
        self.duration = duration

        self.early_start = 0.0
        self.late_start  = 0.0
        self.early_end   = 0.0
        self.late_end    = 0.0
        self.reserve     = 0.0

    #----------------------------------------------------------------------------------------------
    def __repr__(self):
        return 'Activity(id=%r, src_id=%r, dst_id=%r, duration=%r, reserve=%r, wbs_id=%r)' % (
            self.id,
            self.src.id,
            self.dst.id,
            self.duration,
            self.reserve,
            self.wbs_id,
            )
    
#==============================================================================
class _Event:    
    def __init__(self, id, model):
        assert isinstance(id,     int)
        assert isinstance(model, NetworkModel)

        self.id = id
        self.model   = model
        self.early   = 0.0
        self.late    = 0.0
        self.reserve = 0.0
        self.stage   = 0
        
    @property
    def in_activities(self):
        return [a for a in self.model.activities if a.dst == self]

    @property
    def out_activities(self):
        return [a for a in self.model.activities if a.src == self]

    #--------------------------------------------------------------------------
    def __repr__(self):
        return 'Event(id=%r early=%r late=%r reserve=%r stage=%r)' % (
            self.id,
            self.early,
            self.late,
            self.reserve,
            self.stage
        )

#==============================================================================
class NetworkModel:
    def __init__(self, wbs_dict, lnk_src=None, lnk_dst=None, links=None):
        """
        Initialize NetworkModel with multiple link formats support
        
        Parameters:
        -----------
        wbs_dict : dict
            Work Breakdown Structure dictionary
        lnk_src, lnk_dst : array-like, optional
            Old format: separate source and destination arrays
        links : various formats, optional
            New formats:
            - Format 1: two rows [[src1, src2, ...], [dst1, dst2, ...]]
            - Format 2: two columns [[src1, dst1], [src2, dst2], ...]
            - Format 3: dictionary {'src': [src1, src2, ...], 'dst': [dst1, dst2, ...]}
        """
        assert isinstance(wbs_dict, dict)
        
        # Parse links into standard format
        lnk_src, lnk_dst = self._parse_links(lnk_src, lnk_dst, links)
        
        assert len(lnk_src) == len(lnk_dst)

        self.events     = []
        self.next_act   = 1
        self.activities = []

        #Generate network graph
        act_ids = np.array(list(wbs_dict.keys()), dtype=int)

        status, net_src, net_dst, lnk_src, lnk_dst = _ccpm.compute_aoa(act_ids, lnk_src, lnk_dst)
        assert 0 == status

        for i in range(np.max(net_dst)):
            self._add_event(int(i + 1))

        na = len(act_ids)
        d  = np.max(act_ids)
        for i in range(len(net_src)):
            if i < na:
                self._add_activity(int(act_ids[i]), int(net_src[i]), int(net_dst[i]), 
                                  wbs_dict[act_ids[i]].get('duration', 0.))
            else:
                #Add a dummy
                d += 1
                self._add_activity(0, int(net_src[i]), int(net_dst[i]), 0.)
                
        #Compute Event and Actions attributes
        assert 0 < len(self.events)

        self._cpm_compute('stage')

        self._cpm_compute('early')

        l = max([e.early for e in self.events])
        for e in self.events:
            e.late = l

        self._cpm_compute('late')

        for e in self.events:
            e.reserve = e.late - e.early

        for a in self.activities:
            a.early_end = a.early_start + a.duration
            a.late_end  = a.late_start  + a.duration
            a.reserve   = a.late_start  - a.early_start

    #--------------------------------------------------------------------------
    def _parse_links(self, lnk_src, lnk_dst, links):
        """
        Parse links from various formats into standard lnk_src, lnk_dst arrays
        """
        # Case 1: Old format (lnk_src and lnk_dst provided)
        if lnk_src is not None and lnk_dst is not None:
            return np.asarray(lnk_src), np.asarray(lnk_dst)
        
        # Case 2: New formats via links parameter
        if links is None:
            raise ValueError("Either (lnk_src, lnk_dst) or links must be provided")
        
        # Format 1: Two rows [[src...], [dst...]]
        if (isinstance(links, (list, tuple)) and len(links) == 2 and 
            isinstance(links[0], (list, tuple, np.ndarray)) and
            isinstance(links[1], (list, tuple, np.ndarray))):
            return np.asarray(links[0]), np.asarray(links[1])
        
        # Format 2: Two columns [[src, dst], [src, dst], ...]
        elif (isinstance(links, (list, tuple, np.ndarray)) and 
              len(links) > 0 and
              isinstance(links[0], (list, tuple, np.ndarray)) and
              len(links[0]) == 2):
            src_list = [item[0] for item in links]
            dst_list = [item[1] for item in links]
            return np.asarray(src_list), np.asarray(dst_list)
        
        # Format 3: Dictionary {'src': [...], 'dst': [...]}
        elif isinstance(links, dict):
            if 'src' in links and 'dst' in links:
                return np.asarray(links['src']), np.asarray(links['dst'])
            else:
                raise ValueError("Dictionary links must contain 'src' and 'dst' keys")
        
        else:
            raise ValueError(f"Unsupported links format: {type(links)}")

    #--------------------------------------------------------------------------
    def _add_event(self, i):
        self.events.append(_Event(i, self))
    
    #--------------------------------------------------------------------------
    def _add_activity(self, wbs_id, src_id, dst_id, duration):
        assert isinstance(wbs_id,   int)
        assert isinstance(dst_id,   int)
        assert isinstance(wbs_id,   int)
        assert isinstance(duration, float)

        act = _Activity(self.next_act, wbs_id, self, self.events[src_id-1], 
                        self.events[dst_id-1
                                    ], duration)
        self.activities.append(act)
        self.next_act += 1

    #--------------------------------------------------------------------------
    def _cpm_compute(self, target=None):
        if 'early' == target:
            act_base     = 'early_start'
            act_new      = 'early_finish'
            act_next     = 'dst'
            fwd          = 'out_activities'
            rev          = 'in_activities'
            delta        = lambda a : a.duration
            choise       = max
        elif 'stage' == target:
            act_base     = None
            act_new      = None
            act_next     = 'dst'
            fwd          = 'out_activities'
            rev          = 'in_activities'
            delta        = lambda a : 1
            choise       = max
        elif 'late' == target:
            act_base     = 'late_finish'
            act_new      = 'late_start'
            act_next     = 'src'
            fwd          = 'in_activities'
            rev          = 'out_activities'
            delta        = lambda a : - a.duration
            choise       = min
        else:
            raise ValueError("Unknown 'target' value!!!")
        
        if 'stage' != target:
            for a in self.activities:
                setattr(a, act_base, -1)
                setattr(a, act_new,  -1)
           
        n_dep = [len(getattr(e, rev)) for e in self.events]

        evt = [i for i,n in enumerate(n_dep) if 0 == n]
        assert 1 == len(evt)

        i = 0
        while True:

            e = self.events[evt[i]]
            base_val = getattr(e, target)

            for a in getattr(e, fwd):

                if act_base:
                    setattr(a, act_base, base_val)
                    
                new_val = base_val + delta(a)
                
                if act_new:
                    setattr(a, act_new, new_val)
                    
                next_evt = getattr(a, act_next)
                next_i   = self.events.index(next_evt)
                
                setattr(next_evt, target, choise(getattr(next_evt, target), new_val))

                n_dep[next_i] -= 1

                if 0 >= n_dep[next_i]:
                    evt.append(next_i)

            i += 1
            if i >= len(evt):
                break

    #--------------------------------------------------------------------------
    def __repr__(self):
        _repr = 'NetworkModel:\n    Events:\n'
        for e in self.events:
            _repr += '        ' + str(e) + '\n'    
        
        _repr += '    Activities:\n'
        for a in self.activities:
            _repr += '        ' + str(a) + '\n'

        return _repr
    
    #--------------------------------------------------------------------------
    def viz_cpm(self):
        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style':'rounded'})
        dot.graph_attr['rankdir'] = 'LR'

        def _cl(res):
            if res <= 1e-6: #Absolutre precision is nonsense
                return '#ff0000'
            return '#000000'

        for e in self.events:
            dot.node(str(e.id), 
                     '{{%d |{%.1f|%.1f}| %.1f}}' % (e.id, 
                                                    e.early, 
                                                    e.late, 
                                                    e.reserve), 
                     color=_cl(e.reserve))

        for a in self.activities:

            if a.wbs_id:
                lbl  = str(a.wbs_id)
                lbl += '\n t=' + str(a.duration) + '\n r=' + str(a.reserve)
            else:
                lbl = '# \n r=' + str(a.reserve)

            dot.edge(str(a.src.id), str(a.dst.id), 
                     label=lbl, 
                     color=_cl(a.reserve),
                     style='dashed' if a.duration == 0 else 'solid'
                    )

        return dot

#==============================================================================
if __name__ == '__main__':
    
    wbs = {
        1 :{'leter':'A', 'duration':1., 'name':'Heating and frames study'                                },
        2 :{'leter':'B', 'duration':2., 'name':'Scouring and installation of building site establishment'},
        3 :{'leter':'C', 'duration':4., 'name':'Earthwork and concrete well'                             },
        4 :{'leter':'D', 'duration':4., 'name':'Earthwork and concrete longitudinal beams'               },
        5 :{'leter':'E', 'duration':6., 'name':'Frame construction'                                      },
        6 :{'leter':'F', 'duration':2., 'name':'Frame transport'                                         },
        7 :{'leter':'G', 'duration':6., 'name':'Assemblage'                                              },
        8 :{'leter':'H', 'duration':2., 'name':'Earthwork and pose drains'                               },
        9 :{'leter':'I', 'duration':5., 'name':'Heating provisioning and assembly'                       },
        10:{'leter':'J', 'duration':5., 'name':'Electric installation'                                   },
        11:{'leter':'K', 'duration':2., 'name':'Painting'                                                },
        12:{'leter':'L', 'duration':1., 'name':'Pavement'                                                }
        }
    
    # Демонстрация всех форматов:
    
    print("=== Старый формат ===")
    src_old = np.array([1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10])
    dst_old = np.array([5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12])
    n_old = NetworkModel(wbs, src_old, dst_old)
    print("Успешно создана модель со старым форматом")
    
    print("\n=== Новый формат 1 (две строки) ===")
    links_format1 = [
        [1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10],
        [5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12]
    ]
    n_new1 = NetworkModel(wbs, links=links_format1)
    print("Успешно создана модель с форматом 1")
    
    print("\n=== Новый формат 2 (две колонки) ===")
    links_format2 = [
        [1,5], [2,5], [3,5], [2,6], [3,6], [3,7], [4,7],
        [1,8], [6,8], [7,8], [5,9], [6,9], [7,9], [3,10],
        [6,10], [7,10], [6,11], [8,11], [9,11], [7,12],
        [8,12], [9,12], [10,12]
    ]
    n_new2 = NetworkModel(wbs, links=links_format2)
    print("Успешно создана модель с форматом 2")
    
    print("\n=== Новый формат 3 (словарь) ===")
    links_format3 = {
        'src': [1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10],
        'dst': [5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12]
    }
    n_new3 = NetworkModel(wbs, links=links_format3)
    print("Успешно создана модель с форматом 3")
    
    # Проверяем, что все модели идентичны
    print(f"\nПроверка идентичности моделей:")
    print(f"Старый == Новый1: {len(n_old.activities) == len(n_new1.activities)}")
    print(f"Старый == Новый2: {len(n_old.activities) == len(n_new2.activities)}")
    print(f"Старый == Новый3: {len(n_old.activities) == len(n_new3.activities)}")
