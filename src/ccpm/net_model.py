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
        assert isinstance(id,     int)
        assert isinstance(wbs_id, int)
        assert isinstance(model,  NetworkModel)
        assert isinstance(src,    _Event)
        assert isinstance(dst,    _Event)

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
    def __init__(self, wbs_dict, lnk_src, lnk_dst):
        
        assert isinstance(wbs_dict, dict)
        assert isinstance(lnk_src, np.ndarray)
        assert isinstance(lnk_dst, np.ndarray)

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
        
        self._cpm_compute('early')
        
        l = max([e.early for e in self.events])
        for e in self.events:
            e.late = l
        
        self._cpm_compute('late')
        
        for e in self.events:
            e.reserve = e.late - e.early
            
        for a in self.activities:
            a.reserve = a.late_start - a.early_start

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
        elif 'layer' == target:
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
        
        if 'layer' != target:
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
    def generate_viz(self):
        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style':'rounded'})
        dot.graph_attr['rankdir'] = 'LR'
        
        _mr = max([e.reserve for e in self.events])

        def _cl(res):
            if _mr <= 0:
                return '#ff0000'
            
            g = int(res / _mr * 255)
            r = 255 - g
            return '#' + hex(r)[2:] + hex(g)[2:] + '00'
        
        for e in self.events:
            dot.node(str(e.id), 
                     '{{%d |{%.1f|%.1f}| %.1f}}' % (e.id, 
                                                    e.early, 
                                                    e.late, 
                                                    e.reserve), 
                     color=_cl(e.reserve))
        
        _mr = max([a.reserve for a in self.activities])
        
        for a in self.activities:
            
            dot.edge(str(a.src.id), str(a.dst.id), 
                     label=(str(a.wbs_id) + '\n dur=' + str(a.duration) if a.wbs_id else '#') + '\n res=' + str(a.reserve), 
                     color=_cl(a.reserve),
                     style='dashed' if a.duration == 0 else 'solid'
                    )
        
        return dot

#==============================================================================
if __name__ == '__main__':
    
    wbs = {
        1 :{'duration':1., 'name': 'Heating and frames study'                                },
        2 :{'duration':2., 'name': 'Scouring and installation of building site establishment'},
        3 :{'duration':4., 'name': 'Earthwork and concrete well'                             },
        4 :{'duration':4., 'name': 'Earthwork and concrete longitudinal beams'               },
        5 :{'duration':6., 'name': 'Frame construction'                                      },
        6 :{'duration':2., 'name': 'Frame transport'                                         },
        7 :{'duration':5., 'name': 'Assemblage'                                              },
        8 :{'duration':2., 'name': 'Earthwork and pose drains'                               },
        9 :{'duration':5., 'name': 'Heating provisioning and assembly'                       },
        10:{'duration':5., 'name': 'Electric installation'                                   },
        11:{'duration':2., 'name': 'Painting'                                                },
        12:{'duration':1., 'name': 'Pavement'                                                }
        }
    
    src = np.array([1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10])
    dst = np.array([5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12])
    
    n = NetworkModel(wbs, src, dst)
    print(n)
