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
import pandas as pd
import scipy.stats as st
import os
import _ccpm

EPS = np.finfo(float).eps

RES = 0 # Result of time computation
VAR = 1 # Result variance estimation (used for PERT)
ERR = 2 # Computation error upper limit

#==============================================================================
def _p_quantile_estimate(tm, p):
    return tm[RES] + np.sqrt(tm[VAR]) * st.norm.ppf(p)

#==============================================================================
def _prob_estimate(tm, val):
    s = np.sqrt(tm[VAR])
    if s <= EPS * tm[RES]:
        return 1.0 if val > tm[RES] else 0.0
    else:
        return st.norm.cdf((val - tm[RES]) / s)

#==============================================================================
class _Activity:
    def __init__(self, id, wbs_id, letter, model, src, dst, duration=0.0, variance = 0.0, data=None):
        """
        Activity class representing a task in the network

        Parameters:
        -----------
        id : int
            Unique activity identifier
        wbs_id : int
            Work Breakdown Structure identifier
        letter : str
            Activity letter/code for visualization
        model : NetworkModel
            Parent network model
        src : _Event
            Source event
        dst : _Event
            Destination event
        duration : float
            Activity duration
        data : dict
            WBS data excluding fields stored as separate attributes
        """
        assert isinstance(id,       int)
        assert isinstance(wbs_id,   int)
        assert isinstance(letter,    str)
        assert isinstance(model,    NetworkModel)
        assert isinstance(src,      _Event)
        assert isinstance(dst,      _Event)
        assert isinstance(duration, float)
        assert duration >= 0.0
        assert variance >= 0.0
        assert data is None or isinstance(data, dict)

        self.id       = id
        self.wbs_id   = wbs_id
        self.letter   = letter
        self.model    = model
        self.src      = src
        self.dst      = dst
        #                            RES      VAR           ERR
        self.duration = np.array([duration, variance, EPS * duration], dtype=float)
        self.data     = data if data is not None else {}

        # CPM/PERT parameters
        self.early_start = np.zeros_like(self.duration)
        self.late_start  = np.zeros_like(self.duration)
        self.early_end   = np.zeros_like(self.duration)
        self.late_end    = np.zeros_like(self.duration)
        self.reserve     = np.zeros_like(self.duration)

    @property
    def early_start_pqe(self):
        return _p_quantile_estimate(self.early_start, self.model.p)

    def early_start_prob(self, val):
        return _prob_estimate(self.early_start, val)

    @property
    def early_end_pqe(self):
        return _p_quantile_estimate(self.early_end, self.model.p)

    def early_end_prob(self, val):
        return _prob_estimate(self.early_end, val)

    #----------------------------------------------------------------------------------------------
    def __repr__(self):
        return str(self.to_dict())

    #----------------------------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert activity to dictionary representation

        Returns:
        --------
        dict
            Dictionary with activity data
        """
        ret = {
            'id'         : self.id,
            'wbs_id'     : self.wbs_id,
            'letter'     : self.letter,
            'src_id'     : self.src.id,
            'dst_id'     : self.dst.id,
            'duration'   : self.duration[RES],
            'variance'   : self.duration[VAR],
            # CPM things
            'early_start': self.early_start[RES],
            'late_start' : self.late_start[RES],
            'early_end'  : self.early_end[RES],
            'late_end'   : self.late_end[RES],
            'reserve'    : self.reserve[RES],
            #Additional data copy
            'data'       : self.data.copy() # Return a copy to avoid modifying original
        }

        if self.model.is_pert:
            # PERT things
            ret['early_start_var'] = self.early_start[VAR]
            ret['early_end_var'  ] = self.early_end[VAR]
            ret['early_start_pqe'] = self.early_start_pqe
            ret['early_end_pqe'  ] = self.early_end_pqe

        if self.model.debug:
            # CPM computation errors
            ret['early_start_err'] = self.early_start[ERR]
            ret['late_start_err' ] = self.late_start[ERR]
            ret['early_end_err'  ] = self.early_end[ERR]
            ret['late_end_err'   ] = self.late_end[ERR]

        return ret

#==============================================================================
class _Event:
    def __init__(self, id, model):
        """
        Event class representing a milestone in the network

        Parameters:
        -----------
        id : int
            Unique event identifier
        model : NetworkModel
            Parent network model
        """
        assert isinstance(id, int)
        assert isinstance(model, NetworkModel)

        self.id = id
        self.model = model

        # CPM time parameters (calculated later)
        self.early   = np.zeros((3,), dtype=float)
        self.late    = np.zeros((3,), dtype=float)
        self.reserve = np.zeros((3,), dtype=float)
        self.stage   = 0

    @property
    def in_activities(self):
        """Get all activities entering this event"""
        return [a for a in self.model.activities if a.dst == self]

    @property
    def out_activities(self):
        """Get all activities leaving this event"""
        return [a for a in self.model.activities if a.src == self]

    @property
    def early_pqe(self):
        return _p_quantile_estimate(self.early, self.model.p)

    def early_prob(self, val):
        return _prob_estimate(self.early, val)

    #--------------------------------------------------------------------------
    def __repr__(self):
        return str(self.to_dict())

    #--------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert event to dictionary representation

        Returns:
        --------
        dict
            Dictionary with event data
        """
        # Basic action (CPM)
        ret = {
            'id'     : self.id,
            'stage'  : self.stage,
            'early'  : self.early[RES],
            'late'   : self.late[RES],
            'reserve': self.reserve[RES],
        }

        if self.model.is_pert:
            # PERT things
            ret['early_var'  ] = self.early[VAR]
            #ret['late_var'   ] = self.late[VAR]
            #ret['reserve_var'] = self.reserve[VAR]
            ret['early_pqe'] = self.early_pqe


        if self.model.debug:
            # CPM computation errors
            ret['early_err'] = self.early[ERR]
            ret['late_err' ] = self.late[ERR]

        return ret

#==============================================================================
class NetworkModel:
    def __init__(self, wbs_dict, lnk_src=None, lnk_dst=None, links=None, p=0.95, debug=False):
        """
        Initialize NetworkModel with multiple link formats support

        Parameters:
        -----------
        wbs_dict : dict
            Work Breakdown Structure dictionary with activity data including:
            - 'duration': activity duration (required)
            - 'letter': activity letter/code (required)
            - 'name': activity description (optional)
            - any other custom fields
        lnk_src, lnk_dst : array-like, optional
            Old format: separate source and destination arrays
        links : various formats, optional
            New formats:
            - Format 1: two rows [[src1, src2, ...], [dst1, dst2, ...]]
            - Format 2: two columns [[src1, dst1], [src2, dst2], ...]
            - Format 3: dictionary {'src': [src1, src2, ...], 'dst': [dst1, dst2, ...]}
        """
        assert isinstance(wbs_dict, dict)

        self.debug = debug
        self.is_pert = False
        self.p = p

        # Parse links into standard format
        lnk_src, lnk_dst = self._parse_links(lnk_src, lnk_dst, links)

        # Create network model
        self._create_model(wbs_dict, lnk_src, lnk_dst)

        assert 0 < len(self.events)

        # Compute stages of project
        self._compute_target('stage')

        # Compute Event and Activity time parameters
        self._compute_time_params()

    #--------------------------------------------------------------------------
    def _parse_links(self, lnk_src, lnk_dst, links):
        """
        Parse links from various formats into standard lnk_src, lnk_dst arrays

        Returns:
        --------
        lnk_src, lnk_dst : numpy.ndarray
            Standardized source and destination arrays
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
    def _create_model(self, wbs_dict, lnk_src, lnk_dst):
        assert len(lnk_src) == len(lnk_dst)

        act_ids = np.array(list(wbs_dict.keys()), dtype=int)

        # Generate network graph using C++ extension
        status, net_src, net_dst, lnk_src, lnk_dst = _ccpm.compute_aoa(act_ids, lnk_src, lnk_dst)
        assert 0 == status

        self.events     = []
        self.next_act   = 1
        self.activities = []

        # Create events
        for i in range(np.max(net_dst)):
            self._add_event(int(i + 1))

        # Create activities (real and dummy)
        na = len(act_ids)    # Number of actions
        nd = 0               # Number of dunny actions
        for i in range(len(net_src)):
            if i < na:
                # Real activity - get data from WBS
                act_id = act_ids[i]
                wbs_data = wbs_dict[act_id]  # Get complete WBS data
                duration = wbs_data.get('duration', 0.)
                variance = wbs_data.get('variance', 0.)
                letter = wbs_data.get('letter', '')

                # Even one wbs item with ninzero variance is enough to compute PERT`
                if variance > 0.0:
                    self.is_pert = True

                # Create data dict without fields stored as separate attributes
                data_without_duplicates = self._remove_duplicate_fields(wbs_data, duration, letter)

                self._add_activity(int(act_id), int(net_src[i]), int(net_dst[i]),
                                  duration, variance, letter, data_without_duplicates)
            else:
                # Add a dummy activity (no duration, no letter, no data)
                nd += 1 #One more dummy work
                self._add_activity(0, int(net_src[i]), int(net_dst[i]), 0., 0., '#' + str(nd), {})

        #TODO: Высиавлять на длинную сторону треугольников работы с максимальной длительностью

    #--------------------------------------------------------------------------
    def _remove_duplicate_fields(self, wbs_data, duration, letter):
        """
        Remove fields from WBS data that are stored as separate activity attributes

        Parameters:
        -----------
        wbs_data : dict
            Complete WBS data for an activity
        duration : float
            Activity duration (already extracted)
        letter : str
            Activity letter (already extracted)

        Returns:
        --------
        dict
            WBS data without fields stored as separate attributes
        """
        # Create a copy to avoid modifying the original data
        data_copy = wbs_data.copy()

        # Remove fields that are stored as separate attributes
        fields_to_remove = ['duration', 'variance', 'letter']
        for field in fields_to_remove:
            if field in data_copy:
                del data_copy[field]

        return data_copy

    def _compute_time_params(self):
        self._compute_target('early')

        # Set late times starting from project completion
        late = np.zeros((3,), dtype=float)
        for e in self.events:
            if e.early[RES] > late[RES]:
                late = e.early.copy()

        late[VAR] = 0.0 #Start back computation with zero variance

        for e in self.events:
            e.late = late

        self._compute_target('late')

        # Compute reserves
        for e in self.events:
            e.reserve[VAR] = e.late[VAR] + e.early[VAR]
            e.reserve[ERR] = e.late[ERR] + e.early[ERR]
            # Round off insignificant values
            r = e.late[RES] - e.early[RES]
            e.reserve[RES] = r if abs(r) > e.reserve[ERR] else 0.0
            # Check for programming errors
            assert r > -e.reserve[ERR]

        for a in self.activities:
            a.reserve[VAR] = a.late_start[VAR] + a.early_start[VAR]
            a.reserve[ERR] = a.late_start[ERR] + a.early_start[ERR]
            # Round off insignificant values
            r = a.late_start[RES] - a.early_start[RES]
            a.reserve[RES] = r if abs(r) > a.reserve[ERR] else 0.0
            # Check for programming errors
            assert r > -a.reserve[ERR]

    #--------------------------------------------------------------------------
    def _compute_target(self, target=None):
        """
        Compute CPM parameters for events and activities

        Parameters:
        -----------
        target : str
            What to compute: 'stage', 'early', or 'late'
        """

        def _choise(old, new, delta):
            e = new[ERR] + old[ERR]
            if delta >= e:
                return new #Certain result
            elif delta >= -e:
                #Uncertain result, use mixing
                ret = np.zeros((3,), dtype=float)
                ret[RES] = 0.5 * (new[RES] + old[RES])
                ret[VAR] = max(old[VAR], new[VAR])
                ret[ERR] = 0.5 * e
                return ret
            else:
                return old #Certain result

        def _choise_early(old, new):
            return _choise(old, new, new[RES] - old[RES])

        def _choise_late(old, new):
            return _choise(old, new, old[RES] - new[RES])

        def _delta_late(a):
            ret = a.duration.copy()
            ret[RES] = -a.duration[RES]
            return ret

        if 'stage' == target:
            act_base     = None
            act_new      = None
            act_next     = 'dst'
            fwd          = 'out_activities'
            rev          = 'in_activities'
            choise       = max
            delta        = lambda a : 1

        elif 'early' == target:
            act_base     = 'early_start'
            act_new      = 'early_end'
            act_next     = 'dst'
            fwd          = 'out_activities'
            rev          = 'in_activities'
            choise       = _choise_early
            delta        = lambda a : a.duration

        elif 'late' == target:
            act_base     = 'late_end'
            act_new      = 'late_start'
            act_next     = 'src'
            fwd          = 'in_activities'
            rev          = 'out_activities'
            choise       = _choise_late
            delta        = _delta_late

        else:
            raise ValueError("Unknown 'target' value!!!")

        if 'stage' != target:
            for a in self.activities:
                setattr(a, act_base, -1)
                setattr(a, act_new,  -1)

        # Count dependencies for topological sorting
        n_dep = [len(getattr(e, rev)) for e in self.events]

        # Find starting events (no dependencies)
        evt = [i for i,n in enumerate(n_dep) if 0 == n]
        assert 1 == len(evt)

        # Process events in topological order
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
    def _add_event(self, i):
        """Add a new event to the network"""
        self.events.append(_Event(i, self))

    #--------------------------------------------------------------------------
    def _add_activity(self, wbs_id, src_id, dst_id, duration, variance, letter, data):
        """
        Add a new activity to the network

        Parameters:
        -----------
        wbs_id : int
            Work Breakdown Structure identifier (0 for dummy activities)
        src_id : int
            Source event ID
        dst_id : int
            Destination event ID
        duration : float
            Activity duration
        variance : float
            Activity duration variance
        letter : str
            Activity letter/code for visualization
        data : dict
            WBS data excluding fields stored as separate attributes
        """
        assert isinstance(wbs_id,   int)
        assert isinstance(src_id,   int)
        assert isinstance(dst_id,   int)
        assert isinstance(duration, float)
        assert isinstance(letter,    str)
        assert isinstance(data,     dict)

        act = _Activity(self.next_act, wbs_id, letter, self,
                        self.events[src_id-1], self.events[dst_id-1],
                        duration, variance, data)
        self.activities.append(act)
        self.next_act += 1

    #--------------------------------------------------------------------------
    def __repr__(self):
        """String representation of the network model"""
        _repr = 'Events:{\n'
        for e in self.events:
            _repr += '        ' + str(e) + '\n'
        _repr += '}\n'

        _repr += 'Activities:\n'
        for a in self.activities:
            _repr += '        ' + str(a) + '\n'
        _repr += '}\n'

        return _repr

    #--------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert network model to dictionary representation

        Returns:
        --------
        dict
            Dictionary with structure:
            {
                'activities': [list of activity dictionaries],
                'events': [list of event dictionaries]
            }
        """
        activities_data = [activity.to_dict() for activity in self.activities]
        events_data = [event.to_dict() for event in self.events]

        return {
            'activities': activities_data,
            'events': events_data
        }

    #--------------------------------------------------------------------------
    def to_dataframe(self):
        """
        Convert network model to pandas DataFrames

        Returns:
        --------
        tuple
            (activities_df, events_df) - pandas DataFrames for activities and events
        """

        # Convert to dictionaries first
        model_dict = self.to_dict()

        # Create events DataFrame (straightforward)
        events_df = pd.DataFrame(model_dict['events'])

        # Create activities DataFrame with data fields expanded
        activities_list = model_dict['activities']

        # Expand data fields into separate columns
        expanded_activities = []
        for activity in activities_list:
            # Start with basic activity data
            activity_data = {k: v for k, v in activity.items() if k != 'data'}

            # Add data fields as separate columns
            if 'data' in activity and activity['data']:
                activity_data.update(activity['data'])

            expanded_activities.append(activity_data)

        activities_df = pd.DataFrame(expanded_activities).fillna(value='')

        return activities_df, events_df

    #--------------------------------------------------------------------------
    def viz(self, output_path=None):
        """
        Create Graphviz visualization of the CPM network

        Parameters:
        -----------
        output_path : str, optional
            Custom output path for saving the visualization file.
            If None, uses default location.

        Returns:
        --------
        graphviz.Digraph
            Graphviz object for rendering or saving
        """
        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style':'rounded'})
        dot.graph_attr['rankdir'] = 'LR'

        def _cl(res, p):
            """Choose color based on reserve (red for critical path)"""
            if abs(res[RES]) < res[ERR]:  # Absolute precision is nonsense
                return '#ff0000'
            elif p < self.p:
                return '#ffa000'
            else:
                return '#000000'

        # Add events/nodes
        for e in self.events:
            # Format time values to 1 decimal place
            dot.node(str(e.id),
                     '{{%d |{%.1f|%.1f}| %.1f}}' % (e.id,
                                                    e.early[RES],
                                                    e.late[RES],
                                                    e.reserve[RES]),
                     color=_cl(e.reserve, e.early_prob(e.late[RES])))


        # Add activities/edges
        for a in self.activities:
            if a.wbs_id:  # Real activity
                # Use letter instead of wbs_id in visualization
                # Format duration and reserve to 1 decimal place
                lbl  = a.letter
                lbl += '\n t=' + format(a.duration[RES], '.1f') + '\n r=' + format(a.reserve[RES], '.1f')
            else:  # Dummy activity
                lbl = '# \n r=' + format(a.reserve[RES], '.1f')

            dot.edge(str(a.src.id), str(a.dst.id),
                     label=lbl,
                     color=_cl(a.reserve, a.early_end_prob(a.late_end[RES])),
                     style='dashed' if a.duration[RES] == 0.0 else 'solid'
                    )

        # If output path is specified, render to that location
        if output_path is not None:
            dot.render(output_path, format='png', cleanup=True)

        return dot

    #--------------------------------------------------------------------------
    def get_activity_by_wbs_id(self, wbs_id):
        """
        Get activity by WBS ID

        Parameters:
        -----------
        wbs_id : int
            WBS identifier

        Returns:
        --------
        _Activity or None
            Activity with specified WBS ID or None if not found
        """
        for activity in self.activities:
            if activity.wbs_id == wbs_id:
                return activity
        return None

    #--------------------------------------------------------------------------
    def get_activities_by_data_field(self, field_name, field_value):
        """
        Get activities by custom data field value

        Parameters:
        -----------
        field_name : str
            Name of the field in activity.data
        field_value : any
            Value to match

        Returns:
        --------
        list
            List of activities with matching field value
        """
        result = []
        for activity in self.activities:
            if activity.data.get(field_name) == field_value:
                result.append(activity)
        return result

#==============================================================================
if __name__ == '__main__':
    # Example usage with all link formats
    wbs = {
        1 :{'letter':'A', 'duration':3.84, 'variance':0.01, 'name':'Heating and frames study'                                },
        2 :{'letter':'B', 'duration':2., 'variance':0.01, 'name':'Scouring and installation of building site establishment'},
        3 :{'letter':'C', 'duration':3.8, 'variance':0.01, 'name':'Earthwork and concrete well'                             },
        4 :{'letter':'D', 'duration':4., 'variance':0.01, 'name':'Earthwork and concrete longitudinal beams'               },
        5 :{'letter':'E', 'duration':6., 'variance':0.01, 'name':'Frame construction'                                      },
        6 :{'letter':'F', 'duration':6., 'variance':0.01, 'name':'Frame transport'                                         },
        7 :{'letter':'G', 'duration':6., 'variance':0.01, 'name':'Assemblage'                                              },
        8 :{'letter':'H', 'duration':2., 'variance':0.01, 'name':'Earthwork and pose drains'                               },
        9 :{'letter':'I', 'duration':5., 'variance':0.01, 'name':'Heating provisioning and assembly'                       },
        10:{'letter':'J', 'duration':5., 'variance':0.01, 'name':'Electric installation'                                   },
        11:{'letter':'K', 'duration':2., 'variance':0.01, 'name':'Painting'                                                },
        12:{'letter':'L', 'duration':1., 'variance':0.01, 'name':'Pavement'                                                }
        }

    print("=== Demonstration of all link formats ===")

    # Old format
    print("\n1. Old format:")
    src_old = np.array([1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10])
    dst_old = np.array([5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12])
    n_old = NetworkModel(wbs, src_old, dst_old)
    print("Successfully created model with old format")

    print("Full dependency map for old format:")
    act_id = np.array(list(wbs.keys()))
    status, full_dep_map = _ccpm.make_full_map(act_id, src_old, dst_old)
    print(status)
    print(act_id)
    print(full_dep_map)

    # Demonstration of the new data attribute
    print("\n=== Demonstration of the data attribute ===")
    for i, activity in enumerate(n_old.activities[:5]):  # Show first 5 activities
        if activity.wbs_id != 0:  # Skip dummy activities
            print(f"Activity {i+1}: wbs_id={activity.wbs_id}, letter='{activity.letter}'")
            print(f"  Data: {activity.data}")


    # New format 1 (two rows)
    print("\n2. New format 1 (two rows):")
    links_format1 = [
        [1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10],
        [5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12]
    ]
    n_new1 = NetworkModel(wbs, links=links_format1)
    print("Successfully created model with format 1")

    # New format 2 (two columns)
    print("\n3. New format 2 (two columns):")
    links_format2 = [
        [1,5], [2,5], [3,5], [2,6], [3,6], [3,7], [4,7],
        [1,8], [6,8], [7,8], [5,9], [6,9], [7,9], [3,10],
        [6,10], [7,10], [6,11], [8,11], [9,11], [7,12],
        [8,12], [9,12], [10,12]
    ]
    n_new2 = NetworkModel(wbs, links=links_format2)
    print("Successfully created model with format 2")

    # New format 3 (dictionary)
    print("\n4. New format 3 (dictionary):")
    links_format3 = {
        'src': [1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10],
        'dst': [5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12]
    }
    n_new3 = NetworkModel(wbs, links=links_format3)
    print("Successfully created model with format 3")

    # Check model equivalence
    print("\n=== Checking model equivalence ===")
    print(f"Old == New1: {len(n_old.activities) == len(n_new1.activities)}")
    print(f"Old == New2: {len(n_old.activities) == len(n_new2.activities)}")
    print(f"Old == New3: {len(n_old.activities) == len(n_new3.activities)}")

    # Demonstration of the letter attribute
    print("\n=== Demonstration of the letter attribute ===")
    for i, activity in enumerate(n_old.activities[:5]):  # Show first 5 activities
        if activity.wbs_id != 0:  # Skip dummy activities
            print(f"Activity {i+1}: wbs_id={activity.wbs_id}, letter='{activity.letter}', duration={activity.duration}")

    print("\n=== Demonstration of dictionary export ===")
    model_dict = n_old.to_dict()

    print(f"Number of activities: {len(model_dict['activities'])}")
    print(f"Number of events: {len(model_dict['events'])}")

    # Show data structure
    print("\nActivity data structure:")
    if len(model_dict['activities']) > 0:
        first_activity = model_dict['activities'][0]
        print(f"Keys: {list(first_activity.keys())}")
        print(f"Example activity: {first_activity}")

    print("\nEvent data structure:")
    if len(model_dict['events']) > 0:
        first_event = model_dict['events'][0]
        print(f"Keys: {list(first_event.keys())}")
        print(f"Example event: {first_event}")

    # Demonstration of DataFrame export (if pandas is available)
    print("\n=== Demonstration of DataFrame export ===")
    try:
        n_old.debug = True
        activities_df, events_df = n_old.to_dataframe()
        n_old.debug = False

        print("\nActivities DataFrame:")
        print(f"Size: {activities_df.shape}")
        print(f"Columns: {list(activities_df.columns)}")
        print("\nFirst 5 rows:")
        print(activities_df.head())

        print("\nEvents DataFrame:")
        print(f"Size: {events_df.shape}")
        print(f"Columns: {list(events_df.columns)}")
        print("\nFirst 5 rows:")
        print(events_df.head())

        # Show that data from 'data' field is now in separate columns
        print("\nChecking data expansion:")
        if 'name' in activities_df.columns:
            print("Data from 'data' successfully expanded into separate columns:")
            print(activities_df[['letter', 'name', 'duration', 'reserve']].head())

    except Exception as e:
        print(f"Error exporting to DataFrame: {e}")

    # Create visualization in specific directory
    print("\n=== Creating visualization ===")

    # Get the directory of the current module
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the target directory path: <module directory>/../../tests/data
    target_dir = os.path.normpath(os.path.join(module_dir, '../../tests/data'))

    # Create the directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Construct the full file path
    file_path = os.path.join(target_dir, 'cpm_network')

    # Create and save the visualization
    dot = n_old.viz(output_path=file_path)
    print(f"Visualization saved as '{file_path}.png'")
    print(f"Target directory: {target_dir}")

    print("Full dependency map for old format:")
    act_id = np.array(list(wbs.keys()))
    status, full_dep_map = _ccpm.make_full_map(act_id, src_old, dst_old)
    print(status)
    print(act_id)
    print(full_dep_map)

    src_old[0] = 12
    act_id = np.array(list(wbs.keys()))
    status, full_dep_map = _ccpm.make_full_map(act_id, src_old, dst_old)
    print(status)
    print(act_id)
    print(full_dep_map)
