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
    def __init__(self, id, wbs_id, leter, model, src, dst, duration=0.0, data=None):
        assert isinstance(id,       int)
        assert isinstance(wbs_id,   int)
        assert isinstance(leter,    str)
        assert isinstance(model,    NetworkModel)
        assert isinstance(src,      _Event)
        assert isinstance(dst,      _Event)
        assert isinstance(data,     dict)  # data должен быть словарем
        
        # Проверяем тип duration и преобразуем к нужному формату
        if isinstance(duration, (list, tuple, np.ndarray)):
            assert len(duration) == 3
            self.duration = np.array(duration, dtype=float)
        else:
            assert isinstance(duration, float)
            assert duration >= 0.
            self.duration = np.array([duration, duration, duration], dtype=float)

        self.id       = id
        self.wbs_id   = wbs_id
        self.leter    = leter
        self.model    = model
        self.src      = src
        self.dst      = dst
        self.data     = data  # Сохраняем все данные из wbs записи

        # Временные параметры как массивы (3,)
        self.early_start = np.zeros(3)
        self.late_start  = np.zeros(3)
        self.early_end   = np.zeros(3)
        self.late_end    = np.zeros(3)
        self.reserve     = np.zeros(3)

    #----------------------------------------------------------------------------------------------
    def __repr__(self):
        if self.model.is_pert:
            # Для PERT модели выводим все три значения
            return 'Activity(id=%r, src_id=%r, dst_id=%r, duration=%r, reserve=%r, wbs_id=%r, leter=%r)' % (
                self.id,
                self.src.id,
                self.dst.id,
                self.duration.tolist(),
                self.reserve.tolist(),
                self.wbs_id,
                self.leter
            )
        else:
            # Для CPM модели выводим только сбалансированный сценарий (индекс 1)
            return 'Activity(id=%r, src_id=%r, dst_id=%r, duration=%r, reserve=%r, wbs_id=%r, leter=%r)' % (
                self.id,
                self.src.id,
                self.dst.id,
                self.duration[1],
                self.reserve[1],
                self.wbs_id,
                self.leter
            )
    
#==============================================================================
class _Event:    
    def __init__(self, id, model):
        assert isinstance(id,     int)
        assert isinstance(model, NetworkModel)

        self.id = id
        self.model   = model
        
        # Временные параметры как массивы (3,)
        self.early   = np.zeros(3)
        self.late    = np.zeros(3)
        self.reserve = np.zeros(3)
        self.stage   = 0  # Стадия остается скаляром
        
    @property
    def in_activities(self):
        return [a for a in self.model.activities if a.dst == self]

    @property
    def out_activities(self):
        return [a for a in self.model.activities if a.src == self]

    #--------------------------------------------------------------------------
    def __repr__(self):
        if self.model.is_pert:
            # Для PERT модели выводим все три значения
            return 'Event(id=%r early=%r late=%r reserve=%r stage=%r)' % (
                self.id,
                self.early.tolist(),
                self.late.tolist(),
                self.reserve.tolist(),
                self.stage
            )
        else:
            # Для CPM модели выводим только сбалансированный сценарий (индекс 1)
            return 'Event(id=%r early=%r late=%r reserve=%r stage=%r)' % (
                self.id,
                self.early[1],
                self.late[1],
                self.reserve[1],
                self.stage
            )

#==============================================================================
class NetworkModel:
    def __init__(self, wbs_dict, lnk_src=None, lnk_dst=None, links=None):
        """
        Конструктор сетевой модели
        
        Поддерживает несколько форматов задания связей:
        
        1. Старый формат (для обратной совместимости):
           lnk_src = np.array([1, 2, 3, ...])
           lnk_dst = np.array([5, 5, 5, ...])
        
        2. Новый формат 1 - две строки:
           links = np.array([[1, 2, 3, ...],  # src
                            [5, 5, 5, ...]]) # dst
        
        3. Новый формат 2 - две колонки:
           links = np.array([[1, 5],
                            [2, 5], 
                            [3, 5],
                            ...])
        
        4. Новый формат 3 - словарь:
           links = {'src': [1, 2, 3, ...],
                   'dst': [5, 5, 5, ...]}
        
        Parameters:
            wbs_dict: словарь с описанием работ
            lnk_src: массив начал связей (старый формат)
            lnk_dst: массив концов связей (старый формат) 
            links: связи в одном из новых форматов
        """
        assert isinstance(wbs_dict, dict)
        
        # Обрабатываем формат связей
        lnk_src, lnk_dst = self._parse_links(lnk_src, lnk_dst, links)
        assert len(lnk_src) == len(lnk_dst)

        self.events     = []
        self.next_act   = 1
        self.activities = []
        self.is_pert    = False

        # Проверяем, есть ли PERT-оценки в wbs_dict
        for task_data in wbs_dict.values():
            duration = task_data.get('duration', 0.)
            if isinstance(duration, (list, tuple, np.ndarray)):
                self.is_pert = True
                break

        # Преобразуем все duration к массивам (3,)
        processed_wbs = {}
        for task_id, task_data in wbs_dict.items():
            data = task_data.copy()
            duration = data.get('duration', 0.)
            
            if isinstance(duration, (list, tuple, np.ndarray)):
                # PERT-оценка - сортируем: оптимистическая, сбалансированная, пессимистическая
                sorted_duration = sorted(duration)
                data['duration'] = np.array(sorted_duration, dtype=float)
            else:
                # CPM-оценка - дублируем для трех сценариев
                data['duration'] = np.array([duration, duration, duration], dtype=float)
            
            processed_wbs[task_id] = data

        #Generate network graph
        act_ids = np.array(list(processed_wbs.keys()), dtype=int)

        status, net_src, net_dst, lnk_src, lnk_dst = _ccpm.compute_aoa(act_ids, lnk_src, lnk_dst)
        assert 0 == status

        for i in range(np.max(net_dst)):
            self._add_event(int(i + 1))

        na = len(act_ids)
        d  = np.max(act_ids)
        for i in range(len(net_src)):
            if i < na:
                # Получаем буквенное обозначение из processed_wbs
                leter = processed_wbs[act_ids[i]].get('leter', '')
                duration = processed_wbs[act_ids[i]].get('duration', 0.)
                data = processed_wbs[act_ids[i]]  # Получаем все данные записи
                self._add_activity(int(act_ids[i]), leter, int(net_src[i]), int(net_dst[i]), duration, data)
            else:
                #Add a dummy - фиктивная работа
                d += 1
                self._add_activity(0, '', int(net_src[i]), int(net_dst[i]), np.zeros(3), {})

        #Compute Event and Actions attributes
        assert 0 < len(self.events)

        self._cpm_compute('stage')

        self._cpm_compute('early')

        # Для поздних времен находим максимальное раннее время для каждого сценария
        early_times = np.array([e.early for e in self.events])
        l = np.max(early_times, axis=0)
        for e in self.events:
            e.late = l

        self._cpm_compute('late')

        for e in self.events:
            e.reserve = e.late - e.early

        for a in self.activities:
            a.early_end = a.early_start + a.duration
            a.late_end  = a.late_start  + a.duration
            a.reserve   = a.late_start  - a.early_start

        # Заменяем отрицательные значения на 0 и округляем до 1 знака
        self._replace_negative_with_zero()

    #--------------------------------------------------------------------------
    def _parse_links(self, lnk_src, lnk_dst, links):
        """
        Парсит связи в различных форматах и возвращает два массива (src, dst)
        
        Поддерживаемые форматы:
        1. Старый формат: lnk_src и lnk_dst как отдельные массивы
        2. Две строки: массив shape=(2, N)
        3. Две колонки: массив shape=(N, 2)  
        4. Словарь: {'src': [...], 'dst': [...]}
        """
        # Если передан параметр links, используем новые форматы
        if links is not None:
            if isinstance(links, np.ndarray):
                if links.ndim == 2:
                    if links.shape[0] == 2:
                        # Формат 1: две строки
                        lnk_src = links[0, :].copy()
                        lnk_dst = links[1, :].copy()
                    elif links.shape[1] == 2:
                        # Формат 2: две колонки
                        lnk_src = links[:, 0].copy()
                        lnk_dst = links[:, 1].copy()
                    else:
                        raise ValueError("Массив links должен иметь shape=(2, N) или shape=(N, 2)")
                else:
                    raise ValueError("Массив links должен быть двумерным")
                    
            elif isinstance(links, dict):
                # Формат 3: словарь с ключами 'src' и 'dst'
                if 'src' in links and 'dst' in links:
                    lnk_src = np.array(links['src'], dtype=int)
                    lnk_dst = np.array(links['dst'], dtype=int)
                else:
                    raise ValueError("Словарь links должен содержать ключи 'src' и 'dst'")
            else:
                raise ValueError("Неподдерживаемый тип для links. Должен быть np.ndarray или dict")
                
        # Проверяем, что получили валидные массивы
        if lnk_src is None or lnk_dst is None:
            raise ValueError("Не заданы связи между работами")
            
        if not isinstance(lnk_src, np.ndarray) or not isinstance(lnk_dst, np.ndarray):
            raise ValueError("lnk_src и lnk_dst должны быть numpy массивами")
            
        return lnk_src, lnk_dst

    #--------------------------------------------------------------------------
    def _replace_negative_with_zero(self):
        """Заменяет все отрицательные значения временных параметров на 0"""
        for e in self.events:
            e.early = np.maximum(e.early, 0)
            e.late = np.maximum(e.late, 0)
            e.reserve = np.maximum(e.reserve, 0)

        for a in self.activities:
            a.early_start = np.maximum(a.early_start, 0)
            a.late_start = np.maximum(a.late_start, 0)
            a.early_end = np.maximum(a.early_end, 0)
            a.late_end = np.maximum(a.late_end, 0)
            a.reserve = np.maximum(a.reserve, 0)

    #--------------------------------------------------------------------------
    def _add_event(self, i):
        self.events.append(_Event(i, self))
    
    #--------------------------------------------------------------------------
    def _add_activity(self, wbs_id, leter, src_id, dst_id, duration, data):
        assert isinstance(wbs_id, int)
        assert isinstance(leter,  str)
        assert isinstance(dst_id, int)
        assert isinstance(wbs_id, int)
        assert isinstance(data,   dict)

        act = _Activity(self.next_act, wbs_id, leter, self, self.events[src_id-1], 
                        self.events[dst_id-1], duration, data)
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
            choise       = lambda x, y: np.maximum(x, y)
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
            choise       = lambda x, y: np.minimum(x, y)
        else:
            raise ValueError("Unknown 'target' value!!!")
        
        if 'stage' != target:
            for a in self.activities:
                setattr(a, act_base, np.full(3, -1.0))
                if act_new:
                    setattr(a, act_new, np.full(3, -1.0))
           
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
    def _viz_cpm(self):
        """Визуализация для CPM модели"""
        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style':'rounded'})
        dot.graph_attr['rankdir'] = 'LR'

        def _cl(res):
            if res <= 1e-6: # Абсолютная точность бессмысленна
                return '#ff0000'
            return '#000000'

        for e in self.events:
            dot.node(str(e.id), 
                     '{{%d |{%.1f|%.1f}| %.1f}}' % (
                         e.id, e.early[1], #Сбалансированнный сценарий (индекс 1)
                         e.late[1],        #Сбалансированнный сценарий (индекс 1)
                         e.reserve[1]      #Сбалансированнный сценарий (индекс 1)
                         ), 
                     color=_cl(e.reserve[1]))

        for a in self.activities:

            if a.wbs_id:
                lbl  = str(a.leter)
                lbl += '\n t=' + str(a.duration[1]) + '\n r=' + str(a.reserve[1])
            else:
                lbl = '# \n r=' + str(a.reserve[1])

            dot.edge(str(a.src.id), str(a.dst.id), 
                     label=lbl, 
                     color=_cl(a.reserve[1]),
                     style='dashed' if np.all(a.duration == 0) else 'solid'
                    )

        return dot

    #--------------------------------------------------------------------------
    def _viz_pert(self):
        """Визуализация для PERT модели"""
        dot = graphviz.Digraph(node_attr={'shape': 'record', 'style':'rounded'})
        dot.graph_attr['rankdir'] = 'LR'

        def _cl(res):
            if res <= 1e-6: # Абсолютная точность бессмысленна
                return '#ff0000'
            return '#000000'

        # Создаем узлы событий в виде таблиц
        for e in self.events:

            label = '{{%d|{%.1f|%.1f|%.1f}|{%.1f|%.1f|%.1f}|{%.1f|%.1f|%.1f}}}' % (
                e.id,
                # Пессимистический сценарий (индекс 2)
                e.early[2], e.reserve[2], e.late[2],
                # Сбалансированный сценарий (индекс 1)
                e.early[1], e.reserve[1], e.late[1],
                # Оптимистический сценарий (индекс 0)
                e.early[0], e.reserve[0], e.late[0]
            )

            dot.node(str(e.id), label, color=_cl(e.reserve[1]))

        # Создаем ребра работ
        for a in self.activities:
           
            if a.wbs_id:
                # Обычная работа - показываем три оценки через точку с запятой
                duration_str = '%.1f; %.1f; %.1f' % (
                    a.duration[0],
                    a.duration[1],
                    a.duration[2]
                    )
                reserve_str = '%.1f; %.1f; %.1f' % (
                    a.reserve[0],
                    a.reserve[1],
                    a.reserve[2]
                    )
                lbl = '%s\n t: %s\n r: %s' % (a.leter, duration_str, reserve_str)
            else:
                # Фиктивная работа
                reserve_str = '%.1f; %.1f; %.1f' % (
                    a.reserve[0],
                    a.reserve[1],
                    a.reserve[2]
                    )
                lbl = '#\n r: %s' % reserve_str

            dot.edge(str(a.src.id), str(a.dst.id), 
                     label=lbl, 
                     color=_cl(a.reserve[1]),
                     style='dashed' if np.all(a.duration == 0) else 'solid'
                    )

        return dot

    #--------------------------------------------------------------------------
    def viz(self):
        """Основной метод визуализации - выбирает между CPM и PERT"""
        if self.is_pert:
            return self._viz_pert()
        else:
            return self._viz_cpm()

#==============================================================================
if __name__ == '__main__':
    
    # Пример CPM модели
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
    
    # Старый формат (сохраняется для обратной совместимости)
    src_old = np.array([1,2,3, 2,3, 3,4, 1,6,7, 5,6,7, 3, 6, 7,  6, 8, 9,  7, 8, 9, 10])
    dst_old = np.array([5,5,5, 6,6, 7,7, 8,8,8, 9,9,9, 10,10,10, 11,11,11, 12,12,12,12])
    
    print("=== CPM Model (старый формат) ===")
    n_cpm_old = NetworkModel(wbs, src_old, dst_old)
    print(n_cpm_old)
    
    # Новый формат 1: две строки
    links_two_rows = np.array([
        [1, 2, 3, 2, 3, 3, 4, 1, 6, 7, 5, 6, 7, 3, 6, 7, 6, 8, 9, 7, 8, 9, 10],
        [5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12]
    ])
    
    print("=== CPM Model (формат 1: две строки) ===")
    n_cpm_new1 = NetworkModel(wbs, links=links_two_rows)
    print(n_cpm_new1)
    
    # Новый формат 2: две колонки
    links_two_cols = np.array([
        [1, 5], [2, 5], [3, 5], [2, 6], [3, 6], [3, 7], [4, 7], [1, 8], [6, 8], [7, 8],
        [5, 9], [6, 9], [7, 9], [3, 10], [6, 10], [7, 10], [6, 11], [8, 11], [9, 11],
        [7, 12], [8, 12], [9, 12], [10, 12]
    ])
    
    print("=== CPM Model (формат 2: две колонки) ===")
    n_cpm_new2 = NetworkModel(wbs, links=links_two_cols)
    print(n_cpm_new2)
    
    # Новый формат 3: словарь
    links_dict = {
        'src': [1, 2, 3, 2, 3, 3, 4, 1, 6, 7, 5, 6, 7, 3, 6, 7, 6, 8, 9, 7, 8, 9, 10],
        'dst': [5, 5, 5, 6, 6, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12]
    }
    
    print("=== CPM Model (формат 3: словарь) ===")
    n_cpm_new3 = NetworkModel(wbs, links=links_dict)
    print(n_cpm_new3)
    
    # Проверяем доступ к данным активности
    for activity in n_cpm_new3.activities:
        if activity.wbs_id > 0:  # Пропускаем фиктивные работы
            print(f"Activity {activity.leter}: name = {activity.data.get('name', 'N/A')}")
    
    # Пример PERT модели
    for _,w in wbs.items():
        w['duration'] *= np.array([np.random.uniform(0.7, 0.8), 1.0, np.random.uniform(1.1, 1.2)])
    
    print("\n=== PERT Model (формат 1: две строки) ===")
    n_pert = NetworkModel(wbs, links=links_two_rows)
    print(n_pert)
