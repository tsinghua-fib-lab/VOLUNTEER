import time
import math
import random
from tkinter import W
import numpy as np
from collections import defaultdict
from math import sqrt, sin, cos, pi, asin


# global parameter setting
train_ratio = 0.8
active_user_stay = 10
active_home_stay = 5
active_work_stay = 5
time_slot = 10  # 10 minutes
spatial_threshold = 0.005  # approximate to 500m
home_start = 19
home_end = 9
home_threshold = 0.3
work_start = 6
work_end = 23
work_spatial_threshold = 0.005
work_frequency_threshold = 0.6

max_stay_duration = 12
max_explore_range = 100
spatial_resolution = 0.005
eta = 0.035
rho = 0.6
gamma = 0.21

performance_spatial_threshold = 0.005


# Time format function
def date2stamp(time_date):
    time_array = time.strptime(time_date, "%Y%m%d%H%M%S")
    # time_array = time.strptime(time_date, "%Y-%m-%d %H:%M:%S")
    time_stamp = int(time.mktime(time_array))
    return time_stamp


def stamp2date(time_stamp):
    time_array = time.localtime(time_stamp)
    time_date = time.strftime("%Y%m%d%H%M%S", time_array)
    # time_date = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    return time_date


def stamp2array(time_stamp):
    return time.localtime(float(time_stamp))


# Distance function
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def stay_center(points):
    center = [0, 0]
    for p in points:
        center[0] += p[0]
        center[1] += p[1]
    center[0] /= len(points)
    center[1] /= len(points)
    return center


def cut_point(point_str):
    tmp = point_str.split('.')
    return float(tmp[0] + '.' + tmp[1][0:3])


def round_point(lat_lon):
    lat_lon_3 = []
    for p in lat_lon:
        lat_lon_3.append(int(p * 1000) / 1000.0)
    return lat_lon_3


def smooth_traces(user_stay, place):
    for i, p in enumerate(user_stay):
        if distance(p[0:2], place) < spatial_threshold:
            user_stay[i][0:2] = place
    return user_stay


# Pre_processing trace data
def preprocessing(line):
    user_id, traces = line.split(' ')
    # user_id, traces = line.split('\t')
    user_trace = traces.split(';')

    user_traces = []
    for i, point in enumerate(user_trace):
        # point: lat,lon,time_date->time_stamp
        pp = point.strip('\n').split(',')
        if len(pp) != 3:
            continue
        time_stamp = date2stamp(pp[2])
        user_traces.append([float(pp[0]), float(pp[1]), time_stamp])

    user_pass = merge_same(user_traces)  # merge same point
    user_stay_tmp = merge_near(user_pass)  # merge near point

    # filter pass point by duration threshold
    temporal_threshold = 60 * time_slot  # 10 mins
    user_stay = []
    for point in user_stay_tmp:
        if (point[3] - point[2]) < temporal_threshold:
            pass
        else:
            point[0:2] = point[0:2]
            user_stay.append(point)
    if len(user_stay) <= active_user_stay:
        return {'user_stay': len(user_stay)}
    # identify location types
    home = identify_home(user_stay)
    if len(home) > 0:
        user_stay = smooth_traces(user_stay, home)
        work = identify_work(user_stay, home)
    else:
        work = []
    if len(work) > 0:
        user_stay = smooth_traces(user_stay, work)

    stay_count = len(user_stay)
    start_day = stamp2array(user_stay[0][2]).tm_yday
    end_day = stamp2array(user_stay[-1][3]).tm_yday


    home_stay_count = 0
    work_stay_count = 0
    for point in user_stay:
        if len(home) > 0 and distance(point[0:2], home) == 0:
            location_type = 0
            home_stay_count += 1
        elif len(work) > 0 and distance(point[0:2], work) == 0:
            location_type = 1
            work_stay_count += 1
        else:
            location_type = 2
        p = [location_type, point]
        stays.append(p)


    info = {}
    info['user_id'] = user_id
    info['user_stay'] = stay_count
    info['home'] = home
    info['home_stay'] = home_stay_count
    info['work'] = work
    info['work_stay'] = work_stay_count
    info['start_day'] = start_day
    info['end_day'] = end_day
    info['stays'] = stays
    return info


def identify_home(user_stay):
    candidate = {}

    for p in user_stay:
        pid = str(p[0]) + ',' + str(p[1])
        duration = float(p[3] - p[2])
        start_time = stamp2array(p[2])
        end_time = stamp2array(p[3])
        home_duration = float(24 - home_start + home_end)
        r = 0
        if start_time.tm_wday in [0, 1, 2, 3, 4]:
            if start_time.tm_hour > home_start:
                if end_time.tm_hour < home_end:
                    r = min(1, duration / home_duration / 3600)
                else:
                    r = min(1, (24 - start_time.tm_hour + home_end) / home_duration)
            else:
                if end_time.tm_hour < home_end:
                    r = min(1, (24 - home_start + end_time.tm_hour) / home_duration)
                else:
                    r = 1

        if pid in candidate:
            candidate[pid] += r
        else:
            candidate[pid] = r
    res = sorted(candidate.items(), key=lambda e: e[1], reverse=True)

    res_float = []
    for p in res:
        tmp = [float(x) for x in p[0].split(',')]
        pp = [tmp[0], tmp[1], p[1]]
        if len(res_float) == 0:
            res_float.append(pp)
        else:
            flag = 0
            for i, rp in enumerate(res_float):
                if distance(rp[0:2], pp[0:2]) < spatial_threshold:
                    res_float[i][2] += pp[2]
                    flag = 1
                    break
            if flag == 0:
                res_float.append(pp)

    if len(res_float) > 0 and sum([x[2] for x in res_float]) > 0:
        r = res_float[0][2] / float(sum([x[2] for x in res_float]))
    else:
        r = 0

    if r > home_threshold:
        return res_float[0][0:2]
    else:
        return []


def str2geo(geo):
    lon, lat = geo.split(',')
    return [float(lon), float(lat)]


def identify_work(user_stay, home):
    candidate = {}

    day_count = set()
    for p in user_stay:
        id = str(p[0]) + ',' + str(p[1])
        start_time = stamp2array(p[2])
        if start_time.tm_wday not in [0, 1, 2, 3, 4]:
            continue
        end_time = stamp2array(p[3])
        day_count.add(start_time.tm_wday)

        if start_time.tm_hour > work_start and end_time.tm_hour < work_end and start_time.tm_wday == end_time.tm_wday:
            for m in candidate:
                if distance(str2geo(id), str2geo(m)) < spatial_threshold:
                    candidate[m] += 1
                    continue
            candidate[id] = 1

    if len(candidate) == 0:
        return []
    day_count = len(day_count)
    for p in candidate:
        d = distance(home, [float(x) for x in p.split(',')])
        n = candidate[p]
        candidate[p] = [d, n]
    res = sorted(candidate.items(), key=lambda e: e[1][0] * e[1][1], reverse=True)
    if res[0][1][0] > work_spatial_threshold and res[0][1][1] / day_count >= work_frequency_threshold:
        return [float(x) for x in res[0][0].split(',')]
    else:
        return []


def merge_same(user_trace):
    last_place = []
    user_pass = []
    for point in user_trace:
        if last_place and distance(last_place, point[0:2]) < spatial_threshold * 0.1:
            continue
        else:
            if last_place:
                user_pass[-1][3] = point[2]
            stay = [point[0], point[1], point[2], point[2]]
            user_pass.append(stay)
            last_place = point[0:2]
    return user_pass


def merge_near(user_pass):
    merge_tmp = []
    user_stay = []
    for i, point in enumerate(user_pass):
        if len(merge_tmp) == 0:
            merge_tmp.append([point[0], point[1], i])
        else:
            flag = 0
            for p in merge_tmp:
                if distance(p[0:2], point[0:2]) < spatial_threshold:
                    merge_tmp.append([point[0], point[1], i])
                    flag = 1
                    break
            if flag:
                continue
            else:
                center = stay_center(merge_tmp)
                id1 = merge_tmp[0][2]
                id2 = merge_tmp[-1][2]
                ele = [center[0], center[1], user_pass[id1][2], user_pass[id2][3]]
                user_stay.append(ele)
                merge_tmp = [[point[0], point[1], i]]
    if len(merge_tmp) > 0:
        center = stay_center(merge_tmp)
        id1 = merge_tmp[0][2]
        id2 = merge_tmp[-1][2]
        ele = [center[0], center[1], user_pass[id1][2], user_pass[id2][3]]
        user_stay.append(ele)
    return user_stay


def individual_nw(info):
    user_trace = info['stays']

    nw = 0
    for i, p in enumerate(user_trace):
        if i == len(user_trace) - 1:
            break
        else:
            trip_origin = user_trace[i][0]
            trip_end = user_trace[i + 1][0]
            if trip_origin == 0 and trip_end == 2:
                nw += 1
    nw = nw * 7.0 / (info['end_day'] - info['start_day'] + 1)
    nw += 2
    return nw


def global_displacement():
    delta_r = [0] * int(max_explore_range)
    for idx in range(len(delta_r)):
        delta_r[idx] = (idx+1) ** (-0.86)
    delta_r = np.array(delta_r) / sum(np.array(delta_r))
    return delta_r


def global_reduce(pairs):
    c = []
    for i, a in enumerate(list(pairs[1])):
        if i == 0:
            c = a
        else:
            for j, d in enumerate(a):
                c[j] += a[j]
    total = float(sum(c))
    c2 = [x / total for x in c]
    return c2


def predict_next_place_time(n_w, P_t, beta1, beta2, current_location_type):
    p1 = 1 - n_w * P_t
    p2 = 1 - beta1 * n_w * P_t
    p3 = beta1 * n_w * P_t * beta2 * n_w * P_t
    location_is_change = 0
    new_location_type = 'undefined'
    if current_location_type == 'home':
        if random.uniform(0, 1) <= p1:
            new_location_type = 'home'
            location_is_change = 0
        else:
            new_location_type = 'other'
            location_is_change = 1
    elif current_location_type == 'other' or current_location_type == 'work':
        p = random.uniform(0, 1)
        if p <= p2:
            new_location_type = current_location_type
            location_is_change = 0
        elif p <= p2 + p3:
            new_location_type = 'other'
            location_is_change = 1
        else:
            new_location_type = 'home'
            location_is_change = 1
    if new_location_type == 'home':
        return 'home', location_is_change
    else:
        return 'other', location_is_change


def truncated_cauchy_rvs(loc=0, scale=1, a=-1, b=1, size=None):
    ua = np.arctan((a - loc)/scale)/np.pi + 0.5
    ub = np.arctan((b - loc)/scale)/np.pi + 0.5
    U = np.random.uniform(ua, ub, size=size)
    rvs =  loc + scale * np.tan(np.pi*(U - 0.5))
    return rvs

def global_rhythm(info):
    rhythm = [0] * 7 * 24 * int(60 / time_slot)
    for i, p in enumerate(info['stays'][1:-1]):
        # point:[type,[lat,lon,start_time,end_time]]
        np = p[1]
        try:
            np_start = stamp2array(np[2])
            np_end = stamp2array(np[3])
            start_id = int(np_start.tm_wday * 24 * 6 + np_start.tm_hour * 6 + np_start.tm_min / time_slot)
            if info['stays'][i - 1][0] != 1 and p[0] != 1:
                rhythm[start_id] += 1
        except:
            pass
    return rhythm

def global_rhythm_non_commuters(info):
    rhythm = [0] * 7 * 24 * int(60 / time_slot)
    for i, p in enumerate(info['stays'][1:-1]):
        np = p[1]
        try:
            np_start = stamp2array(np[2])
            np_end = stamp2array(np[3])
            start_id = int(np_start.tm_wday * 24 * 6 + np_start.tm_hour * 6 + np_start.tm_min / time_slot)
            if info['stays'][i - 1][0] == 0 and p[0] != 0:
                rhythm[start_id] += 1
        except:
            pass
    return rhythm

def earth_distance(lat_lng1, lat_lng2):
    lat1, lng1 = [l*pi/180 for l in lat_lng1]
    lat2, lng2 = [l*pi/180 for l in lat_lng2]
    dlat, dlng = lat1-lat2, lng1-lng2
    ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    return 6371.01 * ds


def Travel(param,self,gen_day):
    data = np.load(param.save_path + 'data/stay.npy', allow_pickle = True).item()
    gps = np.load('./data/MME/GPS_0.npy', allow_pickle = True)
    
    day = gen_day
    
    key_list = list(data.keys())
    user_list = []
    for i in key_list:
        if len(data[i]) < day:
            user_list.append(i)
        else:
            for b in range(day):
                if data[i][b]['loc'].size == 0:
                    user_list.append(i)

    for a in user_list:
        del data[a]
    
    data_f = defaultdict(dict)
    for i in data:
        for b in range(day):
            data_f[i][b] = {}
            data_f[i][b]['loc'] = data[i][b]['loc']
            data_f[i][b]['gps'] = []
            data_f[i][b]['tim'] = data[i][b]['tim']
            data_f[i][b]['sta'] = data[i][b]['sta']

            for a in range(len(data[i][b]['loc'])):
                data_f[i][b]['gps'].append(gps[data[i][b]['loc'][a]])
    
    ori_data = data_f
    
    for u in ori_data: 
        for b in range(day):
            gps = ori_data[u][b]['gps']
            tim = ori_data[u][b]['tim']
            ori_data[u][b]['travel'] = []
            a = len(ori_data[u][b]['loc'])
            for i in range(a):
                try:
                    if earth_distance(gps[i+1], gps[i]) < 3:
                        ori_data[u][b]['travel'].append(1) 
                    else:
                        ori_data[u][b]['travel'].append(2) 
                except:
                    continue
    
    json_data = defaultdict(dict)

    for u in ori_data:
        for b in range(day):
            json_data[u][b] = {}
            json_data[u][b]['gps'] = []
            json_data[u][b]['tim'] = []
            json_data[u][b]['loc'] = []
            json_data[u][b]['travel'] = []

            json_data[u][b]['gps'] = ori_data[u][b]['gps']
            json_data[u][b]['loc'] = ori_data[u][b]['loc'].tolist()
            json_data[u][b]['travel'] = ori_data[u][b]['travel']
            for i in range(len(ori_data[u][b]['tim'])):
                json_data[u][b]['tim'].append(ori_data[u][b]['tim'][i])

            json_data[u][b]['gps'].append(json_data[u][b]['gps'][0])
            json_data[u][b]['loc'].append(json_data[u][b]['loc'][0])

            if (earth_distance(json_data[u][b]['gps'][-1], json_data[u][b]['gps'][-2]) < 3) and (earth_distance(json_data[u][b]['gps'][-1], json_data[u][b]['gps'][-2])>0):
                json_data[u][b]['travel'].append(1)
            elif earth_distance(json_data[u][b]['gps'][-1], json_data[u][b]['gps'][-2]) == 0:
                json_data[u][b]['travel'].append(0)
            else:
                json_data[u][b]['travel'].append(2) 


            if (json_data[u][b]['tim'][-1] <= 1440*(b+1)) and (json_data[u][b]['tim'][-1] > 1380*(b+1)):
                json_data[u][b]['tim'].append(1440*(b+1)) 
            elif(json_data[u][b]['tim'][-1] <= 1380*(b+1)) and (json_data[u][b]['tim'][-1] >= 1320*(b+1)):
                json_data[u][b]['tim'].append(json_data[u][b]['tim'][-1] + 45)
            else:
                json_data[u][b]['tim'].append(json_data[u][b]['tim'][-1] + 60)
            
    json_data_new = json_data
    
    chazhi_data = defaultdict(dict)
    for u in json_data_new:
        for b in range(day):
            chazhi_data[u][b] = {}
            chazhi_data[u][b]['gps'] = []
            chazhi_data[u][b]['tim'] = []
            chazhi_data[u][b]['loc'] = []
            chazhi_data[u][b]['travel'] = []

            try:
                gps = json_data_new[u][b]['gps']
                tim = json_data_new[u][b]['tim']
                loc = json_data_new[u][b]['loc']
                travel = json_data_new[u][b]['travel']
                a = len(json_data_new[u][b]['travel'])
            except:
                continue

            try:
                if (a < 2):
                    chazhi_data[u][b]['tim'] = tim[0]
                    chazhi_data[u][b]['loc'] = loc[0]
                    chazhi_data[u][b]['gps'] = gps[0]
                    chazhi_data[u][b]['travel'] = travel
            except:
                continue

            if (a >= 2):
                for i in range(a):
                    if json_data_new[u][b]['travel'][i] == 1:  
                        timdiff = (earth_distance(gps[i+1], gps[i]))/0.085
                        timnew =  json_data_new[u][b]['tim'][i+1] - timdiff
                        chazhi_data[u][b]['tim'].append(tim[i])
                        chazhi_data[u][b]['tim'].append(timnew)
                        chazhi_data[u][b]['loc'].append(loc[i])
                        chazhi_data[u][b]['loc'].append(loc[i])
                        chazhi_data[u][b]['gps'].append(gps[i])
                        chazhi_data[u][b]['gps'].append(gps[i])
                        chazhi_data[u][b]['travel'].append(0)
                        chazhi_data[u][b]['travel'].append(travel[i])
                    else:   
                        timdiff = (earth_distance(gps[i+1], gps[i]))/0.5
                        timnew =  json_data_new[u][b]['tim'][i+1] - timdiff
                        chazhi_data[u][b]['tim'].append(tim[i])
                        chazhi_data[u][b]['tim'].append(timnew)
                        chazhi_data[u][b]['loc'].append(loc[i])
                        chazhi_data[u][b]['loc'].append(loc[i])
                        chazhi_data[u][b]['gps'].append(gps[i])
                        chazhi_data[u][b]['gps'].append(gps[i])
                        chazhi_data[u][b]['travel'].append(0)
                        chazhi_data[u][b]['travel'].append(travel[i])
                    if i == a-1:
                        chazhi_data[u][b]['tim'].append(tim[i+1])
                        chazhi_data[u][b]['loc'].append(loc[i+1])
                        chazhi_data[u][b]['gps'].append(gps[i+1])


    for u in chazhi_data:
        for b in range(day):
            try:
                c = len(chazhi_data[u][b]['tim'])
                for i in range(c-1):
                    if chazhi_data[u][b]['tim'][i+1] < chazhi_data[u][b]['tim'][i]:

                        chazhi_data[u][b]['tim'][i+1] = chazhi_data[u][b]['tim'][i]
            except:
                continue
    
    json_data = chazhi_data
    for u in json_data:
        for b in range(day):
            if type(json_data[u][b]['tim']) == list:
                json_data[u][b]['gps'].append(json_data[u][b]['gps'][0])
                json_data[u][b]['loc'].append(json_data[u][b]['loc'][0])

                if (json_data[u][b]['tim'][-1] <= 1440*(b+1)) and (json_data[u][b]['tim'][-1] > 1380*(b+1)):
                    json_data[u][b]['tim'].append(1440*(b+1)) 
                elif(json_data[u][b]['tim'][-1] <= 1380*(b+1)) and (json_data[u][b]['tim'][-1] >= 1320*(b+1)):
                    json_data[u][b]['tim'].append(json_data[u][b]['tim'][-1] + 45)
                else:
                    json_data[u][b]['tim'].append(json_data[u][b]['tim'][-1] + 60)
    
    chazhi_data = json_data
    for u in chazhi_data:
        for b in range(day):
            del chazhi_data[u][b]['travel']
    
    return chazhi_data


def simulate_traces(info, rhythm_global, deltar_global, run_mode, expansion=1):
    n_w = individual_nw(info)
    if run_mode == 'spark':
        P_t = rhythm_global.value[0]
    elif run_mode == 'streaming':
        P_t = rhythm_global
        deltar = deltar_global
    day_duration = min(float(info['end_day'] - info['start_day'])+1, 7)
    start_timestamp = info['stays'][0][1][2]

    stay_duration_history = []
    trip_count_history = len(info['stays'])
    for p in info['stays']:
        duration = p[1][3] - p[1][2]
        stay_duration_history.append(duration)

    test_pool = []
    for x1 in range(2, 20, 2):
        for x2 in range(1, 101, 5):
            location_type_now = 'home'
            trip_count_simulate = 0
            stay_duration_simulate = []
            time_id = 0
            location_duration_start = 0

            for timer_shift in range(time_slot * 60, 60 * 60 * 24 * int(day_duration), time_slot * 60):
                time_id += 1
                location_type_now, location_change = predict_next_place_time(n_w, P_t[time_id], x1, x2, location_type_now)
                if location_change:
                    trip_count_simulate += 1
                    stay_duration_simulate.append(timer_shift-location_duration_start)
                    location_duration_start = timer_shift

            if len(stay_duration_simulate) == 0:
                test_pool.append(1e10)
            else:
                test = pdf_minus(stay_duration_simulate, stay_duration_history) + eta * abs(trip_count_history - trip_count_simulate) / day_duration
                test_pool.append(test)
    beta_id = test_pool.index(min(test_pool))
    beta1 = np.floor(beta_id / 20)*2 + 2 
    beta2 = beta_id%20*5 + 1
    info['beta'] = [beta1, beta2]
    info['n_w'] = n_w

    location_history = {}
    count = 0
    for p in info['stays']:
        if p[1][:2] == info['work']:
            continue
        count += 1
        pid = str(p[1][0])[0:7] + ',' + str(p[1][1])[0:6]
        if pid in location_history.keys():
            location_history[pid][1] += 1
        else:
            location_history[pid] = [p[0], 1]
    s = len(location_history)
    if s == 0:
        P_new = 1
    else:
        for lo in location_history:
            location_history[lo][1] /= float(count)
        P_new = rho * s ** (-gamma)

    def time_filter(traj):
        temporal_threshold = 180 * time_slot
        user_stay = []
        for point in traj:
            if (point[3] - point[2]) < temporal_threshold:
                pass
            else:
                point[0:2] = round_point(point[0:2])
                user_stay.append(point)
        return user_stay

    def same_filter(user_trace):
        last_place = []
        user_pass = []
        for point in user_trace:
            if last_place and distance(last_place, point[0][1]) < spatial_threshold * 0.1:
                user_pass[-1][3] = point[1]
            else:
                if user_pass != []:
                    user_pass[-1][3] = point[1]
                stay = [point[0][1][0], point[0][1][1], point[1], point[1]]
                user_pass.append(stay)
                last_place = point[0][1]
        return user_pass

    new_trajs = []
    for m in range(expansion):
        work_start_time, work_duration = 0.17*np.random.multivariate_normal(mean=[12.8, 6.6], cov=[[3.7*3.7, -4.3], [-4.3, 4.4*4.4]]) + 0.29*np.random.multivariate_normal(mean=[7.9, 7.5], cov=[[1.5*1.5, -2.6], [-2.6, 3.2*3.2]]) + 0.53*np.random.multivariate_normal(mean=[7.6, 9.0], cov=[[1, -0.3], [-0.3, 0.9*0.9]])
        theta = 0.2
        if random.uniform(0, 1) <= theta:
            work_break = True
            work_break_duration = np.random.lognormal(3.9, 0.9)/60
            D = truncated_cauchy_rvs(0, 0.1, -0.5, 0.5)
            work_break_start_time = work_start_time + (work_duration-work_break_duration)*(0.5+D)                                                                         
        else:
            work_break = False

        location_type_now = 'home'
        trajs = [] 
        day_duration = 1
        current_location = [0, info['home']]
        zero_time = date2stamp('20200517000000')
        trajs.append([[0, info['home']], zero_time])

        work_start_time = zero_time+np.floor(work_start_time*3600)
        work_duration = np.floor(work_duration*3600)
        for timer_shift in range(0, 60 * 60 * 24 * int(day_duration), time_slot * 60):
            timer = zero_time + timer_shift
            tm = stamp2array(timer)
            if timer > work_start_time:
                break
            time_id = int(tm.tm_wday * 24 * 6 + tm.tm_hour * 6 + tm.tm_min / time_slot)
            location_type, location_change = predict_next_place_time(n_w, P_t[time_id], beta1, beta2, location_type_now)
            location_type_now = location_type
            if location_change:
                next_location = predict_next_place_location_simplify(P_new, deltar, location_history,
                                                                     current_location[1])
                trajs.append([next_location, timer])
                current_location = next_location
        trajs.append([[1, info['work']], work_start_time])
        current_location = [1, info['work']]
        location_type_now = 'work'

        if work_break is True:
            work_break_start_time = zero_time+np.floor(work_break_start_time*3600)
            next_location = predict_next_place_location_simplify(P_new, deltar, location_history, current_location[1])
            trajs.append([next_location, work_break_start_time])
            current_location = next_location
            for timer_shift in range(0, 60 * 60 * 24 * int(day_duration), time_slot * 60):
                if timer_shift >= work_break_duration*3600:
                    break
                timer = work_break_start_time + timer_shift
                tm = stamp2array(timer)
                time_id = int(tm.tm_wday * 24 * 6 + tm.tm_hour * 6 + tm.tm_min / time_slot)
                location_type, location_change = predict_next_place_time(n_w, P_t[time_id], beta1, beta2, location_type_now)
                location_type_now = location_type
                if location_change:
                    next_location = predict_next_place_location_simplify(P_new, deltar, location_history,
                                                                     current_location[1])
                    trajs.append([next_location, timer])
                    current_location = next_location
            trajs.append([[1, info['work']], work_break_start_time+work_duration*3600])
            current_location = [1, info['work']]
            location_type_now = 'work'

        start_day = stamp2array(start_timestamp).tm_wday
        start_time = zero_time + work_start_time+ work_duration
        for timer_shift in range(0, 60 * 60 * 24 * int(day_duration), time_slot * 60):
            timer = start_time + timer_shift
            tm = stamp2array(timer)
            if tm.tm_wday - start_day >= int(day_duration):
                break
            time_id = int(tm.tm_wday * 24 * 6 + tm.tm_hour * 6 + tm.tm_min / time_slot)
            location_type, location_change = predict_next_place_time(n_w, P_t[time_id], beta1, beta2, location_type_now)
            location_type_now = location_type
            if location_change:
                next_location = predict_next_place_location_simplify(P_new, deltar, location_history,
                                                                     current_location[1])
                trajs.append([next_location, timer])
                current_location = next_location
        new_trajs.append(trajs)
    return new_trajs


def simulate_traces_non_commuters(info, rhythm_global, deltar_global, run_mode, expansion=1):
    n_w = individual_nw(info)
    if run_mode == 'spark':
        P_t = rhythm_global.value[0]
    elif run_mode == 'streaming':
        P_t = rhythm_global
        deltar = deltar_global
    day_duration = min(float(info['end_day'] - info['start_day'])+1, 7)
    start_timestamp = info['stays'][0][1][2]

    stay_duration_history = []
    trip_count_history = len(info['stays'])
    for p in info['stays']:
        duration = p[1][3] - p[1][2]
        stay_duration_history.append(duration)

    test_pool = []
    for x1 in range(2, 20, 2):
        for x2 in range(1, 101, 5):
            location_type_now = 'home'
            trip_count_simulate = 0
            stay_duration_simulate = []
            time_id = 0
            location_duration_start = 0

            for timer_shift in range(time_slot * 60, 60 * 60 * 24 * int(day_duration), time_slot * 60):
                time_id += 1
                location_type_now, location_change = predict_next_place_time(n_w, P_t[time_id], x1, x2, location_type_now)
                if location_change:
                    trip_count_simulate += 1
                    stay_duration_simulate.append(timer_shift-location_duration_start)
                    location_duration_start = timer_shift
            if len(stay_duration_simulate) == 0:
                test_pool.append(1e10)
            else:
                test = pdf_minus(stay_duration_simulate, stay_duration_history) + eta * abs(trip_count_history - trip_count_simulate) / day_duration
                test_pool.append(test)
    beta_id = test_pool.index(min(test_pool))
    beta1 = np.floor(beta_id / 20)*2 + 2 
    beta2 = beta_id%20*5 + 1
    info['beta'] = [beta1, beta2]
    info['n_w'] = n_w

    location_history = {}
    count = 0
    for p in info['stays']:
        count += 1
        pid = str(p[1][0])[0:7] + ',' + str(p[1][1])[0:6]
        if pid in location_history.keys():
            location_history[pid][1] += 1
        else:
            location_history[pid] = [p[0], 1]
    s = len(location_history)
    for lo in location_history:
        location_history[lo][1] /= float(count)
    P_new = rho * s ** (-gamma)

    def time_filter(traj):
        temporal_threshold = 180 * time_slot  # 30 mins
        user_stay = []
        for point in traj:
            if (point[3] - point[2]) < temporal_threshold:
                pass
            else:
                point[0:2] = round_point(point[0:2])
                user_stay.append(point)
        return user_stay

    def same_filter(user_trace):
        last_place = []
        user_pass = []
        for point in user_trace:
            if last_place and distance(last_place, point[0][1]) < spatial_threshold * 0.1:
                user_pass[-1][3] = point[1]
            else:
                if user_pass != []:
                    user_pass[-1][3] = point[1]
                stay = [point[0][1][0], point[0][1][1], point[1], point[1]]
                user_pass.append(stay)
                last_place = point[0][1]
        return user_pass

    new_trajs = []
    for m in range(expansion):
        location_type_now = 'home'
        trajs = [] 
        day_duration = 1
        current_location = [0, info['home']]
        zero_time = date2stamp('20160420000000')
        trajs.append([[0, info['home']], zero_time])
        start_day = stamp2array(start_timestamp).tm_wday

        for timer_shift in range(0, 60 * 60 * 24 * int(day_duration), time_slot * 60):
            timer = start_timestamp + timer_shift
            tm = stamp2array(timer)
            if tm.tm_wday - start_day >= int(day_duration):
                break
            time_id = int(tm.tm_wday * 24 * 6 + tm.tm_hour * 6 + tm.tm_min / time_slot)
            location_type, location_change = predict_next_place_time(n_w, P_t[time_id], beta1, beta2, location_type_now)
            location_type_now = location_type
            if location_change:
                next_location = predict_next_place_location_simplify(P_new, deltar, location_history,
                                                                     current_location[1])
                trajs.append([next_location, timer])
                current_location = next_location
        new_trajs.append(trajs)
    return new_trajs


def pdf_minus(stay_duration_simulate, stay_duration_history):
    delta = time_slot * 60  # 10 minutes
    length = int(max_stay_duration * 60.0 / time_slot)
    res1 = [0] * length
    res2 = [0] * length
    for p in stay_duration_history:
        pid = int(p / delta)
        res1[min(length - 1, pid)] += 1
    for p in stay_duration_simulate:
        pid = int(p / delta)
        res2[min(length - 1, pid)] += 1
    sum1 = float(sum(res1))
    sum2 = float(sum(res2))
    res1 = [x / sum1 for x in res1]
    res2 = [x / sum2 for x in res2]

    pdf_err = 0
    for i in range(0, length):
        pdf_err += abs(res1[i] - res2[i])
    pdf_err *= delta
    return pdf_err


def predict_next_place_location_simplify(P_new, delta_r, location_history, current_location):
    rp = random.uniform(0, 1)
    prob_accum = 0
    if random.uniform(0, 1) < P_new:
        # explore
        for i, r in enumerate(delta_r):
            prob_accum += r
            if rp < prob_accum:
                radius = (i + 1) * spatial_resolution
                direction = random.uniform(0, 1) * 360
                next_lat = current_location[1] + radius * math.sin(math.radians(direction))
                next_lon = current_location[0] + radius * math.cos(math.radians(direction))
                next_location = [2, [next_lon, next_lat]]
                break
    else:
        return_selection = sorted(location_history.items(), key=lambda x: x[1], reverse=True)
        for lo in return_selection:
            prob_accum += lo[1][1]
            if rp < prob_accum:
                next_location = [lo[1][0], [float(x) for x in lo[0].split(',')]]
                break
    return next_location


def predict_next_place_location(info, delta_r, current_location):
    location_history = {}
    for p in info['stays']:
        pid = str(p[1][0]) + ',' + str(p[1][1])
        if pid in location_history:
            location_history[pid][1] += 1
        else:
            location_history[pid] = [p[0], 1]
    s = len(location_history)
    for lo in location_history:
        location_history[lo][1] /= len(info['stays'])

    P_new = rho * s ** (-gamma)
    return predict_next_place_location_simplify(P_new, delta_r, location_history, current_location)


def time_id(time_stamp):
    ti = stamp2array(time_stamp)
    return int(ti.tm_wday * 24 * 60 / time_slot + ti.tm_hour * 60 / time_slot + ti.tm_min / time_slot)


def predict_evaluate(info, rhythm_global, deltar_global, run_mode):
    start_location = info['ground_truth'][0]
    end_location = info['ground_truth'][-1]
    start_time = start_location[1][2]
    end_time = end_location[1][3]
    ground_duration = (end_time - start_time) / 60 / time_slot

    n_w = info['n_w']
    beta1, beta2 = info['beta']
    if run_mode == 'spark':
        deltar = deltar_global.value[0]
        rhythm = rhythm_global.value[0]
    elif run_mode == 'streaming':
        deltar = deltar_global
        rhythm = rhythm_global
    location_type = ['home', 'work', 'other']

    location_history = {}
    for p in info['stays']:
        pid = str(p[1][0])[0:7] + ',' + str(p[1][1])[0:6]
        if pid in location_history.keys():
            location_history[pid][1] += 1
        else:
            location_history[pid] = [p[0], 1]
    s = len(location_history)
    for lo in location_history:
        location_history[lo][1] /= float(len(info['stays']))
    P_new = rho * s ** (-gamma)

    ground_trace = [0] * int(ground_duration)
    for p in info['ground_truth']:
        p_in = p[1][2]
        p_out = p[1][3]
        bins = int((p_out - p_in) / 60 / time_slot)
        time_id_now = int((p_in - start_time) / 60 / time_slot)
        for t in range(time_id_now, time_id_now + bins):
            ground_trace[t] = [p[0], p[1][0], p[1][1], p_in + t * 60 * time_slot]

    predict_trace = [0] * int(ground_duration)
    current_location = [start_location[0], start_location[1][0:2]]
    for tid in range(70, int(ground_duration)):
        t = start_time + tid * 60 * time_slot
        ta = stamp2array(t)
        if ta.tm_hour in [2, 3, 4]:
            tmp = ground_trace[tid]
            predict_trace[tid] = tmp
            if tmp != 0:
                current_location = [tmp[0], [tmp[1], tmp[2]]]
        else:
            P_t = rhythm[time_id(t)]
            now_type, location_change = predict_next_place_time(n_w, P_t, beta1, beta2,
                                                                location_type[current_location[0]])
            if location_change:
                next_location = predict_next_place_location_simplify(P_new, deltar, location_history,
                                                                     current_location[1])
            else:
                next_location = [now_type, current_location[1]]
            current_location = next_location
            predict_trace[tid] = [next_location[0], next_location[1][0], next_location[1][1], t]

    predict_correct = 0
    for n in range(0, len(predict_trace)):
        if ground_trace[n] == 0 or predict_trace[n] == 0:
            continue
        elif distance(predict_trace[n][1:3], ground_trace[n][1:3]) < performance_spatial_threshold and stamp2array(
                t).tm_hour not in [2, 3, 4]:
            predict_correct += 1

    predict = 0
    ground = 0
    for i, j in zip(predict_trace, ground_trace):
        if i != 0:
            predict += 1
        if j != 0:
            ground += 1
    return [predict_correct, predict, ground]






