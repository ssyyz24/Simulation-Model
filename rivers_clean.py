#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 00:59:30 2026

@author: zhaoyiheng
"""
import simpy
import pandas as pd
import numpy as np
import math


SPEED_MPH = 20.0             # 车辆平均速度 20 mph
RIDER_ARRIVAL_RATE = 30.0    # 乘客到达率 (每小时 30 单) - 后续可替换为 piecewise rate
DRIVER_LOGIN_RATE = 3.0      # 司机上线率 (每小时 3 人) - 后续可替换为 piecewise rate
SIM_HOURS = 24.0             # 总仿真时长 (24小时)
WARMUP_HOURS = 2.0           # 预热时长 (前2小时的数据不计入KPI)

# 辅助函数：计算欧氏距离
def calc_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ==========================================
# 1. 仿真环境与核心逻辑 (OOP 设计)
# ==========================================
class BoxCarSim:
    def __init__(self, env, riders_pool, online_hours_pool):
        self.env = env
        self.riders_pool = riders_pool
        self.online_hours_pool = online_hours_pool
        
        # 状态维护 (State variables)
        self.idle_drivers = {}     # driver_id -> {'x': x, 'y': y, 'logout_time': t, 'total_busy': 0}
        self.busy_drivers = set()  # driver_id
        self.waiting_queue = []    # 存放排队乘客的字典
        
        # KPI 统计 (仅记录 Warm-up 之后的数据)
        self.stats = {
            'wait_times': [],
            'completed_trips': 0,
            'total_online_time': 0,
            'total_busy_time': 0
        }
        
        # 启动事件生成器
        self.env.process(self.generate_riders())
        self.env.process(self.generate_drivers())
        
        # 司机 ID 生成器
        self.next_driver_id = 1

    # --- 事件生成器 (Event Generators) ---
    def generate_riders(self):
        """乘客到达事件过程"""
        while True:
            # 抽样下一次到达的时间间隔 (指数分布)
            inter_arrival = np.random.exponential(1.0 / RIDER_ARRIVAL_RATE)
            yield self.env.timeout(inter_arrival)
            
            # 从经验数据中抽样真实的 OD 和行程时间
            sample = self.riders_pool.sample(1).iloc[0]
            rider = {
                'arr_time': self.env.now,
                'px': sample['pickup_x'],
                'py': sample['pickup_y'],
                'dx': sample['dropoff_x'],
                'dy': sample['dropoff_y'],
                'trip_hours': sample['trip_hours']
            }
            
            # 乘客到达，触发派单逻辑
            self.waiting_queue.append(rider)
            self.dispatch()

    def generate_drivers(self):
        """司机上线事件过程"""
        while True:
            inter_login = np.random.exponential(1.0 / DRIVER_LOGIN_RATE)
            yield self.env.timeout(inter_login)
            
            # 从经验池抽样司机的在线时长
            online_duration = np.random.choice(self.online_hours_pool)
            
            # 简单起见，司机初始位置从接驾点经验池里随机抽一个
            sample = self.riders_pool.sample(1).iloc[0]
            
            driver_id = self.next_driver_id
            self.next_driver_id += 1
            
            # 记录司机状态
            self.idle_drivers[driver_id] = {
                'x': sample['pickup_x'],
                'y': sample['pickup_y'],
                'logout_time': self.env.now + online_duration,
                'total_busy': 0,
                'login_time': self.env.now
            }
            
            # 司机上线，检查是否有排队的乘客
            self.dispatch()
            
            # 启动一个独立进程来监控该司机何时下线
            self.env.process(self.monitor_driver_logout(driver_id, online_duration))

    def monitor_driver_logout(self, driver_id, online_duration):
        """监控司机下线，如果时间到了且空闲，则离线"""
        yield self.env.timeout(online_duration)
        if driver_id in self.idle_drivers:
            # 统计在线时长和忙碌时长 (只统计预热后的增量)
            if self.env.now > WARMUP_HOURS:
                d_info = self.idle_drivers[driver_id]
                self.stats['total_online_time'] += (self.env.now - max(d_info['login_time'], WARMUP_HOURS))
                self.stats['total_busy_time'] += d_info['total_busy']
            # 移出系统
            del self.idle_drivers[driver_id]

    # --- 核心调度逻辑 (Dispatching Logic) ---
    # --- 核心调度逻辑 (Dispatching Logic) ---
    def dispatch(self):
        """每次有乘客到达或司机变空闲时，调用匹配规则"""
        riders_to_remove = []
        for rider in self.waiting_queue:
            if not self.idle_drivers:
                break # 没有空闲司机了，停止匹配
            
            best_driver = None
            min_dist = float('inf')
            
            for did, d_info in self.idle_drivers.items():
                if self.env.now >= d_info['logout_time']:
                    continue
                dist = calc_distance(d_info['x'], d_info['y'], rider['px'], rider['py'])
                if dist < min_dist:
                    min_dist = dist
                    best_driver = did
                    
            if best_driver is not None:
                # 匹配成功！
                riders_to_remove.append(rider)
                
                # 【修复核心】立刻把司机从空闲字典里“挖走”，防止他被重复派单！
                d_info = self.idle_drivers.pop(best_driver) 
                
                # 启动行程服务进程 (把挖出来的 d_info 也传过去)
                self.env.process(self.serve_trip(best_driver, d_info, rider, min_dist))
                
        # 将已匹配的乘客从队列中移除
        for r in riders_to_remove:
            self.waiting_queue.remove(r)

    def serve_trip(self, driver_id, d_info, rider, pickup_dist):
        """执行行程服务"""
        # (因为已经在 dispatch 里 pop 出来了，这里就不需要再 pop 了)
        self.busy_drivers.add(driver_id)
        
        # 1. 计算前往接驾的时间
        pickup_time = pickup_dist / SPEED_MPH
        
        # 2. 计算总共忙碌的时间 (接驾 + 行程时间)
        total_service_time = pickup_time + rider['trip_hours']
        
        # 模拟时间的流逝 (车辆行驶中...)
        yield self.env.timeout(total_service_time)
        
        # 行程结束
        self.busy_drivers.remove(driver_id)
        
        # 统计 KPI (仅收集预热期之后的数据)
        if self.env.now > WARMUP_HOURS:
            wait_time_minutes = (self.env.now - total_service_time - rider['arr_time']) * 60
            self.stats['wait_times'].append(wait_time_minutes)
            self.stats['completed_trips'] += 1
            d_info['total_busy'] += total_service_time
        
        # 更新司机位置到下车点
        d_info['x'] = rider['dx']
        d_info['y'] = rider['dy']
        
        # 司机重新变为闲置，或者下线
        if self.env.now < d_info['logout_time']:
            self.idle_drivers[driver_id] = d_info
            # 司机空闲了，触发一次匹配检查队列里是否有人等
            self.dispatch()
        else:
            # 已经超过下线时间，服务完这单就下线，直接统计时间
            if self.env.now > WARMUP_HOURS:
                self.stats['total_online_time'] += (self.env.now - max(d_info['login_time'], WARMUP_HOURS))
                self.stats['total_busy_time'] += d_info['total_busy']

# ==========================================
# 2. 运行实验 (Replications & Reporting)
# ==========================================
def run_experiments(n_reps=10):
    print("Loading empirical data...")
    # 只取 Completed 且非异常坐标的数据作为抽样池
    df_riders = pd.read_csv('riders_clean.csv')
    valid_riders = df_riders[df_riders['status_group'] == 'completed'].dropna(subset=['pickup_x', 'pickup_y', 'dropoff_x', 'dropoff_y', 'trip_hours'])
    
    df_drivers = pd.read_csv('drivers_clean.csv')
    valid_online_hours = df_drivers['online_hours'].dropna().values
    
    all_median_waits = []
    all_p95_waits = []
    all_utilizations = []
    all_throughputs = []

    print(f"Running {n_reps} replications (Sim_time={SIM_HOURS}h, Warmup={WARMUP_HOURS}h)...")
    
    for i in range(n_reps):
        env = simpy.Environment()
        sim = BoxCarSim(env, valid_riders, valid_online_hours)
        env.run(until=SIM_HOURS)
        
        # 计算该 Replication 的 KPI
        waits = sim.stats['wait_times']
        if len(waits) > 0:
            all_median_waits.append(np.median(waits))
            all_p95_waits.append(np.percentile(waits, 95))
        else:
            all_median_waits.append(0)
            all_p95_waits.append(0)
            
        all_throughputs.append(sim.stats['completed_trips'])
        
        if sim.stats['total_online_time'] > 0:
            util = sim.stats['total_busy_time'] / sim.stats['total_online_time']
        else:
            util = 0
        all_utilizations.append(util)
        
    # 打印总体统计信息 (Point estimate + 95% CI 近似)
    print("\n" + "="*40)
    print("🎯 BASELINE v0 KPI SCORECARD (Post Warm-up)")
    print("="*40)
    print(f"Median Wait Time: {np.mean(all_median_waits):.2f} mins (± {1.96 * np.std(all_median_waits)/math.sqrt(n_reps):.2f})")
    print(f"95th Pct Wait Time: {np.mean(all_p95_waits):.2f} mins (± {1.96 * np.std(all_p95_waits)/math.sqrt(n_reps):.2f})")
    print(f"Driver Utilization: {np.mean(all_utilizations)*100:.1f}% (± {1.96 * np.std(all_utilizations)*100/math.sqrt(n_reps):.1f}%)")
    print(f"Throughput (Trips): {np.mean(all_throughputs):.1f} (± {1.96 * np.std(all_throughputs)/math.sqrt(n_reps):.1f})")

if __name__ == "__main__":
    # 测试运行 5 次 replication (你可以改成 50 次)
    run_experiments(n_reps=5)
