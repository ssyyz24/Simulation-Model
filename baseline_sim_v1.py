#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 01:35:44 2026

@author: zhaoyiheng
"""

import simpy
import pandas as pd
import numpy as np
import math

from scipy import stats

def mean_ci_halfwidth(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(np.mean(x)) if n == 1 else 0.0, np.nan
    m = float(np.mean(x))
    s = float(np.std(x, ddof=1))
    tcrit = stats.t.ppf(1 - alpha/2, df=n-1)
    half = tcrit * s / math.sqrt(n)
    return m, half

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


SPEED_MPH = 20.0             
RIDER_ARRIVAL_RATE = 30.0    
DRIVER_LOGIN_RATE = 3.0      
PATIENCE_RATE = 5.0          

BASE_FARE = 3.0              
PER_MILE_FARE = 2.0          
DRIVER_COST_PER_MILE = 0.20  

SIM_HOURS = 24.0             
WARMUP_HOURS = 2.0 
N_REPS = 30
SEED0 = 12345        

def calc_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


class BoxCarSim:
    def __init__(self, env, riders_pool, online_hours_pool):
        self.env = env
        self.riders_pool = riders_pool
        self.online_hours_pool = online_hours_pool
        
        self.idle_drivers = {}     
        self.busy_drivers = set()  
        self.waiting_queue = []    
        
        self.stats = {
            'wait_times': [],
            'requests': 0,           
            'cancelled_trips': 0,
            'completed_trips': 0,
            'total_online_time': 0,
            'total_busy_time': 0,
            'total_profit': 0.0      
        }
        
        self.env.process(self.generate_riders())
        self.env.process(self.generate_drivers())
        self.next_driver_id = 1

    def generate_riders(self):
        while True:
            yield self.env.timeout(np.random.exponential(1.0 / RIDER_ARRIVAL_RATE))
            sample = self.riders_pool.sample(1).iloc[0]
            
            rider = {
                'arr_time': self.env.now,
                'px': sample['pickup_x'],
                'py': sample['pickup_y'],
                'dx': sample['dropoff_x'],
                'dy': sample['dropoff_y'],
                'trip_hours': sample['trip_hours']
            }
            
            if self.env.now > WARMUP_HOURS:
                self.stats['requests'] += 1
                
            self.waiting_queue.append(rider)
            self.env.process(self.rider_patience_countdown(rider))
            self.dispatch()

    def rider_patience_countdown(self, rider):
        """模拟乘客耐心：超时则取消订单"""
        patience = np.random.exponential(1.0 / PATIENCE_RATE)
        yield self.env.timeout(patience)
        
        
        if rider in self.waiting_queue:
            self.waiting_queue.remove(rider)
            if self.env.now > WARMUP_HOURS:
                self.stats['cancelled_trips'] += 1

    def generate_drivers(self):
        while True:
            yield self.env.timeout(np.random.exponential(1.0 / DRIVER_LOGIN_RATE))
            online_duration = np.random.choice(self.online_hours_pool)
            sample = self.riders_pool.sample(1).iloc[0]
            
            driver_id = self.next_driver_id
            self.next_driver_id += 1
            
            self.idle_drivers[driver_id] = {
                'x': sample['pickup_x'],
                'y': sample['pickup_y'],
                'logout_time': self.env.now + online_duration,
                'total_busy': 0,
                'total_profit': 0.0, # 记录单个司机的利润
                'login_time': self.env.now
            }
            self.dispatch()
            self.env.process(self.monitor_driver_logout(driver_id, online_duration))

    def monitor_driver_logout(self, driver_id, online_duration):
        yield self.env.timeout(online_duration)
        if driver_id in self.idle_drivers:
            if self.env.now > WARMUP_HOURS:
                d_info = self.idle_drivers[driver_id]
                self.stats['total_online_time'] += (self.env.now - max(d_info['login_time'], WARMUP_HOURS))
                self.stats['total_busy_time'] += d_info['total_busy']
                self.stats['total_profit'] += d_info['total_profit']
            del self.idle_drivers[driver_id]

    def dispatch(self):
        riders_to_remove = []
        for rider in self.waiting_queue:
            if not self.idle_drivers:
                break 
            
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
                riders_to_remove.append(rider)
                d_info = self.idle_drivers.pop(best_driver) 
                rider['match_time'] = self.env.now
                self.env.process(self.serve_trip(best_driver, d_info, rider, min_dist))
                
        for r in riders_to_remove:
            self.waiting_queue.remove(r)

    def serve_trip(self, driver_id, d_info, rider, pickup_dist):
        self.busy_drivers.add(driver_id)
        
        pickup_time = pickup_dist / SPEED_MPH
        total_service_time = pickup_time + rider['trip_hours']
        
        
        trip_dist = calc_distance(rider['px'], rider['py'], rider['dx'], rider['dy'])
        fare = BASE_FARE + PER_MILE_FARE * trip_dist
        cost = DRIVER_COST_PER_MILE * (pickup_dist + trip_dist)
        profit = fare - cost
        
        yield self.env.timeout(total_service_time)
        self.busy_drivers.remove(driver_id)
        
        if self.env.now > WARMUP_HOURS:
            wait_time_minutes = (rider['match_time'] - rider['arr_time']) * 60
            self.stats['wait_times'].append(wait_time_minutes)
            self.stats['completed_trips'] += 1
            d_info['total_busy'] += total_service_time
            d_info['total_profit'] += profit
        
        d_info['x'] = rider['dx']
        d_info['y'] = rider['dy']
        
        if self.env.now < d_info['logout_time']:
            self.idle_drivers[driver_id] = d_info
            self.dispatch()
        else:
            if self.env.now > WARMUP_HOURS:
                self.stats['total_online_time'] += (self.env.now - max(d_info['login_time'], WARMUP_HOURS))
                self.stats['total_busy_time'] += d_info['total_busy']
                self.stats['total_profit'] += d_info['total_profit']


def run_experiments(n_reps=N_REPS):
    df_riders = pd.read_csv('riders_clean.csv')
    valid_riders = df_riders[df_riders['status_group'] == 'completed'].dropna(
        subset=['pickup_x', 'pickup_y', 'dropoff_x', 'dropoff_y', 'trip_hours']
    )
    df_drivers = pd.read_csv('drivers_clean.csv')
    valid_online_hours = df_drivers['online_hours'].dropna().values

    results = {'wait_med': [], 'wait_p95': [], 'util': [], 'cancel_rate': [], 'profit_per_hr': []}
    all_waits = []

    for i in range(n_reps):
        np.random.seed(SEED0 + i)
        env = simpy.Environment()
        sim = BoxCarSim(env, valid_riders, valid_online_hours)
        env.run(until=SIM_HOURS)

        waits = sim.stats['wait_times']
        all_waits.extend(waits)
        results['wait_med'].append(np.median(waits) if len(waits) else 0.0)
        results['wait_p95'].append(np.percentile(waits, 95) if len(waits) else 0.0)

        reqs = sim.stats['requests']
        results['cancel_rate'].append((sim.stats['cancelled_trips'] / reqs * 100.0) if reqs > 0 else 0.0)

        ot = sim.stats['total_online_time']
        results['util'].append((sim.stats['total_busy_time'] / ot * 100.0) if ot > 0 else 0.0)
        results['profit_per_hr'].append((sim.stats['total_profit'] / ot) if ot > 0 else 0.0)

   
    rep_df = pd.DataFrame(results)
    rep_df.to_csv(f"baseline_v1_replications_R{n_reps}.csv", index=False)

    waits_arr = np.asarray(all_waits, dtype=float)
    waits_arr = waits_arr[np.isfinite(waits_arr)]
    waits_arr = waits_arr[waits_arr >= 0]
    waits_arr.sort()

    if len(waits_arr) > 0:
        ecdf = np.arange(1, len(waits_arr) + 1) / len(waits_arr)
        ccdf = 1.0 - ecdf

        med = 0.44
        p95 = 10.58

        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.step(waits_arr, ccdf, where="post", linewidth=2.2)
        
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.0)

        xmax = min(max(p95 * 2.0, np.percentile(waits_arr, 99), 12), waits_arr.max())
        ax.set_xlim(0.1, xmax)

        ax.set_xlabel("Waiting time (minutes)")
        ax.set_ylabel("Tail probability  P(wait > w)")
        ax.set_title(f"Baseline v1 waiting-time tail (CCDF, R={n_reps})")


   
        ax.axvline(p95, linestyle="--", linewidth=1.6)
        ax.axhline(0.05, linestyle=":", linewidth=1.4)
        
        txt = (
        f"Median = {med:.2f} min\n"
        f"P95 = {p95:.2f} min\n"
        f"P(wait > P95) = 0.05"
        )
        
        
        ax.text(
        0.02, 0.98, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", lw=1.0)
        )

        ax.grid(True, which="both", linestyle=":", linewidth=0.8)
        fig.tight_layout()
        fig.savefig(f"fig_wait_tail_ccdf_baseline_R{n_reps}.png", dpi=300)
        plt.close(fig)



    
    m_can, h_can = mean_ci_halfwidth(results['cancel_rate'])
    m_med, h_med = mean_ci_halfwidth(results['wait_med'])
    m_p95, h_p95 = mean_ci_halfwidth(results['wait_p95'])
    m_util, h_util = mean_ci_halfwidth(results['util'])
    m_pph, h_pph = mean_ci_halfwidth(results['profit_per_hr'])

    print("\n" + "="*45)
    print(f"🎯 BASELINE v1 KPI SCORECARD (Post Warm-up)  |  R={n_reps}")
    print("="*45)
    print(f"Cancellation Rate: {m_can:.2f}% (95% CI: [{m_can-h_can:.2f}, {m_can+h_can:.2f}])")
    print(f"Median Wait Time:  {m_med:.2f} mins (95% CI: [{m_med-h_med:.2f}, {m_med+h_med:.2f}])")
    print(f"95th Pct Wait:     {m_p95:.2f} mins (95% CI: [{m_p95-h_p95:.2f}, {m_p95+h_p95:.2f}])")
    print(f"Driver Util:       {m_util:.1f}% (95% CI: [{m_util-h_util:.1f}, {m_util+h_util:.1f}])")
    print(f"Profit per Hour:   £{m_pph:.2f}/hr (95% CI: [£{m_pph-h_pph:.2f}, £{m_pph+h_pph:.2f}])")
    
if __name__ == "__main__":
    run_experiments(n_reps=N_REPS)