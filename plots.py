#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 03:56:02 2026

@author: zhaoyiheng
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置画图风格
matplotlib.use("Agg")
plt.style.use('seaborn-v0_8-whitegrid')

labels = ['Baseline v2\n(Time-Varying)', 'Policy A\n(Radius Cap)', 'Policy B\n(Spatio-Temporal)']


cancel_rates = [7.79, 41.83, 15.70]      
wait_95th = [16.20, 18.98, 20.03]        
utilization = [72.4, 45.7, 89.4]         
profit_hr = [17.53, 14.70, 21.34]        

x = np.arange(len(labels))
width = 0.35


fig, ax1 = plt.subplots(figsize=(8, 5))

color1 = '#e74c3c' 
color2 = '#f39c12' 

rects1 = ax1.bar(x - width/2, cancel_rates, width, label='Cancellation Rate (%)', color=color1, alpha=0.8)
ax1.set_ylabel('Cancellation Rate (%)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold')
ax1.set_ylim(0, 50)
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, wait_95th, width, label='95th Pct Wait (mins)', color=color2, alpha=0.8)
ax2.set_ylabel('95th Percentile Wait (mins)', fontweight='bold')
ax2.set_ylim(0, 25) 


lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('Rider Outcomes: Cancellation vs. Tail Wait\n(Comparison against Baseline v2)', fontweight='bold', pad=15)

for rect in rects1:
    height = rect.get_height()
    ax1.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
for rect in rects2:
    height = rect.get_height()
    ax2.annotate(f'{height}m', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

fig.tight_layout()
plt.savefig('fig_rider_outcomes.png', dpi=300)
plt.close()


fig, ax1 = plt.subplots(figsize=(8, 5))

color3 = '#2980b9' 
color4 = '#27ae60' 

rects3 = ax1.bar(x - width/2, utilization, width, label='Driver Utilization (%)', color=color3, alpha=0.8)
ax1.set_ylabel('Utilization (%)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold')
ax1.set_ylim(0, 100)

ax2 = ax1.twinx()
rects4 = ax2.bar(x + width/2, profit_hr, width, label='Profit per Hour (£)', color=color4, alpha=0.8)
ax2.set_ylabel('Profit per Hour (£)', fontweight='bold')
ax2.set_ylim(0, 25) 

lines3, labels3 = ax1.get_legend_handles_labels()
lines4, labels4 = ax2.get_legend_handles_labels()
ax1.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
plt.title('Driver Outcomes: Utilization vs. Hourly Profit\n(Comparison against Baseline v2)', fontweight='bold', pad=15)

for rect in rects3:
    height = rect.get_height()
    ax1.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
for rect in rects4:
    height = rect.get_height()
    ax2.annotate(f'£{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

fig.tight_layout()
plt.savefig('fig_driver_outcomes.png', dpi=300)
plt.close()

print("✅ 更新后的对比图已生成，数据已与 Table 8 同步。")
