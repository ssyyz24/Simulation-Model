import pandas as pd
import numpy as np

# ===============================
# 1. 读取清理后的数据
# ===============================

riders = pd.read_csv("riders_clean.csv")
drivers = pd.read_csv("drivers_clean.csv")

# ===============================
# 2. 计算 Arrival Rates
# ===============================

# 时间跨度（小时）
r_time_span = riders["request_time"].max() - riders["request_time"].min()
d_time_span = drivers["arrival_time"].max() - drivers["arrival_time"].min()

# Poisson MLE
lambda_r = len(riders) / r_time_span
lambda_d = len(drivers) / d_time_span


# ===============================
# 3. Trip duration statistics
# ===============================

trip = riders["trip_hours"].dropna()

trip_mean = trip.mean()
trip_median = trip.median()
trip_p95 = trip.quantile(0.95)


# ===============================
# 4. Driver online duration
# ===============================

online = drivers["online_hours"].dropna()

online_mean = online.mean()
online_median = online.median()
online_p95 = online.quantile(0.95)


# ===============================
# 5. Cancellation rate
# ===============================

cancel_rate = (riders["status"] == "abandoned").mean()


# ===============================
# 6. 生成 LaTeX Table
# ===============================

latex = f"""
\\begin{{table}}[H]
\\centering
\\caption{{Final calibrated simulation inputs}}
\\label{{tab:calibration}}

\\begin{{tabular}}{{llll}}
\\hline
Parameter & Distribution / Value & Units & Data source \\\\
\\hline

Rider arrival rate $\\lambda_r$
& {lambda_r:.2f}
& requests/hour
& request\\_time
\\\\

Driver arrival rate $\\lambda_d$
& {lambda_d:.2f}
& drivers/hour
& arrival\\_time
\\\\

Trip duration
& mean={trip_mean:.2f}, median={trip_median:.2f}, p95={trip_p95:.2f}
& hours
& trip\\_hours
\\\\

Driver online duration
& mean={online_mean:.2f}, median={online_median:.2f}, p95={online_p95:.2f}
& hours
& online\\_hours
\\\\

Cancellation rate
& {cancel_rate:.3f}
& probability
& rider status
\\\\

Spatial distribution
& Empirical
& coordinates
& pickup/dropoff locations
\\\\

\\hline
\\end{{tabular}}

\\end{{table}}
"""


# ===============================
# 7. 保存文件
# ===============================

with open("calibration_table.tex", "w") as f:

    f.write(latex)


# ===============================
# 8. 打印结果
# ===============================

print("Calibration table generated successfully.\n")

print(latex)