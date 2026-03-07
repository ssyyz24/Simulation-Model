import pandas as pd
import numpy as np
import re
from pathlib import Path

RIDERS_IN = Path("riders.csv")
DRIVERS_IN = Path("drivers.csv")

RIDERS_OUT = Path("riders_clean.csv")
DRIVERS_OUT = Path("drivers_clean.csv")
REPORT_OUT = Path("data_quality_report.csv")


POINT_RE = re.compile(
    r"^\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)\s*$"
)

def parse_point(s):
    if pd.isna(s):
        return (np.nan, np.nan)
    m = POINT_RE.match(str(s))
    if not m:
        return (np.nan, np.nan)
    return (float(m.group(1)), float(m.group(2)))

def in_square(x: pd.Series) -> pd.Series:
    return x.between(0, 20)



def clean_riders(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()


    for c in ["pickup_time", "dropoff_time"]:
        if c in r.columns:
            r.loc[r[c] == -1, c] = np.nan


    for c in ["request_datetime", "pickup_datetime", "dropoff_datetime"]:
        if c in r.columns:
            r[c] = pd.to_datetime(r[c], errors="coerce")


    if "pickup_location" in r.columns:
        r[["pickup_x", "pickup_y"]] = r["pickup_location"].apply(
            lambda s: pd.Series(parse_point(s))
        )
    if "dropoff_location" in r.columns:
        r[["dropoff_x", "dropoff_y"]] = r["dropoff_location"].apply(
            lambda s: pd.Series(parse_point(s))
        )


    status_map = {
        "dropped-off": "completed",
        "abandoned": "abandoned",
        "pickup-scheduled": "in_progress_pickup",
        "dropoff-scheduled": "in_progress_dropoff",
    }
    if "status" in r.columns:
        r["status_group"] = r["status"].map(status_map).fillna("other")
    else:
        r["status_group"] = "unknown"

    if {"request_time", "pickup_time"}.issubset(r.columns):
        r["wait_hours"] = r["pickup_time"] - r["request_time"]
    else:
        r["wait_hours"] = np.nan

    if {"pickup_time", "dropoff_time"}.issubset(r.columns):
        r["trip_hours"] = r["dropoff_time"] - r["pickup_time"]
    else:
        r["trip_hours"] = np.nan


    if {"pickup_x", "pickup_y"}.issubset(r.columns):
        r["bad_coord_pickup"] = ~(
            in_square(r["pickup_x"]) & in_square(r["pickup_y"])
        )
    else:
        r["bad_coord_pickup"] = False

    if {"dropoff_x", "dropoff_y"}.issubset(r.columns):
        r["bad_coord_dropoff"] = ~(
            in_square(r["dropoff_x"]) & in_square(r["dropoff_y"])
        )
    else:
        r["bad_coord_dropoff"] = False

    r["bad_time_order"] = False
    if "wait_hours" in r.columns:
        r.loc[r["wait_hours"].notna(), "bad_time_order"] |= (r["wait_hours"] < 0)
    if "trip_hours" in r.columns:
        r.loc[r["trip_hours"].notna(), "bad_time_order"] |= (r["trip_hours"] < 0)

    return r



def clean_drivers(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()


    for c in ["arrival_datetime", "offline_datetime"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")


    if "initial_location" in d.columns:
        d[["initial_x", "initial_y"]] = d["initial_location"].apply(
            lambda s: pd.Series(parse_point(s))
        )
    if "current_location" in d.columns:
        d[["current_x", "current_y"]] = d["current_location"].apply(
            lambda s: pd.Series(parse_point(s))
        )


    if {"arrival_time", "offline_time"}.issubset(d.columns):
        d["online_hours"] = d["offline_time"] - d["arrival_time"]
    else:
        d["online_hours"] = np.nan


    if {"initial_x", "initial_y"}.issubset(d.columns):
        d["bad_coord_initial"] = ~(
            in_square(d["initial_x"]) & in_square(d["initial_y"])
        )
    else:
        d["bad_coord_initial"] = False

    if {"current_x", "current_y"}.issubset(d.columns):
        d["bad_coord_current"] = ~(
            in_square(d["current_x"]) & in_square(d["current_y"])
        )
    else:
        d["bad_coord_current"] = False

    d["bad_time_order"] = False
    if "online_hours" in d.columns:
        d.loc[d["online_hours"].notna(), "bad_time_order"] |= (d["online_hours"] < 0)

    return d



def make_report(r_clean: pd.DataFrame, d_clean: pd.DataFrame) -> pd.DataFrame:
    report = {
        "riders_rows": len(r_clean),
        "drivers_rows": len(d_clean),
        "riders_status_counts": r_clean["status"].value_counts().to_dict() if "status" in r_clean.columns else {},
        "riders_missing_pickup_time": int(r_clean["pickup_time"].isna().sum()) if "pickup_time" in r_clean.columns else None,
        "riders_missing_dropoff_time": int(r_clean["dropoff_time"].isna().sum()) if "dropoff_time" in r_clean.columns else None,
        "riders_bad_coords_pickup": int(r_clean.get("bad_coord_pickup", pd.Series(False)).sum()),
        "riders_bad_coords_dropoff": int(r_clean.get("bad_coord_dropoff", pd.Series(False)).sum()),
        "riders_bad_time_order": int(r_clean.get("bad_time_order", pd.Series(False)).sum()),
        "drivers_status_counts": d_clean["status"].value_counts().to_dict() if "status" in d_clean.columns else {},
        "drivers_bad_coords_initial": int(d_clean.get("bad_coord_initial", pd.Series(False)).sum()),
        "drivers_bad_coords_current": int(d_clean.get("bad_coord_current", pd.Series(False)).sum()),
        "drivers_bad_time_order": int(d_clean.get("bad_time_order", pd.Series(False)).sum()),
    }
    return pd.DataFrame([{"metric": k, "value": v} for k, v in report.items()])



def main():
    riders = pd.read_csv(RIDERS_IN)
    drivers = pd.read_csv(DRIVERS_IN)

    riders_clean = clean_riders(riders)
    drivers_clean = clean_drivers(drivers)

    report_df = make_report(riders_clean, drivers_clean)

    riders_clean.to_csv(RIDERS_OUT, index=False)
    drivers_clean.to_csv(DRIVERS_OUT, index=False)
    report_df.to_csv(REPORT_OUT, index=False)

    print("Done.")
    print(f"  -> {RIDERS_OUT.resolve()}")
    print(f"  -> {DRIVERS_OUT.resolve()}")
    print(f"  -> {REPORT_OUT.resolve()}")
    print("\nReport preview:")
    print(report_df.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
