import pandas as pd
from pathlib import Path

DATA_ONE = Path("data/one_case.csv")
DATA_EVENTS = Path("data/v2_steps14_q2_2025/events.csv")
OUT = Path("data/one_case_bottleneck.csv")

EXTRA_MINUTES = 600          # tăng thêm 10 tiếng
TARGET_KEYWORD = "LOADING"   # ưu tiên làm nghẽn LOADING (rõ nhất)

def main():
    # 1) Load 1-case file nếu có, nếu không thì tự cắt từ events.csv
    if DATA_ONE.exists():
        df = pd.read_csv(DATA_ONE)
        print("Loaded:", DATA_ONE)
    else:
        df_all = pd.read_csv(DATA_EVENTS)
        df_all = df_all[df_all["process_code"] == "TRUCKING_DELIVERY_FLOW"].copy()
        cid = df_all["case_id"].iloc[0]
        df = df_all[df_all["case_id"] == cid].copy()
        df.to_csv(DATA_ONE, index=False)
        print("Created:", DATA_ONE, "case_id =", cid)

    # 2) Parse time
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    if df["start_time"].isna().any() or df["end_time"].isna().any():
        raise ValueError("Bad timestamps in file")

    # 3) Pick target step (LOADING)
    mask = df["step_code"].astype(str).str.contains(TARGET_KEYWORD, case=False, na=False)
    if not mask.any():
        # fallback: nếu không có LOADING, lấy step có duration lớn nhất
        df["dur_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
        idx = df["dur_min"].idxmax()
        target_step = df.loc[idx, "step_code"]
    else:
        idx = df[mask].index[0]
        target_step = df.loc[idx, "step_code"]

    old_start = df.loc[idx, "start_time"]
    old_end = df.loc[idx, "end_time"]
    old_dur = (old_end - old_start).total_seconds() / 60

    delta = pd.Timedelta(minutes=EXTRA_MINUTES)

    # 4) Extend target step end_time
    df.loc[idx, "end_time"] = old_end + delta

    # 5) Shift subsequent steps to keep timeline consistent
    later = df["start_time"] >= old_end
    df.loc[later, "start_time"] = df.loc[later, "start_time"] + delta
    df.loc[later, "end_time"] = df.loc[later, "end_time"] + delta

    new_end = df.loc[idx, "end_time"]
    new_dur = (new_end - old_start).total_seconds() / 60

    # 6) Format ISO (no ms)
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["end_time"] = df["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print("Saved bottleneck case:", OUT)
    print("Target step:", target_step)
    print(f"Duration min: {old_dur:.2f} -> {new_dur:.2f} (added {EXTRA_MINUTES} min)")
    print("case_id:", df['case_id'].iloc[0], "process_code:", df['process_code'].iloc[0])
    print("rows:", len(df))

if __name__ == "__main__":
    main()