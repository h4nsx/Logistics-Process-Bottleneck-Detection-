from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # D:\logistics_AI
DATA_EVENTS = PROJECT_ROOT / "data" / "v2_steps14_q2_2025" / "events.csv"
OUT_DIR = PROJECT_ROOT / "data" / "test_cases"

print("DATA_EVENTS =", DATA_EVENTS)
print("EXISTS =", DATA_EVENTS.exists())

def extract_one_case(process_code: str, out_csv: Path) -> str:
    df = pd.read_csv(DATA_EVENTS)
    df = df[df["process_code"] == process_code].copy()
    if df.empty:
        raise ValueError(f"No rows for process_code={process_code} in {DATA_EVENTS}")

    cid = str(df["case_id"].iloc[0])
    one = df[df["case_id"].astype(str) == cid].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    one.to_csv(out_csv, index=False)
    return cid

def make_bottleneck(in_csv: Path, out_csv: Path, keyword: str, extra_minutes: int = 600):
    df = pd.read_csv(in_csv)

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    if df["start_time"].isna().any() or df["end_time"].isna().any():
        raise ValueError(f"Bad timestamps in {in_csv}")

    # pick target step by keyword
    mask = df["step_code"].astype(str).str.contains(keyword, case=False, na=False)
    if mask.any():
        idx = df[mask].index[0]
    else:
        # fallback: largest duration
        df["dur_min"] = (df["end_time"] - df["start_time"]).dt.total_seconds() / 60
        idx = df["dur_min"].idxmax()

    old_end = df.loc[idx, "end_time"]
    delta = pd.Timedelta(minutes=extra_minutes)

    # extend that step end_time
    df.loc[idx, "end_time"] = df.loc[idx, "end_time"] + delta

    # shift all later steps to keep timeline consistent
    later = df["start_time"] >= old_end
    df.loc[later, "start_time"] = df.loc[later, "start_time"] + delta
    df.loc[later, "end_time"] = df.loc[later, "end_time"] + delta

    # format no-ms
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["end_time"] = df["end_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # WAREHOUSE
    wh_one = OUT_DIR / "warehouse_one_case.csv"
    wh_bot = OUT_DIR / "warehouse_one_case_bottleneck.csv"
    wh_cid = extract_one_case("WAREHOUSE_FULFILLMENT", wh_one)
    make_bottleneck(wh_one, wh_bot, keyword="PICK", extra_minutes=600)  # nghẽn picking
    print("WAREHOUSE one_case:", wh_one, "case_id=", wh_cid)
    print("WAREHOUSE bottleneck:", wh_bot)

    # CUSTOMS
    cu_one = OUT_DIR / "customs_one_case.csv"
    cu_bot = OUT_DIR / "customs_one_case_bottleneck.csv"
    cu_cid = extract_one_case("IMPORT_CUSTOMS_CLEARANCE", cu_one)
    make_bottleneck(cu_one, cu_bot, keyword="INSPECTION", extra_minutes=1200)  # nghẽn inspection (thường lâu)
    print("CUSTOMS one_case:", cu_one, "case_id=", cu_cid)
    print("CUSTOMS bottleneck:", cu_bot)

if __name__ == "__main__":
    main()