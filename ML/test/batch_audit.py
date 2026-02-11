import pandas as pd
import requests
import json
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(CURRENT_DIR, "business_input.csv")
OUTPUT_FILE = os.path.join(CURRENT_DIR, "audit_report_final.csv")

# 👇 SỬA LẠI ĐƯỜNG DẪN API CHO ĐÚNG VỚI APP.PY MỚI
API_URL = "http://127.0.0.1:8000/analyze_shipment"


def analyze_business_process():
    print(f"📂 Đang đọc dữ liệu doanh nghiệp từ: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"   -> Tìm thấy {len(df)} hồ sơ vận chuyển.")
    except Exception as e:
        print(f"❌ Lỗi đọc file: {e}")
        return

    print("\n🤖 Đang tiến hành Thanh tra Toàn diện (AI Scanning)...")

    results = []
    stats = {
        "total": 0, "high_risk": 0, "low_risk": 0,
        "bottlenecks": {"driver": 0, "fleet": 0, "ops": 0}
    }

    for index, row in df.iterrows():
        # Chuẩn bị dữ liệu (Mapping đúng tên cột trong CSV)
        payload = {
            "case_id": str(row['trip_id']),  # API mới cần case_id
            "years_experience": row['years_experience'],
            "total_accidents": row['total_accidents'],
            "avg_ontime_rate": row['avg_ontime_rate'],
            "avg_miles_per_month": row['avg_miles_per_month'],
            "avg_mpg_driver": row['avg_mpg_driver'],

            "truck_age": row['truck_age'],
            "lifetime_maint_cost": row['lifetime_maint_cost'],
            "maint_frequency": row['maint_frequency'],
            "total_downtime": row['total_downtime'],
            "avg_monthly_miles_truck": row['avg_monthly_miles_truck'],

            "detention_hours": row['detention_hours'],
            "real_mpg_trip": row['real_mpg_trip'],
            "delay_hours": row['delay_hours'],
            "actual_distance_miles": row['actual_distance_miles']
        }

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                data = response.json()

                # --- ĐỌC KẾT QUẢ THEO FORMAT MỚI CỦA LEADER ---
                analysis = data['analysis']
                explain = data['explainability']
                rec = data['recommendation']

                # Thống kê
                stats["total"] += 1
                if analysis['is_anomaly']:
                    stats["high_risk"] += 1
                else:
                    stats["low_risk"] += 1

                # Tìm nguyên nhân chính (Contributors)
                if explain['primary_contributors']:
                    top_reason = explain['primary_contributors'][0]['step_code']
                    if 'DRIVER' in top_reason:
                        stats['bottlenecks']['driver'] += 1
                    elif 'FLEET' in top_reason:
                        stats['bottlenecks']['fleet'] += 1
                    elif 'OPERATIONS' in top_reason:
                        stats['bottlenecks']['ops'] += 1

                # Lưu kết quả
                row['Risk_Score'] = analysis['risk_score']
                row['Is_Anomaly'] = analysis['is_anomaly']
                row['Action'] = rec['action']
                results.append(row)
            else:
                print(f"⚠️ Lỗi API dòng {index}: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"⚠️ Lỗi dòng {index}: {e}")

    # --- XUẤT BÁO CÁO ---
    print("\n" + "=" * 60)
    print("📊 BÁO CÁO QUẢN TRỊ (EXECUTIVE SUMMARY)")
    print("=" * 60)

    total = stats['total']
    if total == 0: return

    print(f"1️⃣  TỔNG QUAN:")
    print(f"   - Tổng số chuyến: {total}")
    print(f"   - ✅ An toàn (Low Risk): {stats['low_risk']}")
    print(f"   - ⛔ Rủi ro cao (High Risk): {stats['high_risk']}")

    print(f"\n2️⃣  PHÂN TÍCH ĐIỂM NGHẼN (BOTTLENECK):")
    print(f"   👨‍✈️ Tài xế: {stats['bottlenecks']['driver']} vấn đề")
    print(f"   🚛 Xe cộ:   {stats['bottlenecks']['fleet']} vấn đề")
    print(f"   ⏱️ Vận hành: {stats['bottlenecks']['ops']} vấn đề")

    # Xuất file CSV
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Đã lưu chi tiết vào: {OUTPUT_FILE}")


if __name__ == "__main__":
    analyze_business_process()