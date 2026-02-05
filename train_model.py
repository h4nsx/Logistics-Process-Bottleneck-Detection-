import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


# 1. Giả lập dữ liệu (Lúc thi ông quăng tui file CSV thật nhé)
def load_data():
    # Ví dụ các feature: Khoảng cách, Cân nặng, Loại xe, Thời tiết, Giờ khởi hành...
    data = pd.DataFrame({
        'distance_km': np.random.randint(10, 500, 1000),
        'weight_kg': np.random.randint(1, 100, 1000),
        'vehicle_type': np.random.choice(['truck', 'van', 'bike'], 1000),
        'weather_condition': np.random.choice(['sunny', 'rainy', 'foggy'], 1000),
        'is_delayed': np.random.choice([0, 1], 1000)  # Target: 0 đúng hạn, 1 trễ
    })
    return data


# 2. Data Processing & Feature Engineering Pipeline
def build_pipeline(model_type='xgboost'):
    numeric_features = ['distance_km', 'weight_kg']
    categorical_features = ['vehicle_type', 'weather_condition']

    # Xử lý số: Chuẩn hóa
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Xử lý chữ: One-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Chọn thuật toán
    if model_type == 'xgboost':
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    elif model_type == 'rf':
        clf = RandomForestClassifier()
    else:
        # Logistic Regression làm baseline
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()

    # Ghép lại thành 1 ống dẫn hoàn chỉnh
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', clf)])
    return pipeline


# 3. Training & Anomaly Detection
def main():
    df = load_data()

    # --- Module A: Đánh giá rủi ro trễ hàng (Supervised Learning) ---
    X = df.drop('is_delayed', axis=1)
    y = df['is_delayed']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline('xgboost')

    # Hyperparameter Tuning (GridSearchCV)
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1]
    }

    search = GridSearchCV(pipeline, param_grid, cv=3)
    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    print("Model Score:", search.score(X_test, y_test))

    # --- Module B: Phát hiện vận đơn bất thường (Unsupervised Learning) ---
    # Dùng Isolation Forest để tìm ra các dòng dữ liệu "kỳ quái" trong quy trình
    # Ví dụ: Xe máy nhưng chở 500kg -> Bất thường
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    # Cần preprocess dữ liệu trước khi đưa vào Isolation Forest (tái sử dụng preprocessor)
    preprocessor = search.best_estimator_.named_steps['preprocessor']
    X_processed = preprocessor.transform(X)
    iso_forest.fit(X_processed)

    # 4. Serialization (Lưu model)
    # Lưu cả model dự đoán và model bắt lỗi
    artifacts = {
        'classifier_pipeline': search.best_estimator_,
        'anomaly_detector': iso_forest,
        'preprocessor': preprocessor  # Lưu cái này để transform input cho anomaly detector
    }
    joblib.dump(artifacts, 'logistics_ai_brain.pkl')
    print("Đã lưu model vào 'logistics_ai_brain.pkl' ✅")


if __name__ == "__main__":
    main()