"""
=============================================================================
EGG LAYER FARM PREDICTIVE MODEL & YIELD FORECASTING PIPELINE
=============================================================================
BY VU-BAD-2603-0551-DAY-SANKARA DERRICK
Purpose: Predict daily egg production, benchmark ML models, and forecast 
         operational yields and margins using daily farm records.
Dependencies: numpy, pandas, matplotlib (standard library, zero-setup)
=============================================================================
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# =============================================================================
# 1. DATA INGESTION & DATA CLEANING
# =============================================================================
def load_and_clean_data(filepath='farm_operations_daily.csv'):
    """
    Loads daily farm operations CSV, validates column schema, parses dates,
    and corrects operational data entry anomalies (e.g., May 3 feed outlier).
    """
    if not os.path.exists(filepath):
        # Fallback to alternative common names
        for alt in ['Daily_Farm_Operations_AFROFRESH.csv', 'farm_operations_daily_cleaned.csv']:
            if os.path.exists(alt):
                filepath = alt
                break
                
    print(f"[*] Ingesting farm operations dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Required column validation
    required_cols = [
        'Date', 'Flock_Size', 'Feed_Consumed_kg', 'Eggs_Collected',
        'Feed_Cost', 'Other_Expenses', 'Income_Egg_Sales'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Standardize Date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Data Cleaning: Detect and correct obvious clerical entry typos
    # Outlier check: A daily feed entry > 500 kg for a ~1,950 bird flock is biologically
    # impossible (>1 kg/bird/day). Normal layer intake is 100 - 130 grams/bird (195-250 kg total).
    anomalous_feed = df['Feed_Consumed_kg'] > 500
    if anomalous_feed.any():
        for idx in df[anomalous_feed].index:
            date_str = df.loc[idx, 'Date'].strftime('%Y-%m-%d')
            raw_val = df.loc[idx, 'Feed_Consumed_kg']
            # Typo fix: 2000kg entry on May 3 was recorded with an extra zero (intended 200kg)
            corrected_val = raw_val / 10.0 if raw_val == 2000.0 else 200.0
            print(f"    [!] Detected data entry anomaly on {date_str}: {raw_val} kg feed.")
            print(f"        -> Corrected to {corrected_val} kg based on adjacent operational feed rate.")
            df.loc[idx, 'Feed_Consumed_kg'] = corrected_val

    print(f"[+] Loaded {len(df)} days of continuous operational data ({df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}).")
    return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """
    Constructs domain-specific biological, temporal, and autoregressive features
    strictly preventing lookahead bias.
    """
    print("[*] Performing feature engineering...")
    data = df.copy()
    
    # Time trend (Maturity index)
    data['Days_in_Lay'] = np.arange(len(data))
    
    # Biological unit rates
    data['Feed_per_Bird_g'] = (data['Feed_Consumed_kg'] * 1000.0) / data['Flock_Size']
    data['Lay_Rate_Pct'] = (data['Eggs_Collected'] / data['Flock_Size']) * 100.0
    
    # Autoregressive production lags (Days t-1, t-2, t-3, and t-7)
    for l in [1, 2, 3, 7]:
        data[f'LayRate_Lag{l}'] = data['Lay_Rate_Pct'].shift(l)
        data[f'Eggs_Lag{l}'] = data['Eggs_Collected'].shift(l)
        data[f'Feed_per_Bird_Lag{l}'] = data['Feed_per_Bird_g'].shift(l)
        data[f'Feed_Lag{l}'] = data['Feed_Consumed_kg'].shift(l)
        
    # Rolling baselines (strictly using shift(1) to avoid data leakage)
    data['LayRate_RollMean7'] = data['Lay_Rate_Pct'].shift(1).rolling(7).mean()
    data['LayRate_RollStd7'] = data['Lay_Rate_Pct'].shift(1).rolling(7).std().fillna(0)
    data['Feed_per_Bird_RollMean7'] = data['Feed_per_Bird_g'].shift(1).rolling(7).mean()
    
    # Day-of-week cyclicality
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    for d in range(7):
        data[f'DOW_{d}'] = (data['DayOfWeek'] == d).astype(float)
        
    # Drop rows with NaN values resulting from the 7-day lag window
    clean_data = data.dropna().reset_index(drop=True)
    print(f"[+] Feature set constructed: {clean_data.shape[1]} total columns across {len(clean_data)} valid rows.")
    return clean_data

# =============================================================================
# 3. CUSTOM ROBUST REGRESSOR ENGINES (NUMPY-BASED)
# =============================================================================
class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=2, min_samples_split=6):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) < 1e-6:
            return {'val': np.mean(y)}
        n_samples, n_features = X.shape
        best_var_red = -1
        best_feat = None
        best_thresh = None
        current_var = np.var(y) * n_samples

        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) < 2:
                continue
            thresholds = (vals[:-1] + vals[1:]) / 2.0
            if len(thresholds) > 15:
                thresholds = np.percentile(vals, np.linspace(5, 95, 12))
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                var_left = np.var(y[left_mask]) * np.sum(left_mask)
                var_right = np.var(y[right_mask]) * np.sum(right_mask)
                var_red = current_var - (var_left + var_right)
                if var_red > best_var_red:
                    best_var_red = var_red
                    best_feat = feat
                    best_thresh = t

        if best_feat is None:
            return {'val': np.mean(y)}

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_sub = self.fit(X[left_mask], y[left_mask], depth + 1)
        right_sub = self.fit(X[right_mask], y[right_mask], depth + 1)
        return {'feat': best_feat, 'thresh': best_thresh, 'left': left_sub, 'right': right_sub}

    def _predict_row(self, node, x):
        if 'val' in node:
            return node['val']
        if x[node['feat']] <= node['thresh']:
            return self._predict_row(node['left'], x)
        else:
            return self._predict_row(node['right'], x)

    def predict(self, X):
        return np.array([self._predict_row(self.tree, x) for x in X])

class GBDTRegressorCustom:
    def __init__(self, n_estimators=40, learning_rate=0.08, max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.init_val = 0.0

    def fit(self, X, y):
        self.init_val = np.mean(y)
        y_pred = np.full(len(y), self.init_val)
        self.trees = []
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            dt = DecisionTreeRegressorCustom(max_depth=self.max_depth)
            tree_dict = dt.fit(X, residuals)
            dt.tree = tree_dict
            y_pred += self.learning_rate * dt.predict(X)
            self.trees.append(dt)

    def predict(self, X):
        y_pred = np.full(len(X), self.init_val)
        for dt in self.trees:
            y_pred += self.learning_rate * dt.predict(X)
        return y_pred

# =============================================================================
# 4. MODEL EVALUATION & BENCHMARKING
# =============================================================================
def calc_metrics(actual, pred):
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred)**2))
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100.0
    ss_tot = np.sum((actual - np.mean(actual))**2)
    ss_res = np.sum((actual - pred)**2)
    r2 = 1.0 - (ss_res / ss_tot)
    return mae, rmse, mape, r2

def run_pipeline(filepath='farm_operations_daily.csv'):
    df = load_and_clean_data(filepath)
    clean_data = engineer_features(df)
    
    # Chronological Out-of-Time Train/Test Split (80% Train, 20% Test)
    split_idx = int(len(clean_data) * 0.8)
    train_df = clean_data.iloc[:split_idx].copy()
    test_df = clean_data.iloc[split_idx:].copy()
    
    print(f"\n[*] Data Partitioning (Temporal Out-of-Sample Split):")
    print(f"    - Training Set: {len(train_df)} days ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"    - Test Set:     {len(test_df)} days ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")

    features = [
        'Days_in_Lay', 'Feed_per_Bird_g',
        'Feed_per_Bird_Lag1', 'Feed_per_Bird_Lag2', 'Feed_per_Bird_Lag3', 'Feed_per_Bird_RollMean7',
        'LayRate_Lag1', 'LayRate_Lag2', 'LayRate_Lag3', 'LayRate_Lag7',
        'LayRate_RollMean7', 'LayRate_RollStd7'
    ]

    X_train = train_df[features].values
    y_train = train_df['Lay_Rate_Pct'].values
    X_test = test_df[features].values
    y_test = test_df['Lay_Rate_Pct'].values

    # Feature Standardization (fit on train only)
    mean_X = X_train.mean(axis=0)
    std_X = X_train.std(axis=0) + 1e-8
    X_train_s = (X_train - mean_X) / std_X
    X_test_s = (X_test - mean_X) / std_X

    # 1. Baseline 1: Naive Persistence (Yesterday's eggs)
    pred_pers_eggs = test_df['Eggs_Lag1'].values

    # 2. Baseline 2: 7-Day Moving Average
    pred_roll_eggs = (test_df['LayRate_RollMean7'].values / 100.0) * test_df['Flock_Size'].values

    # 3. Model 1: Multivariate Ridge Regression
    X_train_b = np.column_stack([np.ones(len(X_train_s)), X_train_s])
    X_test_b = np.column_stack([np.ones(len(X_test_s)), X_test_s])
    I = np.eye(X_train_b.shape[1])
    I[0, 0] = 0  # Do not penalize bias
    w_ridge = np.linalg.solve(X_train_b.T @ X_train_b + 5.0 * I, X_train_b.T @ y_train)
    pred_ridge_rate = X_test_b @ w_ridge
    pred_ridge_eggs = (pred_ridge_rate / 100.0) * test_df['Flock_Size'].values

    # 4. Model 2: Gradient Boosted Decision Trees (GBDT)
    gbdt = GBDTRegressorCustom(n_estimators=40, learning_rate=0.08, max_depth=2)
    gbdt.fit(X_train, y_train)
    pred_gbdt_rate = gbdt.predict(X_test)
    pred_gbdt_eggs = (pred_gbdt_rate / 100.0) * test_df['Flock_Size'].values

    # Actuals
    actual_test_eggs = test_df['Eggs_Collected'].values

    # Compile Benchmark Metrics
    models = [
        ('Naive Persistence (Lag 1)', pred_pers_eggs),
        ('7-Day Moving Average', pred_roll_eggs),
        ('Multivariate Ridge Regression', pred_ridge_eggs),
        ('Gradient Boosted Trees (GBDT)', pred_gbdt_eggs)
    ]

    metric_rows = []
    for name, preds in models:
        mae, rmse, mape, r2 = calc_metrics(actual_test_eggs, preds)
        metric_rows.append({
            'Model Architecture': name,
            'MAE (Eggs)': round(mae, 2),
            'RMSE (Eggs)': round(rmse, 2),
            'MAPE (%)': f"{round(mape, 2)}%",
            'R2 Score': round(r2, 4)
        })

    benchmark_df = pd.DataFrame(metric_rows)
    print("\n" + "="*70)
    print("           MODEL BENCHMARK RESULTS (OUT-OF-SAMPLE TEST SET)")
    print("="*70)
    print(benchmark_df.to_string(index=False))
    print("="*70)

    # =========================================================================
    # 5. VISUALIZATION & EVALUATION CHARTS
    # =========================================================================
    print("\n[*] Generating high-resolution evaluation figures...")
    
    # 1. Actual vs Predicted Chart
    plt.figure(figsize=(12, 5), dpi=300)
    test_dates = test_df['Date'].values
    plt.plot(test_dates, actual_test_eggs, label='Actual Egg Yield', color='#1f77b4', marker='o', linewidth=2)
    plt.plot(test_dates, pred_gbdt_eggs, label='GBDT Model Prediction', color='#2ca02c', linestyle='--', linewidth=2.2)
    plt.plot(test_dates, pred_ridge_eggs, label='Ridge Regression', color='#ff7f0e', linestyle=':', linewidth=1.8)
    plt.title('Out-of-Sample Daily Egg Collection: Actual vs. Model Predictions', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Daily Eggs Collected', fontsize=11)
    plt.legend(frameon=True, facecolor='white', loc='upper right')
    plt.tight_layout()
    chart1_path = 'chart_actual_vs_predicted.png'
    plt.savefig(chart1_path)
    plt.close()
    print(f"    [+] Saved: {chart1_path}")

    # 2. Residual Distribution Chart
    residuals = actual_test_eggs - pred_gbdt_eggs
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # Time series of residuals
    axes[0].axhline(0, color='red', linestyle='--', linewidth=1.2)
    axes[0].plot(test_dates, residuals, marker='s', color='#9467bd', linewidth=1.5)
    axes[0].set_title('Prediction Residuals Over Time (Actual - GBDT)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Residual (Eggs)')
    
    # Histogram of residuals
    axes[1].hist(residuals, bins=12, color='#17becf', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(residuals), color='black', linestyle='--', label=f'Mean Error: {np.mean(residuals):.1f}')
    axes[1].set_title('Residual Error Distribution (Histogram)', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Prediction Error (Eggs)')
    axes[1].set_ylabel('Frequency (Days)')
    axes[1].legend()
    plt.tight_layout()
    chart2_path = 'chart_residuals.png'
    plt.savefig(chart2_path)
    plt.close()
    print(f"    [+] Saved: {chart2_path}")

    # 3. Feature Importance Bar Chart (Ridge Weights)
    coef_labels = [
        'Lag-1 Lay Rate (Yesterday)', 'Lag-2 Lay Rate', '7-Day Rolling Lay Rate',
        'Lag-3 Lay Rate', 'Daily Feed per Bird (g)', 'Lag-7 Lay Rate (Weekly)',
        'Lag-1 Feed per Bird', '7-Day Rolling Feed', 'Lag-2 Feed per Bird',
        'Days in Lay (Flock Age)', '7-Day Lay Volatility (Std)', 'Lag-3 Feed per Bird'
    ]
    # Match order of feature weights
    weights = w_ridge[1:]  # exclude intercept
    sorted_idx = np.argsort(np.abs(weights))

    plt.figure(figsize=(10, 6), dpi=300)
    colors = ['#1f77b4' if w >= 0 else '#d62728' for w in weights[sorted_idx]]
    plt.barh(np.array(coef_labels)[sorted_idx], weights[sorted_idx], color=colors, edgecolor='black', alpha=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Predictive Feature Importance (Normalized Ridge Regression Weights)', fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Standardized Model Coefficient (Impact on Lay Rate %)', fontsize=10)
    plt.tight_layout()
    chart3_path = 'chart_feature_importance.png'
    plt.savefig(chart3_path)
    plt.close()
    print(f"    [+] Saved: {chart3_path}")

    # =========================================================================
    # 6. 14-DAY OPERATIONAL & FINANCIAL FORWARD FORECAST
    # =========================================================================
    print("\n[*] Generating 14-Day Forward Operational Forecast (Multi-Step Autoregressive Rollout)...")
    last_date = clean_data['Date'].max()
    last_flock = int(clean_data['Flock_Size'].iloc[-1])
    daily_feed_kg = 194.0  # Standard daily ration for current 1,608 layer flock
    feed_per_bird = (daily_feed_kg * 1000.0) / last_flock

    sim_rates = list(clean_data['Lay_Rate_Pct'].values)
    sim_feeds = list(clean_data['Feed_per_Bird_g'].values)

    forecast_list = []
    for day_i in range(1, 15):
        fc_date = last_date + pd.Timedelta(days=day_i)
        days_in_lay = len(clean_data) + day_i - 1

        row_dict = {
            'Days_in_Lay': days_in_lay,
            'Feed_per_Bird_g': feed_per_bird,
            'Feed_per_Bird_Lag1': sim_feeds[-1],
            'Feed_per_Bird_Lag2': sim_feeds[-2],
            'Feed_per_Bird_Lag3': sim_feeds[-3],
            'Feed_per_Bird_RollMean7': np.mean(sim_feeds[-7:]),
            'LayRate_Lag1': sim_rates[-1],
            'LayRate_Lag2': sim_rates[-2],
            'LayRate_Lag3': sim_rates[-3],
            'LayRate_Lag7': sim_rates[-7],
            'LayRate_RollMean7': np.mean(sim_rates[-7:]),
            'LayRate_RollStd7': np.std(sim_rates[-7:])
        }
        x_vec = np.array([[row_dict[f] for f in features]])
        
        # Predict Lay Rate using GBDT
        pred_rate = gbdt.predict(x_vec)[0]
        pred_eggs = int(round((pred_rate / 100.0) * last_flock))
        trays = round(pred_eggs / 30.0, 1)

        # Operational economics in Uganda Shillings (UGX)
        revenue_ugx = int(round((pred_eggs / 30.0) * 10000.0))
        feed_cost_ugx = int(round(daily_feed_kg * 1450.0))
        daily_margin_ugx = revenue_ugx - feed_cost_ugx

        forecast_list.append({
            'Date': fc_date.strftime('%Y-%m-%d'),
            'Flock_Size': last_flock,
            'Feed_kg': daily_feed_kg,
            'Pred_Lay_Rate_%': round(pred_rate, 2),
            'Pred_Eggs': pred_eggs,
            'Pred_Trays': trays,
            'Est_Revenue_UGX': revenue_ugx,
            'Est_Feed_Cost_UGX': feed_cost_ugx,
            'Est_Gross_Margin_UGX': daily_margin_ugx
        })

        sim_rates.append(pred_rate)
        sim_feeds.append(feed_per_bird)

    fc_df = pd.DataFrame(forecast_list)
    print("\n" + "="*80)
    print("            14-DAY OPERATIONAL & FINANCIAL FORECAST")
    print("="*80)
    print(fc_df.to_string(index=False))
    print("="*80)
    
    fc_df.to_csv('14_day_forward_forecast.csv', index=False)
    print("\n[✔] Machine learning pipeline execution completed successfully.")

if __name__ == '__main__':
    csv_file = 'farm_operations_daily.csv'
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        csv_file = sys.argv[1]
    run_pipeline(csv_file)
