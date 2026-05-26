"""
Supply Chain Demand Forecasting
Store Item Demand Forecasting — 10 Stores × 50 Items (2013-2017)
Author: Dheeraj Kranthi
GitHub Project 3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH = "/home/claude/train.csv"
OUTPUT_DIR = "/mnt/user-data/outputs"

PALETTE = {
    "primary":   "#2563EB",
    "secondary": "#16A34A",
    "accent":    "#DC2626",
    "orange":    "#EA580C",
    "purple":    "#7C3AED",
    "neutral":   "#6B7280",
    "bg":        "#F8FAFC",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ─────────────────────────────────────────────
# 1. LOAD & PARSE
# ─────────────────────────────────────────────
print("=" * 60)
print("SUPPLY CHAIN DEMAND FORECASTING")
print("=" * 60)

df = pd.read_csv(TRAIN_PATH, parse_dates=['date'])
print(f"\n✅ Loaded: {df.shape[0]:,} rows")
print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
print(f"   Stores: {df['store'].nunique()} | Items: {df['item'].nunique()}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[1/5] Engineering time-series features...")

def add_features(data):
    d = data.copy()
    d['year']        = d['date'].dt.year
    d['month']       = d['date'].dt.month
    d['day']         = d['date'].dt.day
    d['dayofweek']   = d['date'].dt.dayofweek      # 0=Mon, 6=Sun
    d['dayofyear']   = d['date'].dt.dayofyear
    d['weekofyear']  = d['date'].dt.isocalendar().week.astype(int)
    d['quarter']     = d['date'].dt.quarter
    d['is_weekend']  = (d['dayofweek'] >= 5).astype(int)
    d['is_month_start'] = d['date'].dt.is_month_start.astype(int)
    d['is_month_end']   = d['date'].dt.is_month_end.astype(int)

    # Lag features per store-item
    df_sorted = d.sort_values(['store', 'item', 'date'])
    grp = df_sorted.groupby(['store', 'item'])['sales']
    d['lag_7']   = grp.shift(7)
    d['lag_14']  = grp.shift(14)
    d['lag_30']  = grp.shift(30)
    d['lag_90']  = grp.shift(90)
    d['lag_365'] = grp.shift(365)

    # Rolling means
    d['rolling_mean_7']  = grp.shift(1).rolling(7).mean()
    d['rolling_mean_30'] = grp.shift(1).rolling(30).mean()
    d['rolling_std_7']   = grp.shift(1).rolling(7).std()

    # Sin/cos encoding for seasonality
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['dow_sin']   = np.sin(2 * np.pi * d['dayofweek'] / 7)
    d['dow_cos']   = np.cos(2 * np.pi * d['dayofweek'] / 7)

    return d

df = add_features(df)
df.dropna(inplace=True)
print(f"   Features created: {df.shape[1]} columns")
print(f"   Rows after dropping NaN lags: {df.shape[0]:,}")

# ─────────────────────────────────────────────
# 3. EDA VISUALIZATIONS
# ─────────────────────────────────────────────
print("\n[2/5] Generating EDA charts...")

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(PALETTE["bg"])
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("Supply Chain Demand Forecasting — Exploratory Analysis",
             fontsize=18, fontweight='bold', y=0.98)

# ── Chart 1: Total daily sales across all stores
ax1 = fig.add_subplot(gs[0, :2])
daily = df.groupby('date')['sales'].sum().reset_index()
ax1.plot(daily['date'], daily['sales'], color=PALETTE["primary"],
         linewidth=1.2, alpha=0.8)
# Monthly moving average
monthly_ma = daily.set_index('date')['sales'].rolling(30).mean()
ax1.plot(daily['date'], monthly_ma.values, color=PALETTE["accent"],
         linewidth=2.5, label='30-day MA')
ax1.set_title("Total Daily Sales (All Stores)", fontweight='bold', fontsize=13)
ax1.set_xlabel("Date")
ax1.set_ylabel("Total Units Sold")
ax1.legend(fontsize=10)

# ── Chart 2: Sales by day of week
ax2 = fig.add_subplot(gs[0, 2])
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_avg = df.groupby('dayofweek')['sales'].mean()
colors_dow = [PALETTE["accent"] if i >= 5 else PALETTE["primary"] for i in range(7)]
bars = ax2.bar(dow_labels, dow_avg.values, color=colors_dow,
               edgecolor='white', linewidth=1.5)
ax2.set_title("Avg Sales by Day of Week", fontweight='bold', fontsize=13)
ax2.set_ylabel("Avg Units Sold")
weekend_patch = plt.Rectangle((0, 0), 1, 1, fc=PALETTE["accent"], label='Weekend')
weekday_patch = plt.Rectangle((0, 0), 1, 1, fc=PALETTE["primary"], label='Weekday')
ax2.legend(handles=[weekday_patch, weekend_patch], fontsize=9)

# ── Chart 3: Monthly seasonality
ax3 = fig.add_subplot(gs[1, 0])
month_avg = df.groupby('month')['sales'].mean()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']
ax3.plot(range(1, 13), month_avg.values, 'o-',
         color=PALETTE["secondary"], linewidth=2.5,
         markersize=8, markerfacecolor='white', markeredgewidth=2.5)
ax3.fill_between(range(1, 13), month_avg.values,
                 alpha=0.15, color=PALETTE["secondary"])
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels(month_labels, rotation=45, fontsize=9)
ax3.set_title("Monthly Seasonality", fontweight='bold', fontsize=13)
ax3.set_ylabel("Avg Units Sold")

# ── Chart 4: Store performance comparison
ax4 = fig.add_subplot(gs[1, 1])
store_total = df.groupby('store')['sales'].sum().sort_values(ascending=True)
colors_store = [PALETTE["primary"] if v < store_total.median()
                else PALETTE["secondary"] for v in store_total.values]
ax4.barh([f'Store {s}' for s in store_total.index], store_total.values / 1e6,
         color=colors_store, edgecolor='white', linewidth=1.2)
ax4.set_title("Total Sales by Store (M units)", fontweight='bold', fontsize=13)
ax4.set_xlabel("Total Sales (Millions)")

# ── Chart 5: Year-over-year growth
ax5 = fig.add_subplot(gs[1, 2])
yearly = df.groupby('year')['sales'].sum()
yoy_growth = yearly.pct_change() * 100
bars = ax5.bar(yearly.index[1:], yoy_growth.values[1:],
               color=[PALETTE["secondary"] if v > 0 else PALETTE["accent"]
                      for v in yoy_growth.values[1:]],
               edgecolor='white', linewidth=1.5)
ax5.axhline(0, color=PALETTE["neutral"], linewidth=1, linestyle='--')
ax5.set_title("Year-over-Year Sales Growth", fontweight='bold', fontsize=13)
ax5.set_ylabel("Growth (%)")
ax5.set_xlabel("Year")
for bar, val in zip(bars, yoy_growth.values[1:]):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.1 if val > 0 else bar.get_height() - 0.4,
             f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.savefig(f"{OUTPUT_DIR}/supply_chain_eda.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ EDA dashboard saved")

# ─────────────────────────────────────────────
# 4. MODEL — Gradient Boosting
# ─────────────────────────────────────────────
print("\n[3/5] Training Gradient Boosting model...")

FEATURES = [
    'store', 'item', 'year', 'month', 'day', 'dayofweek',
    'dayofyear', 'weekofyear', 'quarter', 'is_weekend',
    'is_month_start', 'is_month_end',
    'lag_7', 'lag_14', 'lag_30', 'lag_90', 'lag_365',
    'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7',
    'month_sin', 'month_cos', 'dow_sin', 'dow_cos'
]

# Train on 2013-2016, validate on 2017
train_df = df[df['year'] < 2017]
val_df   = df[df['year'] == 2017]

X_train, y_train = train_df[FEATURES], train_df['sales']
X_val,   y_val   = val_df[FEATURES],   val_df['sales']

model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=5,
    min_samples_leaf=20, subsample=0.8, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_pred = np.maximum(y_pred, 0)  # no negative sales

mae  = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
smape = np.mean(2 * np.abs(y_pred - y_val) / (np.abs(y_pred) + np.abs(y_val))) * 100

print(f"\n   ── Validation Results (2017) ──")
print(f"   MAE:   {mae:.2f} units")
print(f"   RMSE:  {rmse:.2f} units")
print(f"   SMAPE: {smape:.2f}%")

# ─────────────────────────────────────────────
# 5. MODEL PERFORMANCE CHARTS
# ─────────────────────────────────────────────
print("\n[4/5] Generating model performance charts...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Gradient Boosting — Forecasting Performance Dashboard",
             fontsize=17, fontweight='bold')

# ── Chart 1: Actual vs Predicted for one store-item
ax = axes[0]
sample = val_df[(val_df['store'] == 1) & (val_df['item'] == 1)].copy()
sample_pred = model.predict(sample[FEATURES])
ax.plot(sample['date'].values, sample['sales'].values,
        color=PALETTE["primary"], linewidth=2, label='Actual', alpha=0.9)
ax.plot(sample['date'].values, sample_pred,
        color=PALETTE["accent"], linewidth=2, linestyle='--', label='Forecast')
ax.set_title("Forecast vs Actual\n(Store 1, Item 1 — 2017)", fontweight='bold', fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Units Sold")
ax.legend(fontsize=10)
ax.tick_params(axis='x', rotation=30)

# ── Chart 2: Feature Importance
ax = axes[1]
importances = pd.Series(model.feature_importances_, index=FEATURES)
top10 = importances.nlargest(10).sort_values()
colors_fi = [PALETTE["primary"] if v > top10.median() else PALETTE["neutral"]
             for v in top10.values]
ax.barh(top10.index, top10.values, color=colors_fi,
        edgecolor='white', linewidth=1.2)
ax.set_title("Top 10 Feature Importances", fontweight='bold', fontsize=13)
ax.set_xlabel("Importance Score")

# ── Chart 3: Error distribution
ax = axes[2]
errors = y_val.values - y_pred
ax.hist(errors, bins=60, color=PALETTE["purple"], alpha=0.8,
        edgecolor='white', linewidth=0.8)
ax.axvline(0, color=PALETTE["accent"], linewidth=2, linestyle='--', label='Zero error')
ax.axvline(errors.mean(), color=PALETTE["secondary"], linewidth=2,
           label=f'Mean error: {errors.mean():.2f}')
ax.set_title("Forecast Error Distribution", fontweight='bold', fontsize=13)
ax.set_xlabel("Actual − Predicted (units)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/supply_chain_model.png", dpi=150, bbox_inches='tight')
plt.close()
print("   ✅ Model performance dashboard saved")

# ─────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────
print("\n[5/5] Key numbers for your README:")
print(f"   Training rows:    {len(train_df):,}")
print(f"   Validation rows:  {len(val_df):,}")
print(f"   Features:         {len(FEATURES)}")
print(f"   Model:            Gradient Boosting (300 estimators)")
print(f"   MAE:              {mae:.2f} units")
print(f"   RMSE:             {rmse:.2f} units")
print(f"   SMAPE:            {smape:.2f}%")
print(f"\n✅ All outputs saved to {OUTPUT_DIR}/")
print("=" * 60)
