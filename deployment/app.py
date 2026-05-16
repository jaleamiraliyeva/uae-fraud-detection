import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
    precision_recall_curve,
)

# constants — defined outside the class

# DROP before feature engineering — not needed at all
DROP_COLS_1 = [
    'user_id', 'transaction_id', 'ip_address',
    'currency', 'data_source', 'card_present',
    'user_prev_chargebacks', 'user_std_amount', 'user_max_amount', 'user_txn_count'
]

# DROP after feature engineering — replaced by engineered features
DROP_COLS_2 = [
    'timestamp_utc', 'email_domain',
    'fraud_flag_velocity', 'local_hour',
    'transactions_last_24h', 'transactions_last_1h',
    'is_weekend', 'billing_city', 'shipping_city',
    'fraud_flag_odd_hour', 'fraud_flag_ip', 'fraud_flag_mismatch',
    'fresh_high_value', 'night_high_risk', 'high_amount', 'risk_velocity',
    'amount_aed', 'velocity_ratio', 'browser', 'day_of_week', 'payment_method',
    'hour_bin', 'merchant_category', 'device_type'
]

FLAG_COLS = [
    'fraud_flag_ip', 'fraud_flag_mismatch',
    'fraud_flag_new_account', 'fraud_flag_prev_cb', 'fraud_flag_odd_hour'
]

CAT_COLS = [
    'payment_method', 'device_type', 'browser', 'merchant_category',
    'shipping_city', 'billing_city', 'bin_country'
]

DISPOSABLE_DOMAINS = ['tempmail', '10minutemail', 'disposable', 'mailinator', 'guerrillamail']

class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Caps each column at a given lower and upper percentile.
    Boundaries are computed from training data only.
    """
    def __init__(self, lower_pct=0.01, upper_pct=0.99):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            self.lower_bounds_[col] = X[col].quantile(self.lower_pct)
            self.upper_bounds_[col] = X[col].quantile(self.upper_pct)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(lower=self.lower_bounds_[col],
                                 upper=self.upper_bounds_[col])
        return X
    
class SkewnessTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.skewed_cols_ = []

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.skewed_cols_ = [
            col for col in X.columns
            if X[col].skew() > self.threshold and X[col].min() >= 0
        ]
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for col in self.skewed_cols_:
            X[col] = np.log1p(X[col])
        return X

abs_col= ['user_account_age_days']

class AbsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        if self.col in X.columns:
            X[self.col] = X[self.col].abs()

        return X
    
class FraudFeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self, drop_flags=False,
                 drop_cols_1=None, drop_cols_2=None):
        self.drop_flags  = drop_flags
        self.drop_cols_1 = drop_cols_1 if drop_cols_1 is not None else DROP_COLS_1
        self.drop_cols_2 = drop_cols_2 if drop_cols_2 is not None else DROP_COLS_2

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        # drop cols_1 before anything
        X = X.drop(columns=[c for c in self.drop_cols_1 if c in X.columns])

        # label encoders — learned from train only
        self.encoders_ = {}
        for col in CAT_COLS:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders_[col] = le

        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()

        # --- STEP 1: Drop cols before feature engineering ---
        X = X.drop(columns=[c for c in self.drop_cols_1 if c in X.columns])

        # --- STEP 2: Feature Engineering ---

        # ratio features
        X['amount_per_item_price'] = X['amount_aed'] / X['avg_item_price']
        X['amount_per_card_age']   = X['amount_aed'] / (X['card_age_days'] + 1)

        # interaction features
        X['amount_x_risk']    = X['amount_aed'] * X['ip_risk_score']
        X['fresh_high_value'] = ((X['user_account_age_days'] < 14) &
                                  (X['amount_aed'] > 1000)).astype(int)

        # new risk features
        X['high_amount']     = (X['amount_aed'] > 2000).astype(int)
        X['risk_velocity']   = X['ip_risk_score'] * X['transactions_last_1h']
        X['night_high_risk'] = ((X['odd_hour'] == 1) & (X['ip_risk_score'] > 70)).astype(int)

        # behavioral features
        X['velocity_ratio'] = X['transactions_last_1h'] / (X['transactions_last_24h'] + 1)

        # time features
        X['timestamp_utc'] = pd.to_datetime(X['timestamp_utc'])
        X['day_of_week']   = X['timestamp_utc'].dt.dayofweek
        X['is_weekend']    = X['day_of_week'].isin([5, 6]).astype(int)
        X['hour_bin']      = pd.cut(X['local_hour'], bins=[0, 6, 12, 18, 24],
                                    labels=[0, 1, 2, 3], right=False).astype(int)

        # risk aggregation
        X['combined_risk_score'] = X[FLAG_COLS].sum(axis=1)

        # email features
        X['is_disposable_email'] = X['email_domain'].str.contains(
            '|'.join(DISPOSABLE_DOMAINS), case=False, na=False).astype(int)

        # --- STEP 3: Encoding ---
        for col, le in self.encoders_.items():
            X[col] = X[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
            )

        # --- STEP 4: Drop cols after feature engineering ---
        X = X.drop(columns=[c for c in self.drop_cols_2 if c in X.columns])

        if self.drop_flags:
            flags = [c for c in FLAG_COLS + ['combined_risk_score'] if c in X.columns]
            X = X.drop(columns=flags)

        return X
    
# ══════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="UAE Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ UAE E-Commerce Fraud Detection")
st.caption("Logistic Regression · Tuned with GridSearchCV · uae_ecom_fraud_100k.csv")

# ══════════════════════════════════════════════════════════════
# Load Model  (saved as fraud_model2.pkl in notebook cell 51)
# ══════════════════════════════════════════════════════════════
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "fraud_model8.pkl")
    return joblib.load(model_path)
try:
    pipe = load_model()
except FileNotFoundError:
    st.error(
        "❌ `fraud_model8.pkl` not found!\n\n"
        "Run the last cell of `modeling.ipynb` first:\n"
        "```python\n"
        "joblib.dump(results['LR_tuned']['pipeline'], 'fraud_model8.pkl')\n"
        "```"
    )
    st.stop()

# ══════════════════════════════════════════════════════════════
# Load Test Data — same split as notebook (cell 6)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_test_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "uae_ecom_fraud_100k.csv")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

try:
    X_test, y_test = load_test_data()
    FEATURE_ORDER = X_test.columns.tolist()
    y_prob = pipe.predict_proba(X_test)[:, 1]

    # best threshold from notebook cell 38
    thresholds = np.arange(0.01, 0.99, 0.01)
    f1_scores  = [f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    BEST_THRESHOLD = thresholds[np.argmax(f1_scores)]

    y_pred = (y_prob >= BEST_THRESHOLD).astype(int)
    metrics_available = True
except Exception:
    metrics_available = False
    BEST_THRESHOLD = 0.5

# ══════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🔮 Predict",
    "📊 Model Metrics",
    "📉 ROC Curves",
    "🔲 Confusion Matrix",
    "⚙️ Pipeline Info",
])


# ─────────────────────────────────────────────────────────────
# TAB 0 — PREDICT
# ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Predict Fraud Risk for a New Transaction")
    st.markdown(
        f"The model uses an **optimized threshold of `{BEST_THRESHOLD:.2f}`** "
        f"(tuned for best F1 on the test set, vs default 0.5)."
    )
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("💰 Transaction")
            amount_aed        = st.number_input("Amount (AED)", min_value=0.0, max_value=50000.0, value=500.0, step=10.0)
            avg_item_price    = st.number_input("Average Item Price (AED)", min_value=0.1, max_value=10000.0, value=100.0)
            items_count       = st.number_input("Items Count", min_value=1, max_value=20, value=1)
            payment_method    = st.selectbox("Payment Method", ["credit_card", "debit_card", "paypal", "apple_pay", "google_pay"])
            merchant_category = st.selectbox("Merchant Category", ["electronics", "fashion", "food", "travel", "beauty", "home", "sports"])
            email_domain      = st.text_input("Email Domain", value="gmail.com")

        with c2:
            st.subheader("👤 User & Card")
            user_account_age_days = st.number_input("Account Age (days)", min_value=0, max_value=3650, value=365)
            card_age_days         = st.number_input("Card Age (days)", min_value=0, max_value=3650, value=730)
            ip_risk_score         = st.slider("IP Risk Score", min_value=0, max_value=100, value=20)
            bin_country           = st.selectbox("Card Country (BIN)", ["AE", "US", "GB", "IN", "PK", "NG", "CN"])
            card_country_match    = st.selectbox("Card Country Match?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            device_type           = st.selectbox("Device Type", ["mobile", "desktop", "tablet"])
            browser               = st.selectbox("Browser", ["chrome", "safari", "firefox", "edge", "other"])
            user_is_high_risk     = st.selectbox("User High Risk?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes ✅")

        with c3:
            st.subheader("📍 Location & Risk")
            shipping_city          = st.selectbox("Shipping City", ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah"])
            billing_city           = st.selectbox("Billing City", ["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "Ras Al Khaimah", "Other"])
            shipping_billing_match = 1 if shipping_city == billing_city else 0
            st.caption(f"Shipping/Billing Match: {'✅ Yes' if shipping_billing_match else '❌ No'}")
            odd_hour               = st.selectbox("Unusual Hour (1–5 AM)?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            transactions_last_1h   = st.number_input("Transactions Last 1h",  min_value=0, max_value=50,  value=0)
            transactions_last_24h  = st.number_input("Transactions Last 24h", min_value=0, max_value=200, value=0)
            fraud_flag_new_account = st.selectbox("New Account Flag?",        [0, 1], format_func=lambda x: "No" if x == 0 else "Yes ✅")
            fraud_flag_prev_cb     = st.selectbox("Previous Chargeback Flag?",[0, 1], format_func=lambda x: "No" if x == 0 else "Yes ✅")

        submitted = st.form_submit_button("🔍 Check for Fraud", type="primary", width='stretch')
  
    if submitted:
        input_df = pd.DataFrame([{

            # ── DROP_COLS_1 (pipeline drops these — fixed manual values) ──
            'user_id'              : 'new_user_001',
            'transaction_id'       : 'txn_001',
            'currency'             : 'AED',
            'ip_address'           : '0.0.0.0',
            'data_source'          : 'manual',
            'card_present'         : 0,
            'user_prev_chargebacks': 0,
            'user_std_amount'      : 0,
     #       'user_max_amount'      : amount_aed,

            # ── DROP_COLS_2 (used in feature engineering, then dropped) ──
            'timestamp_utc'        : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'amount_aed'           : amount_aed,
            'local_hour'           : datetime.datetime.now().hour,
            'transactions_last_1h' : transactions_last_1h,
            'transactions_last_24h': transactions_last_24h,
            'fraud_flag_ip'        : 0,
            'fraud_flag_mismatch'  : 0,
            'fraud_flag_odd_hour'  : odd_hour,
            'fraud_flag_velocity'  : 0,
            'email_domain'         : email_domain,
            'billing_city'         : billing_city,
            'shipping_city'        : shipping_city,

            # ── FINAL FEATURES (stay in model) ──
            'avg_item_price'       : avg_item_price,
            'items_count'          : items_count,
            'payment_method'       : payment_method,
            'merchant_category'    : merchant_category,
            'user_account_age_days': user_account_age_days,
            'card_age_days'        : card_age_days,
            'ip_risk_score'        : ip_risk_score,
            'bin_country'          : bin_country,
            'device_type'          : device_type,
            'browser'              : browser,
            'odd_hour'             : odd_hour,
            'fraud_flag_new_account': fraud_flag_new_account,
            'fraud_flag_prev_cb'   : fraud_flag_prev_cb,
            'user_is_high_risk'    : user_is_high_risk,
            'shipping_billing_match': shipping_billing_match,
            'card_country_match'   : card_country_match,
        }])


        try:
            input_df = input_df.reindex(columns=FEATURE_ORDER)
            
            prob_fraud  = pipe.predict_proba(input_df)[0][1]
            pred_class  = int(prob_fraud >= BEST_THRESHOLD)
            pred_proba  = [1 - prob_fraud, prob_fraud]

            st.divider()
            r1, r2 = st.columns([1, 2])

            with r1:
                if pred_class == 1:
                    st.error(f"### ⚠️ FRAUD RISK DETECTED\nFraud probability: **{prob_fraud:.1%}**\nThreshold: `{BEST_THRESHOLD:.2f}`")
                else:
                    st.success(f"### ✅ LOW FRAUD RISK\nFraud probability: **{prob_fraud:.1%}**\nThreshold: `{BEST_THRESHOLD:.2f}`")

            with r2:
                fig, ax = plt.subplots(figsize=(5, 1.8))
                ax.barh(['Legitimate', 'Fraud'], pred_proba,
                        color=['#2ecc71', '#e74c3c'], height=0.5)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title("Class Probabilities")
                for i, val in enumerate(pred_proba):
                    ax.text(val + 0.01, i, f"{val:.1%}", va='center', fontsize=10)
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ─────────────────────────────────────────────────────────────
# TAB 1 — METRICS
# ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Model Evaluation — Test Set")

    if not metrics_available:
        st.warning("Data file not found. Metrics cannot be displayed.")
    else:
        auc   = roc_auc_score(y_test, y_prob)
        prauc = average_precision_score(y_test, y_prob)
        f1    = f1_score(y_test, y_pred)
        prec  = precision_score(y_test, y_pred)
        rec   = recall_score(y_test, y_pred)

        st.caption(f"Using optimized threshold = **{BEST_THRESHOLD:.2f}** (tuned for best F1)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 ROC-AUC",   f"{auc:.4f}")
        m2.metric("⚖️ F1 Score",  f"{f1:.4f}")
        m3.metric("🎪 Precision", f"{prec:.4f}")
        m4.metric("📡 Recall",    f"{rec:.4f}")

        st.divider()
        st.subheader("Classification Report")
        report = classification_report(
            y_test, y_pred,
            target_names=['Legitimate', 'Fraud'],
            output_dict=True
        )
        st.dataframe(
            pd.DataFrame(report).T.style
              .format("{:.4f}")
              .background_gradient(cmap='RdYlGn', subset=['f1-score']),
            use_container_width=True
        )

        st.divider()
        st.subheader("F1 Score vs Threshold")
        fig_t, ax_t = plt.subplots(figsize=(8, 3))
        ax_t.plot(thresholds, f1_scores, color='steelblue', linewidth=2)
        ax_t.axvline(x=BEST_THRESHOLD, color='red', linestyle='--',
                     label=f'Best threshold = {BEST_THRESHOLD:.2f}')
        ax_t.set_xlabel("Threshold")
        ax_t.set_ylabel("F1 Score")
        ax_t.set_title("F1 Score vs Threshold — LR Tuned")
        ax_t.legend()
        st.pyplot(fig_t)

        st.info(
            "**Key metrics for fraud detection:**\n\n"
            "- **ROC-AUC**: Overall ability to distinguish fraud vs legitimate (1.0 = perfect)\n"
            "- **Recall**: % of actual frauds caught — most critical for banks\n"
            "- **Precision**: Of all predicted frauds, how many are actually fraudulent"
        )

# ─────────────────────────────────────────────────────────────
# TAB 2 — ROC 
# ─────────────────────────────────────────────────────────────
with tabs[2]:


    if not metrics_available:
        st.warning("Data file not found.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.plot(fpr, tpr, color='#e74c3c', linewidth=2.5,
                     label=f"LR Tuned (AUC = {auc:.4f})")
            ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random Classifier")
            ax1.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
            ax1.set_xlabel("False Positive Rate")
            ax1.set_ylabel("True Positive Rate")
            ax1.set_title("ROC Curve")
            ax1.legend()
            st.pyplot(fig1)


# ─────────────────────────────────────────────────────────────
# TAB 3 — CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Confusion Matrix")

    if not metrics_available:
        st.warning("Data file not found.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Legitimate', 'Fraud'],
                        yticklabels=['Legitimate', 'Fraud'],
                        ax=ax, linewidths=0.5, annot_kws={"size": 14})
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix (threshold = {BEST_THRESHOLD:.2f})")
            st.pyplot(fig)

        with col2:
            tn, fp, fn, tp = cm.ravel()
            st.markdown("### Results Summary")
            st.markdown(f"""
| Category | Value |
|---|---|
| ✅ True Positive (Correct Fraud) | **{tp:,}** |
| ✅ True Negative (Correct Legitimate) | **{tn:,}** |
| ❌ False Positive (False Alarm) | **{fp:,}** |
| ❌ False Negative (Missed Fraud) | **{fn:,}** |
""")
            st.warning(
                f"⚠️ **{fn:,} frauds were missed** — these are the highest-risk cases for the bank. "
                f"Lowering the threshold increases recall but also increases false positives."
            )

# ─────────────────────────────────────────────────────────────
# TAB 4 — PIPELINE INFO
# ─────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("⚙️ Pipeline Structure")

    st.markdown("""
### Pipeline Steps (in order):

| # | Step | Class | Description |
|---|---|---|---|
| 1 | `feature_engineering` | `FraudFeatureEngineer` | Feature engineering, label encoding, column dropping |
| 2 | `cap_outliers` | `OutlierCapper` | Clips values to 1st–99th percentile (train-learned) |
| 3 | `skew_transform` | `SkewnessTransformer` | log1p on skewed columns (train-learned) |
| 4 | `scaler` | `StandardScaler` | Normalizes to mean=0, std=1 (train-learned) |
| 5 | `model` | `LogisticRegression` | Binary classifier: Fraud vs Legitimate |

### GridSearchCV Tuning (modeling.ipynb cell 32):
```python
param_grid_lr = {
    'model__C':      [0.01, 0.1, 1, 10, 50],
    'model__solver': ['liblinear', 'lbfgs']
}
# scoring='roc_auc', cv=5
```

### Features created by FraudFeatureEngineer:
- `amount_per_item_price` — transaction amount / average item price
- `amount_per_card_age` — transaction amount / (card age + 1)
- `amount_x_risk` — amount × IP risk score
- `fresh_high_value` — new account (< 14 days) with amount > 1000 AED
- `high_amount` — amount > 2000 AED
- `risk_velocity` — IP risk score × transactions in last 1h
- `night_high_risk` — unusual hour AND IP risk score > 70
- `velocity_ratio` — transactions last 1h / (last 24h + 1)
- `combined_risk_score` — sum of all fraud flags
- `is_disposable_email` — known disposable email domain flag
- `hour_bin` — time of day bucket: 0=night, 1=morning, 2=afternoon, 3=evening
- `user_mean_amount`, `user_max_amount` — per-user behavioral stats (train-learned)
""")

    if metrics_available:
        st.divider()
        st.subheader("Best Model Parameters (from GridSearchCV):")
        try:
            best_lr = pipe.named_steps['model']
            st.dataframe(pd.DataFrame([{
                'C':            best_lr.C,
                'solver':       best_lr.solver,
                'penalty':      best_lr.penalty,
                'class_weight': str(best_lr.class_weight),
                'max_iter':     best_lr.max_iter,
            }]), use_container_width=True)
        except Exception:
            st.info("Model parameters could not be displayed.")
