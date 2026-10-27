"""Simple Linear Regression: predict a house price from ONE input (the area). Run: streamlit run app.py"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Simple Linear Regression", page_icon="📈", layout="wide")
SQM = 10.7639
DATA = Path(__file__).parent / "data" / "housing.csv"

@st.cache_data
def load():
    d = pd.read_csv(DATA)
    if "price_lakh" not in d.columns:
        return None
    return d[["area", "price_lakh"]].dropna()

@st.cache_resource
def fit(d):
    Xtr, Xte, ytr, yte = train_test_split(d[["area"]], d["price_lakh"], test_size=0.2, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    p = m.predict(Xte)
    return m, Xte, yte, {"R2": r2_score(yte, p), "RMSE": mean_squared_error(yte, p) ** 0.5, "MAE": mean_absolute_error(yte, p), "n_train": len(Xtr), "n_test": len(Xte)}

def fmt_inr(lakh):
    return f"₹{lakh / 100:,.2f} crore" if lakh >= 100 else f"₹{lakh:,.1f} lakh"

d = load()
st.title("📈 Simple Linear Regression: house price from area only")
if d is None:
    st.error("This app needs data/housing.csv with the Mumbai listings (a 'price_lakh' column). Copy that file into this project.")
    st.stop()
model, Xte, yte, m = fit(d)
a, b = float(model.intercept_), float(model.coef_[0])

st.markdown(f"""
**Simple linear regression** uses **one input** to predict one output. Here the input is the **area** (sq ft) and the output is the
**price** (lakh rupees). The model draws one straight line through {len(d):,} real Mumbai-region listings:
""")
st.latex(rf"\text{{price (lakh)}} = {a:,.2f} + {b:.4f} \times \text{{area (sq ft)}}")
st.caption(f"Slope: each extra square foot adds about ₹{b * 1e5:,.0f} to the price. Trained on {m['n_train']:,} listings, tested on {m['n_test']:,} unseen ones.")

c1, c2, c3 = st.columns(3)
c1.metric("R² (test)", f"{m['R2']:.2f}")
c2.metric("Typical error (RMSE)", f"{m['RMSE']:,.0f} lakh")
c3.metric("Average error (MAE)", f"{m['MAE']:,.0f} lakh")

left, right = st.columns([3, 2])
with left:
    s = d.sample(min(3000, len(d)), random_state=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(s["area"], s["price_lakh"], s=6, alpha=0.3, label="listings")
    xs = np.linspace(d["area"].min(), d["area"].max(), 100)
    ax.plot(xs, a + b * xs, color="red", lw=2, label="fitted line")
    ax.set_xlabel("Area (sq ft)"); ax.set_ylabel("Price (lakh rupees)"); ax.legend(); fig.tight_layout()
    st.pyplot(fig)
with right:
    st.subheader("Predict from area")
    unit = st.radio("Unit", ["square feet", "square metres"], horizontal=True)
    val = st.number_input("Area", 10.0, 10000.0, 860.0 if unit == "square feet" else 80.0, step=5.0)
    cur = st.radio("Show price in", ["Indian rupees (₹)", "US dollars ($)"], horizontal=True)
    rate = st.number_input("Today's rate: ₹ for $1", 1.0, 1000.0, 85.0, step=0.1)
    area = val * SQM if unit == "square metres" else val
    lakh = a + b * area
    if st.button("Predict price", type="primary"):
        if lakh <= 0:
            st.error(f"The straight line gives {lakh:,.0f} lakh, an impossible negative price. One input cannot describe small flats.")
        else:
            st.metric("Estimated price", f"${lakh * 1e5 / rate:,.0f}" if "$" in cur else fmt_inr(lakh))
            st.caption(f"Typical error is about ±{m['RMSE']:,.0f} lakh, so treat this as a rough guess.".replace("$", "\\$"))
        if not d["area"].min() <= area <= d["area"].max():
            st.warning(f"{area:,.0f} sq ft is outside the data ({d['area'].min():.0f} to {d['area'].max():.0f}).")

st.subheader("Why one input is not enough")
sub = d[d["area"].between(area * 0.95, area * 1.05)]
if len(sub) >= 30:
    lo, hi = sub["price_lakh"].quantile([0.1, 0.9])
    st.info(f"Real listings of about {area:,.0f} sq ft cost anywhere from {fmt_inr(lo)} to {fmt_inr(hi)} (middle 80%), "
            "because location, BHK, type and age also matter. Using several inputs at once is **multiple linear regression**, "
            "which is the next project.")
st.markdown("Multiple regression project: https://multiple-linear-regression-fxoezxtlldd25e2qfqcmvv.streamlit.app")
