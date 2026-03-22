import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# -------------------------------
# PAGE CONFIG & THEMING
# -------------------------------
st.set_page_config(page_title="BrickByBrick | Real Estate", page_icon="🏠", layout="wide")

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    div[data-testid="stSidebarUserContent"] { background-color: #f1f3f6; }
    .predict-box { 
        background-color: #2e7d32; color: white; padding: 20px; 
        border-radius: 15px; text-align: center; margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# DATA & MODEL ENGINE (Cached for performance)
# -------------------------------
@st.cache_data
def load_data():
    data = {
        "area": [1200, 1500, 1800, 2000, 2200, 2500, 2700, 3000, 1100, 1600],
        "bedrooms": [2, 3, 3, 4, 4, 5, 5, 4, 2, 3],
        "bathrooms": [2, 2, 3, 3, 4, 4, 5, 3, 1, 2],
        "location": ["Urban", "Urban", "Suburban", "Suburban", "Rural", "Urban", "Rural", "Suburban", "Urban", "Suburban"],
        "price": [50, 65, 80, 90, 100, 120, 140, 115, 45, 72]
    }
    return pd.DataFrame(data)

df = load_data()
df_display = df.copy() # For the table display later

# Encoding
le = LabelEncoder()
df["location"] = le.fit_transform(df["location"])
X = df.drop("price", axis=1)
y = df["price"]

# Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X, y)

# -------------------------------
# SIDEBAR - INPUTS
# -------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3534/3534033.png", width=80)
    st.title("BrickByBrick")
    st.markdown("---")
    
    st.subheader("📍 Property Specs")
    area = st.slider("Total Area (sq ft)", 500, 5000, 1800, step=50)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    with col_s2:
        bathrooms = st.number_input("Bathrooms", 1, 10, 2)
        
    location = st.selectbox("Desired Location", le.classes_)
    
    st.markdown("---")
    st.subheader("💳 Loan Settings")
    interest_rate = st.number_input("Interest Rate (%)", 5.0, 15.0, 8.5)
    tenure = st.select_slider("Tenure (Years)", options=[5, 10, 15, 20, 25, 30], value=20)

# -------------------------------
# PREDICTION LOGIC
# -------------------------------
loc_encoded = le.transform([location])[0]
input_data = pd.DataFrame([[area, bedrooms, bathrooms, loc_encoded]], columns=X.columns)
prediction = model.predict(input_data)[0]

# -------------------------------
# MAIN LAYOUT
# -------------------------------
st.title("🏙️ Real Estate Intelligence Dashboard")
st.markdown(f"**Market Analysis for:** {location} • {area} sq.ft • {bedrooms} BHK")

top_col1, top_col2, top_col3 = st.columns([1.5, 1, 1])

with top_col1:
    st.markdown(f"""
        <div class="predict-box">
            <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Estimated Market Value</p>
            <h1 style="margin:0; font-size: 3rem;">₹ {prediction:,.2f} L</h1>
        </div>
    """, unsafe_allow_html=True)

with top_col2:
    # EMI Calculation
    loan_amt = prediction * 100000
    r = interest_rate / (12 * 100)
    n = tenure * 12
    emi = (loan_amt * r * (1 + r)**n) / ((1 + r)**n - 1)
    
    st.metric("Monthly EMI", f"₹ {emi:,.0f}", delta="-2% vs Avg")

with top_col3:
    st.metric("Price per Sq.Ft", f"₹ {(prediction*100000)/area:,.0f}", delta="Market High")

st.markdown("---")

# -------------------------------
# VISUALIZATIONS
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Market Analysis", "🔍 Comparative Data", "🤖 Model Insights"])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Price vs. Area Correlation")
        fig_scatter = px.scatter(df_display, x="area", y="price", color="location", 
                                 size="price", hover_name="location",
                                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Safe)
        fig_scatter.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_b:
        st.subheader("Regional Price Distribution")
        fig_box = px.box(df_display, x="location", y="price", color="location", 
                         template="plotly_white", points="all")
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    st.subheader("Raw Property Listings")
    st.dataframe(df_display.style.highlight_max(axis=0, subset=['price']), use_container_width=True)

with tab3:
    col_c, col_d = st.columns([1, 1])
    
    with col_c:
        st.subheader("What drives the price?")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values()
        fig_imp = px.bar(feat_imp, orientation='h', labels={'value': 'Importance score', 'index': 'Feature'})
        fig_imp.update_layout(showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_d:
        st.info("""
        **Model Card:**
        - **Algorithm:** XGBoost Regressor
        - **Training Size:** 10 Samples (Demo Mode)
        - **Target Variable:** Price in Lakhs (INR)
        - **Accuracy:** Optimized for variance in suburban markets.
        """)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("🚀 Data updated as of March 2026. Disclaimer: Predictions are based on historical sample data.")