
# 🏙️ BrickByBrick | AI Real Estate Predictor

**BrickByBrick** is an end-to-end Machine Learning dashboard built to estimate real estate prices using the **XGBoost** algorithm. The application features an interactive UI for property valuation and financial planning (EMI calculation).

## 🚀 Features
- **Price Prediction:** Real-time property valuation based on area, BHK, and location.
- **EMI Calculator:** Integrated mortgage tool to estimate monthly payments.
- **Market Insights:** Interactive visualizations for price trends and feature importance.
- **Tech Stack:** Python, Streamlit, Scikit-Learn, XGBoost, and Plotly.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/BrickByBrick.git](https://github.com/yourusername/BrickByBrick.git)
   cd BrickByBrick
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## 📊 How it Works
The model uses an **XGBoost Regressor** trained on historical property data. Categorical data (like Location) is handled via **Label Encoding**, and the dashboard provides transparency by showing **Feature Importance** (what factors drive the price most).

