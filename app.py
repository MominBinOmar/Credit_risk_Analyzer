# your_money_your_mirror_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# Custom theme configuration
st.set_page_config(
    page_title="Your Money, Your Mirror",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import Mario theme CSS from external file
with open("mario_theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Helper for card container
card_style = "background-color: #fffbe6; color: #111; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(230,57,70,0.10); border: 1.5px solid #e63946;"

main_bg = "#fffbe6"

# -------------- Sidebar Navigation (Mario Style) ----------------
st.sidebar.title("🍄 Mario Navigation")
page = st.sidebar.radio("Go to", ["🏠 Mario Welcome", "🍄 Credit Risk Analyzer", "⭐ Lifestyle Clusterer", "💰 Smart Budget Allocator"])

# -------------- Welcome Page ----------------
if page == "🏠 Mario Welcome":
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <h1 class='mario-header' style="text-align: center;">🍄 Your Money, Your Mirror</h1>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{card_style} text-align: center; margin-bottom: 20px;">
            <p style="font-size: 18px;">Welcome to your personalized Mario-themed finance insights app. Choose a module from the sidebar:</p>
            <ul style="list-style-type: none; padding-left: 0; font-size: 16px;">
                <li style="margin: 10px 0;">👨‍🔬 <strong>Credit Risk Analyzer</strong> - Assess your creditworthiness</li>
                <li style="margin: 10px 0;">⭐ <strong>Lifestyle Clusterer</strong> - Understand your spending patterns</li>
                <li style="margin: 10px 0;">💰 <strong>Smart Budget Allocator</strong> - Optimize your financial planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{card_style} display: flex; justify-content: center; align-items: center; margin-bottom: 20px;">
            <img src="https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExMjZhajAxZDVteWw5YXkybWVxem50YzRmcnRjOTlzOTA1Mnk1eDlscCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/qfridHRcLlMLMtWueu/giphy.gif" width="400">
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{card_style} text-align: center;">
            <h3 class='mario-section'>🔍 See Your Financial Future Clearly</h3>
            <p style="color: #457b9d;">Make informed decisions about your money with AI-powered insights</p>
        </div>
        """, unsafe_allow_html=True)

# -------------- Module 1: Credit Risk Analyzer ----------------
elif page == "🍄 Credit Risk Analyzer":
    st.header("🍄 Credit Risk Analyzer")
    st.markdown(f"""
    <div style="{card_style}">
        <h3 class='mario-section'>Understanding Your Credit Risk</h3>
        <p>This module helps you assess your credit risk based on various financial and personal factors. 
        The analysis considers multiple aspects of your financial health to provide a comprehensive risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div style="{card_style} margin-bottom: 20px;">
            <h3 class='mario-section'>Data Source</h3>
        </div>
        """, unsafe_allow_html=True)
        data_source = st.radio("Choose data source:", ["📊 Upload Dataset", "🎲 Use Simulated Data"])
        if data_source == "📊 Upload Dataset":
            uploaded_file = st.file_uploader("Upload credit risk dataset (CSV)", type=["csv"])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.success("Dataset uploaded successfully!")
            else:
                st.markdown(f"""
                <div style="{card_style} text-align: center; margin: 10px 0;">
                    <p style="color: #000; margin: 0;">🍄 Please upload a dataset or switch to simulated data.</p>
                </div>
                """, unsafe_allow_html=True)
                st.stop()
        else:
            np.random.seed(42)
            # Simulate base features
            base_df = pd.DataFrame({
                'Income': np.random.normal(50000, 15000, 100).astype(int),
                'Age': np.random.normal(35, 10, 100).astype(int),
                'Debt': np.random.normal(15000, 5000, 100).astype(int),
                'Employment_Length': np.random.normal(5, 3, 100).astype(int),
                'Married': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
                'Dependents': np.random.poisson(1, 100),
                'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 100, p=[0.2, 0.5, 0.2, 0.1]),
                'Home_Ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], 100, p=[0.3, 0.2, 0.5])
            })
            # Map education and home ownership to numeric
            education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
            home_map = {'Rent': 0, 'Mortgage': 1, 'Own': 2}
            base_df['Education_num'] = base_df['Education'].map(education_map)
            base_df['Home_num'] = base_df['Home_Ownership'].map(home_map)
            # Simulate expenses and savings
            base_df['Expenses'] = (base_df['Income'] * (0.4 + 0.1 * base_df['Dependents']) + np.random.normal(0, 2000, 100)).clip(5000, None)
            base_df['Savings'] = (base_df['Income'] - base_df['Expenses'] - 0.1 * base_df['Debt']).clip(0, None)
            base_df['Debt_to_Income'] = base_df['Debt'] / (base_df['Income'] + 1)
            # Improved credit score simulation
            base_df['Credit_Score'] = (
                600
                + (base_df['Income'] - 50000) / 1000
                - (base_df['Debt'] - 15000) / 2000
                + base_df['Employment_Length'] * 5
                + base_df['Education_num'] * 20
                + base_df['Home_num'] * 15
                + np.random.normal(0, 30, 100)
            ).clip(300, 850).astype(int)
            # Improved default probability
            prob_default = (
                0.15
                + 0.25 * base_df['Debt_to_Income']
                + 0.03 * base_df['Dependents']
                - 0.02 * base_df['Employment_Length']
                - 0.03 * base_df['Home_num']
                - 0.01 * base_df['Education_num']
                + 0.01 * np.abs(base_df['Age'] - 40) / 10
                + 0.02 * (base_df['Married'] == 1)
                - 0.05 * (base_df['Savings'] > 10000)
                + np.random.normal(0, 0.03, 100)
            )
            base_df['Default'] = (np.random.rand(100) < prob_default.clip(0.05, 0.7)).astype(int)
            df = base_df
            st.markdown(f"""
            <div style="{card_style} text-align: center; margin: 10px 0;">
                <p style="color: #000; margin: 0;">🍄 Using simulated data for demonstration</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{card_style} margin-top: 20px;">
            <h3 class='mario-section'>Data Preview</h3>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(df.head())
        # --- Train mini credit score model ---
        mini_features = ['Income', 'Age', 'Debt', 'Employment_Length', 'Married', 'Dependents', 'Education_num', 'Home_num', 'Expenses', 'Savings', 'Debt_to_Income']
        mini_X = df[mini_features]
        mini_y = df['Credit_Score']
        mini_model = RandomForestRegressor(n_estimators=100, random_state=42)
        mini_model.fit(mini_X, mini_y)
        st.session_state['mini_model'] = mini_model
        # --- End mini model training ---
        if st.button("🚀 Train Model", key="train_button"):
            with st.spinner("Training model..."):
                df_processed = df.copy()
                df_processed = pd.get_dummies(df_processed, columns=['Education', 'Home_Ownership'])
                X = df_processed.drop('Default', axis=1)
                y = df_processed['Default']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LogisticRegression()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                pred_proba = model.predict_proba(X_test)[:, 1]
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['preds'] = preds
                st.session_state['pred_proba'] = pred_proba
                st.session_state['X'] = X
                st.session_state['feature_names'] = X.columns
                st.markdown(f"""
                <div style="{card_style} text-align: center; margin: 10px 0;">
                    <h3 style="color: #000; margin: 0;">🍄 Model trained successfully!</h3>
                </div>
                """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="{card_style}">
            <h3 class='mario-section'>📝 Your Credit Profile</h3>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        education = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD'])
        married = st.selectbox("Marital Status", ["Single", "Married"])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        st.subheader("Financial Information")
        income = st.number_input("Monthly Income (PKR)", min_value=0, value=50000)
        debt = st.number_input("Current Debt (PKR)", min_value=0, value=10000)
        employment_length = st.number_input("Years of Employment", min_value=0, max_value=50, value=5)
        home_ownership = st.selectbox("Home Ownership", ['Rent', 'Own', 'Mortgage'])
        married_encoded = 1 if married == "Married" else 0
        # Map user education and home ownership to numeric
        education_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
        home_map = {'Rent': 0, 'Mortgage': 1, 'Own': 2}
        education_num = education_map[education]
        home_num = home_map[home_ownership]
        # --- Calculate user expenses, savings, debt-to-income ---
        user_expenses = (income * (0.4 + 0.1 * dependents))
        user_savings = max(income - user_expenses - 0.1 * debt, 0)
        user_debt_to_income = debt / (income + 1)
        # --- Predict credit score using mini model ---
        if 'mini_model' in st.session_state:
            mini_input = pd.DataFrame({
                'Income': [income],
                'Age': [age],
                'Debt': [debt],
                'Employment_Length': [employment_length],
                'Married': [married_encoded],
                'Dependents': [dependents],
                'Education_num': [education_num],
                'Home_num': [home_num],
                'Expenses': [user_expenses],
                'Savings': [user_savings],
                'Debt_to_Income': [user_debt_to_income]
            })
            predicted_credit_score = int(st.session_state['mini_model'].predict(mini_input)[0])
        else:
            predicted_credit_score = 650
        st.markdown(f"<div style='{card_style} margin-bottom:10px; text-align:center;'><b>Predicted Credit Score:</b> <span style='font-size:1.5em;'>{predicted_credit_score}</span></div>", unsafe_allow_html=True)
        # --- End credit score prediction ---
        if 'model' in st.session_state:
            if st.button("🔮 Predict Credit Risk", key="predict_button"):
                user_expenses = (income * (0.4 + 0.1 * dependents))
                user_savings = max(income - user_expenses - 0.1 * debt, 0)
                user_debt_to_income = debt / (income + 1)
                user_input = pd.DataFrame({
                    'Income': [income],
                    'Age': [age],
                    'Debt': [debt],
                    'Credit_Score': [predicted_credit_score],
                    'Employment_Length': [employment_length],
                    'Married': [married_encoded],
                    'Dependents': [dependents],
                    'Education_num': [education_num],
                    'Home_num': [home_num],
                    'Expenses': [user_expenses],
                    'Savings': [user_savings],
                    'Debt_to_Income': [user_debt_to_income]
                })
                user_input = user_input.reindex(columns=st.session_state['feature_names'], fill_value=0)
                risk_score = st.session_state['model'].predict_proba(user_input)[0, 1]
                risk_level = "Low" if risk_score < 0.3 else "Medium" if risk_score < 0.7 else "High"
                risk_color = "#4CAF50" if risk_level == "Low" else "#FFC107" if risk_level == "Medium" else "#F44336"
                st.markdown(f"""
                <div style="{card_style} margin-top: 20px;">
                    <h3 class='mario-section'>📊 Prediction Results</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background-color: {risk_color}; color: white; padding: 10px; border-radius: 5px; text-align: center; margin: 10px 0;">
                    <h3>Risk Level: {risk_level}</h3>
                    <p>Probability of Default: {risk_score:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk_score * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Credit Risk Level"},
                    gauge={'axis': {'range': [0, 100]},
                           'steps': [
                               {'range': [0, 30], 'color': "lightgreen", 'name': 'Low Risk'},
                               {'range': [30, 70], 'color': "yellow", 'name': 'Medium Risk'},
                               {'range': [70, 100], 'color': "red", 'name': 'High Risk'}
                           ]}
                ))
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div style="{card_style} margin-top: 20px;">
                    <h3 class='mario-section'>📈 Feature Importance</h3>
                    <p>The following factors most influence your credit risk assessment:</p>
                </div>
                """, unsafe_allow_html=True)
                importance = pd.DataFrame({
                    'Feature': st.session_state['X'].columns,
                    'Importance': st.session_state['model'].coef_[0]
                })
                importance['Absolute_Importance'] = abs(importance['Importance'])
                importance = importance.sort_values('Absolute_Importance', ascending=False).head(5)
                fig = px.bar(importance, x='Feature', y='Importance', 
                           title='Top 5 Most Influential Factors',
                           color='Importance',
                           color_continuous_scale=['red', 'yellow', 'green'])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div style="{card_style} margin-top: 20px;">
                    <h4 class='mario-section'>What This Means:</h4>
                    <ul>
                        <li>Positive values indicate factors that increase your credit risk</li>
                        <li>Negative values indicate factors that decrease your credit risk</li>
                        <li>The longer the bar, the more influence that factor has on your risk assessment</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                results = pd.DataFrame({
                    'Feature': ['Income', 'Age', 'Debt', 'Credit Score', 'Employment Length', 
                              'Marital Status', 'Dependents', 'Education', 'Home Ownership', 'Risk Score'],
                    'Value': [income, age, debt, predicted_credit_score, employment_length, 
                            married, dependents, education, home_ownership, risk_score]
                })
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Your Credit Analysis",
                    data=csv,
                    file_name='credit_risk_analysis.csv',
                    mime='text/csv'
                )
                # --- Credit Risk Analyzer Power-Up/Down Notification ---
                if risk_level == "Low":
                    st.markdown(f"""
                    <div style="{card_style} text-align: center; margin: 10px 0;">
                        <h3 style="color: #000; margin: 0;">🍄 Power-Up! You have a LOW credit risk. Mario would be proud!</h3>
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "Medium":
                    st.warning("⭐ Be Careful! You have a MEDIUM credit risk. Collect more coins!", icon="⭐")
                else:
                    st.error("💣 Power-Down! You have a HIGH credit risk. Watch out for Bowser!", icon="💣")

# -------------- Module 2: Lifestyle Clusterer ----------------
elif page == "⭐ Lifestyle Clusterer":
    st.header("⭐ Lifestyle Clusterer")
    uploaded_cluster = st.file_uploader("Upload spending behavior data", type=["csv"])
    if uploaded_cluster:
        df = pd.read_csv(uploaded_cluster)
    else:
        df = pd.DataFrame({
            'Groceries': np.random.randint(500, 2000, 100),
            'Dining': np.random.randint(100, 1000, 100),
            'Entertainment': np.random.randint(100, 1500, 100),
            'Savings': np.random.randint(500, 3000, 100)
        })
    st.dataframe(df.head())
    if st.button("Cluster Users"):
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
        inertia = []
        for k in range(1, 6):
            km = KMeans(n_clusters=k, random_state=42)
            km.fit(df_scaled)
            inertia.append(km.inertia_)
        fig = px.line(x=list(range(1, 6)), y=inertia, labels={'x': 'k', 'y': 'Inertia'}, title="Elbow Method")
        st.plotly_chart(fig)
        km = KMeans(n_clusters=3)
        clusters = km.fit_predict(df_scaled)
        df['Cluster'] = clusters
        st.markdown(f"""
        <div style="{card_style} text-align: center; margin: 10px 0;">
            <h3 style="color: #000;">🍄 Users Clustered Successfully!</h3>
        </div>
        """, unsafe_allow_html=True)
        # Store models in session state
        st.session_state['lifestyle_scaler'] = scaler
        st.session_state['lifestyle_km'] = km
        st.session_state['lifestyle_df'] = df
        # Cluster summary
        cluster_names = {0: "Frugal 🍄", 1: "Moderate ⭐", 2: "Lavish 💰"}
        cluster_summary = df.groupby('Cluster').mean().round(0)
        st.markdown(f"<div style='{card_style} margin-top:10px;'><b>Cluster Averages:</b></div>", unsafe_allow_html=True)
        st.dataframe(cluster_summary)
        st.markdown(f"<div style='{card_style} margin-top:10px;'><b>Cluster Legend:</b> 0 = Frugal 🍄, 1 = Moderate ⭐, 2 = Lavish 💰</div>", unsafe_allow_html=True)
    # User input for real-time analysis
    st.markdown(f"<div style='{card_style} margin-top:10px;'><b>Find Your Lifestyle Cluster:</b></div>", unsafe_allow_html=True)
    g = st.number_input("Your Groceries Spending", min_value=0, value=1000)
    d = st.number_input("Your Dining Spending", min_value=0, value=500)
    e = st.number_input("Your Entertainment Spending", min_value=0, value=500)
    s = st.number_input("Your Savings", min_value=0, value=1000)
    if st.button("Analyze My Lifestyle", key="analyze_life"):
        if 'lifestyle_scaler' in st.session_state and 'lifestyle_km' in st.session_state:
            user_scaled = st.session_state['lifestyle_scaler'].transform([[g, d, e, s]])
            user_cluster = st.session_state['lifestyle_km'].predict(user_scaled)[0]
            cluster_names = {0: "Frugal 🍄", 1: "Moderate ⭐", 2: "Lavish 💰"}
            st.markdown(f"""
            <div style='{card_style} text-align: center; margin: 10px 0;'>
                <h3 style='color: #000;'>You belong to the <b>{cluster_names[user_cluster]}</b> group!</h3>
            </div>
            """, unsafe_allow_html=True)
            if user_cluster == 0:
                st.markdown(f"<div style='{card_style} text-align: center; margin: 10px 0;'><h4 style='color: #000;'>Power-Up! You are frugal like Mario collecting coins! 🍄</h4></div>", unsafe_allow_html=True)
            elif user_cluster == 1:
                st.markdown(f"<div style='{card_style} text-align: center; margin: 10px 0;'><h4 style='color: #000;'>Balanced! You are a moderate spender. ⭐</h4></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='{card_style} text-align: center; margin: 10px 0;'><h4 style='color: #000;'>Power-Down! You are living lavishly. Watch out for Bowser's bills! 💣</h4></div>", unsafe_allow_html=True)
            # Show cluster plot if available
            if 'lifestyle_df' in st.session_state:
                cluster_fig = px.scatter(st.session_state['lifestyle_df'], x='Groceries', y='Savings', color='Cluster', title="Lifestyle Clusters",
                                        color_discrete_map={0: "#43aa8b", 1: "#ffd60a", 2: "#e63946"})
                st.plotly_chart(cluster_fig)
        else:
            st.markdown(f"<div style='{card_style} text-align: center; margin: 10px 0;'><h4 style='color: #000;'>Please cluster users first!</h4></div>", unsafe_allow_html=True)

# -------------- Module 3: Smart Budget Allocator ----------------
elif page == "💰 Smart Budget Allocator":
    st.header("💰 Smart Budget Allocator")
    uploaded_budget = st.file_uploader("Upload income and spending data", type=["csv"])
    if uploaded_budget:
        df = pd.read_csv(uploaded_budget)
    else:
        df = pd.DataFrame({
            'Income': np.random.randint(30000, 150000, 100),
            'FamilySize': np.random.randint(1, 6, 100),
            'Needs': np.random.randint(10000, 70000, 100),
            'Wants': np.random.randint(5000, 30000, 100),
            'Savings': np.random.randint(2000, 40000, 100)
        })
    st.dataframe(df.head())
    if st.button("Predict Budget Allocation"):
        df['Total'] = df['Needs'] + df['Wants'] + df['Savings']
        X = df[['Income', 'FamilySize']]
        y = df[['Needs', 'Wants', 'Savings']]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        st.markdown(f"""
        <div style="{card_style} text-align: center; margin: 10px 0;">
            <h3 style="color: #000; margin: 0;">🍄 Prediction Complete</h3>
        </div>
        """, unsafe_allow_html=True)
        st.write("R² Score:", r2_score(y, y_pred))
        st.write("MAE:", mean_absolute_error(y, y_pred))
        # Mario-style feedback
        st.markdown(f"<div style='{card_style} margin-top:10px;'><b>Mario Budget Advice:</b></div>", unsafe_allow_html=True)
        avg_savings = np.mean(y_pred[:,2])
        if avg_savings > 20000:
            st.markdown(f"<div style='{card_style} text-align: center; margin: 10px 0;'><h4 style='color: #000;'>🍄 Power-Up! You're saving like Mario! Keep collecting those coins.</h4></div>", unsafe_allow_html=True)
        elif avg_savings > 10000:
            st.markdown(f"<div style='{card_style} text-align: center; margin: 10px 0;'><h4 style='color: #000;'>⭐ Not bad! But you can save even more to reach the next level.</h4></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='{card_style} text-align: center; margin: 10px 0;'><h4 style='color: #000;'>💣 Power-Down! Your savings are low. Watch out for Bowser's bills!</h4></div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Actual Needs', x=df.index, y=df['Needs']))
        fig.add_trace(go.Bar(name='Predicted Needs', x=df.index, y=y_pred[:, 0]))
        fig.update_layout(barmode='group', title="Actual vs Predicted Budget", plot_bgcolor=main_bg)
        st.plotly_chart(fig)

# ------------------- End -------------------
st.markdown("---")
st.markdown(f"""
<div style="{card_style} text-align: center; margin: 10px 0;">
    <p style="color: #000; margin: 0;">Developed for AF3005 - Programming for Finance | FAST NUCES Islamabad</p>
</div>
""", unsafe_allow_html=True)
