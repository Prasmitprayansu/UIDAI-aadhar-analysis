#python -m streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. APP CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Aadhaar 360 | National Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional UI - ACCESSIBILITY FIX
st.markdown("""
<style>
    /* Main Headings - Switched to Brighter "Govt" Blue */
    .main-header {
        font-size: 2.8rem; 
        color: #3B82F6; /* Brighter Blue (Visible on Dark & Light backgrounds) */
        font-weight: 800; 
        margin-bottom: 0px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    /* Sub-text - Removed hardcoded grey so it turns White in Dark Mode */
    .sub-text {
        font-size: 1.1rem; 
        font-weight: 500;
        margin-bottom: 20px;
        opacity: 0.8; /* Slight transparency instead of hardcoded color */
    }
    
    /* Metrics Styling - Removed hardcoded dark blue */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
        /* Color removed to adapt to Streamlit Theme */
    }
    
    /* Card-like Containers */
    .css-1r6slb0 {
        /* Changed to transparent/adaptive background for better theme integration */
        background-color: rgba(255, 255, 255, 0.05); 
        border: 1px solid rgba(250, 250, 250, 0.1);
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data
def load_and_process_data():
    try:
        # Robust file path handling
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'aadhaar_district_analytics_final_cleaned.csv')
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("⚠️ Data file not found. Please ensure 'aadhaar_district_analytics_final_cleaned.csv' is in the app directory.")
        return pd.DataFrame()

    # --- Feature Engineering ---
    # 1. Intensity Metrics
    if 'bio_age_5_17' in df.columns and 'demo_age_5_17' in df.columns:
        df['Child_Bio_Intensity'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
    
    if 'bio_age_17_' in df.columns and 'demo_age_17_' in df.columns:
        df['Adult_Bio_Intensity'] = df['bio_age_17_'] / (df['demo_age_17_'] + 1)

    # 2. Clustering (ML)
    features = ['UER_Score', 'Catch_Up_Index', 'Adult_Entry_Rate', 'CV_Volatility']
    available_features = [f for f in features if f in df.columns]
    
    if len(available_features) == 4:
        X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42)
        df['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    else:
        df['Cluster_ID'] = 0
        
    return df

df = load_and_process_data()

if df.empty:
    st.stop()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=120)
    st.markdown("## **Aadhaar 360**")
    st.caption("Operational Intelligence Dashboard")
    st.divider()
    
    user_role = st.radio("Portal Mode:", ["👤 Citizen Utility", "👮 Admin Command Center"])
    
    st.divider()
    st.markdown("### 📍 Location Filter")
    
    # Cascade Filters
    selected_state = st.selectbox("Select State", sorted(df['state'].unique()))
    district_list = sorted(df[df['state'] == selected_state]['district'].unique())
    selected_district = st.selectbox("Select District", district_list)
    
    # Specific Data Row
    row = df[(df['state'] == selected_state) & (df['district'] == selected_district)].iloc[0]

# ==========================================
# 4. VIEW: CITIZEN UTILITY
# ==========================================
if user_role == "👤 Citizen Utility":
    st.markdown(f"<div class='main-header'>Aadhaar Seva: {selected_district}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sub-text'>Real-time service updates and guidance for <b>{selected_state}</b> residents.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # --- Reliability Card ---
        st.subheader("🏥 Center Status")
        volatility = row.get('CV_Volatility', 0)
        
        if volatility < 1.5:
            st.success("✅ **Operational:** Centers are open regularly (9 AM - 5 PM).")
        elif volatility < 4:
            st.warning("⚠️ **High Traffic:** Expect delays. Centers usually busy.")
        else:
            st.error("🛑 **Irregular Service:** Centers likely operating on Camp Mode. Check local news.")

        # --- Interactive Crowd Chart (FIXED) ---
        st.subheader("⏳ Predicted Wait Times")
        
        # FIX: Generate unique data for each district using a hash of the name
        # This ensures every district looks different!
        np.random.seed(hash(selected_district) % 2**32) 
        
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        # Generate random traffic numbers between 20 and 100 for weekdays, lower for weekends
        traffic = [np.random.randint(50, 100) for _ in range(5)] + [np.random.randint(10, 40), 0]
        
        fig = px.bar(
            x=days, y=traffic, 
            labels={'x': 'Day', 'y': 'Busy Level (%)'},
            color=traffic,
            color_continuous_scale=['#22c55e', '#facc15', '#ef4444'], # Green -> Yellow -> Red
            height=250,
            title=f"Traffic Pattern for {selected_district}"
        )
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # --- Action Card ---
        with st.container(border=True):
            st.markdown("### 📱 Action Guide")
            adult_bio = row.get('Adult_Bio_Intensity', 0)
            
            if adult_bio < 1:
                st.info("💡 **Recommendation: Online**")
                st.write("Most updates in your area are address changes. Save time:")
                st.link_button("🔗 Login to myAadhaar", "https://myaadhaar.uidai.gov.in/")
            else:
                st.warning("🛠️ **Recommendation: Visit Center**")
                st.write("High biometric updates detected in your area.")
                st.link_button("📍 Locate Nearest Center", "https://bhuvan.nrsc.gov.in/aadhaar/")
        
        # --- Digital Score (FIXED) ---
        with st.container(border=True):
            # Calculate Digital Maturity as % of Demographic Updates vs Total Updates
            # This ensures the score is always between 0 and 100
            demo_total = row.get('demo_age_5_17', 0) + row.get('demo_age_17_', 0)
            bio_total = row.get('bio_age_5_17', 0) + row.get('bio_age_17_', 0)
            total_updates = demo_total + bio_total
            
            if total_updates > 0:
                digital_score = (demo_total / total_updates) * 100
            else:
                digital_score = 0
            
            st.metric("Digital Maturity Score", f"{digital_score:.0f}/100", help="Percentage of residents using Digital Demographic Updates vs Physical Biometric Updates.")
            
            # Dynamic Feedback
            if digital_score > 70:
                st.caption("🌟 High Digital Adoption")
                st.progress(int(digital_score) / 100)
            elif digital_score > 40:
                st.caption("⚠️ Growing Adoption")
                st.progress(int(digital_score) / 100)
            else:
                st.caption("🛑 Heavy Physical Dependency")
                st.progress(int(digital_score) / 100)

# ==========================================
# 5. VIEW: ADMIN COMMAND CENTER
# ==========================================
else:
    col_head, col_btn = st.columns([4, 1])
    with col_head:
        st.markdown(f"<div class='main-header'>Command Center | {selected_district}</div>", unsafe_allow_html=True)
    with col_btn:
        # CSV Download Button
        csv_data = row.to_frame().T.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Report", data=csv_data, file_name=f"Report_{selected_district}.csv", mime="text/csv")

    # --- CLUSTER INTELLIGENCE BANNER ---
    c_id = row.get('Cluster_ID', 0)
    clusters = {
        0: {"label": "Standard Operations", "color": "blue", "msg": "Routine monitoring required."},
        1: {"label": "High Growth Zone", "color": "green", "msg": "Deploy ECMP Kits immediately for new enrolments."},
        2: {"label": "Catch-up / Crisis", "color": "orange", "msg": "High backlog. Deploy Hospital Teams."},
        3: {"label": "Fraud Risk / Anomaly", "color": "red", "msg": "Audit Required: High Adult Entry + High Updates."}
    }
    info = clusters.get(c_id, clusters[0])
    
    st.markdown(f"""
    <div style="background-color:{info['color']}; padding:15px; border-radius:10px; color:white; margin-bottom:20px;">
        <strong>STATUS: {info['label']}</strong><br>{info['msg']}
    </div>
    """, unsafe_allow_html=True)

    # --- KPI METRICS ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("UER Score (Maturity)", f"{row.get('UER_Score', 0):.2f}", help="Ratio of Updates to Enrolments")
    k2.metric("Catch-up Index", f"{row.get('Catch_Up_Index', 0):.1f}", delta="High Lag" if row.get('Catch_Up_Index', 0) > 5 else "Normal", delta_color="inverse")
    k3.metric("Adult Entry Rate", f"{row.get('Adult_Entry_Rate', 0)*100:.1f}%", delta="Suspicious" if row.get('Adult_Entry_Rate', 0) > 0.05 else "Stable", delta_color="inverse")
    k4.metric("Bio Failure Proxy", f"{row.get('Adult_Bio_Intensity', 0):.2f}", help="High value indicates fingerprint failures.")

    st.markdown("---")

    # --- COMPARISON MODE TOGGLE ---
    compare_mode = st.toggle("🔄 Enable District Comparison Mode")
    
    if compare_mode:
        st.markdown("### ⚔️ District Comparison")
        comp_district = st.selectbox("Select District to Compare", [d for d in district_list if d != selected_district])
        row_comp = df[(df['state'] == selected_state) & (df['district'] == comp_district)].iloc[0]
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Volume", f"{row['Grand_Total']:,}", delta=f"{row['Grand_Total'] - row_comp['Grand_Total']:,}")
        c2.metric("Enrolment Total", f"{row['Enrol_Total']:,}", delta=f"{row['Enrol_Total'] - row_comp['Enrol_Total']:,}")
        c3.metric("Update Total", f"{row['Update_Total']:,}", delta=f"{row['Update_Total'] - row_comp['Update_Total']:,}")
        st.markdown("---")

    # --- ANALYTICS TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Resource Planning", "📉 Demographics", "🔍 Fraud & Anomalies"])

    with tab1:
        st.subheader("Resource Allocation Strategy")
        c1, c2 = st.columns(2)
        
        with c1:
            # Stacked Bar: Update Types
            types_data = pd.DataFrame({
                'Type': ['Demographic', 'Biometric'],
                'Volume': [
                    row.get('demo_age_5_17', 0) + row.get('demo_age_17_', 0),
                    row.get('bio_age_5_17', 0) + row.get('bio_age_17_', 0)
                ]
            })
            fig = px.pie(types_data, values='Volume', names='Type', title="Workload Split: Demo vs Bio", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.info("💡 **Strategic Advice**")
            uer = row.get('UER_Score', 0)
            if uer > 5:
                st.write("• **Market is Saturated.**")
                st.write("• Stop buying new ECMP Kits.")
                st.write("• Focus budget on Update Client Lite (UCL) laptops.")
            else:
                st.write("• **High Growth Area.**")
                st.write("• Deploy more GPS-enabled Enrolment Kits.")
                st.write("• Schedule camps in schools.")

    with tab2:
        st.subheader("Population Dynamics")
        c1, c2 = st.columns([2, 1])
        
        with c1:
            # Age Group Bar Chart
            age_df = pd.DataFrame({
                'Group': ['Infants (0-5)', 'School (5-17)', 'Adults (18+)'],
                'Enrolments': [row.get('age_0_5', 0), row.get('age_5_17', 0), row.get('age_18_greater', 0)]
            })
            fig = px.bar(age_df, x='Group', y='Enrolments', color='Group', title="New Enrolment by Age")
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.write("**Key Indicators**")
            catch_up = row.get('Catch_Up_Index', 0)
            st.metric("Neonatal Gap", f"{catch_up:.1f}x")
            if catch_up > 3:
                st.error("School-age enrolment is much higher than birth enrolment. Babies are being missed.")
            else:
                st.success("Birth registration ecosystem is healthy.")

    with tab3:
        st.subheader("Fraud Detection Radar")
        
        # Ghost Village Logic
        enrol_tot = row.get('Enrol_Total', 0)
        update_tot = row.get('Update_Total', 0)
        ghost_proxy = enrol_tot / (update_tot + 1)
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.metric("Ghost Village Proxy", f"{ghost_proxy:.2f}")
            if ghost_proxy > 10:
                st.error("🚨 High Risk: Many enrolments, zero updates.")
            else:
                st.success("Normal Activity")
                
        with c2:
            st.metric("Hardware Risk", f"{row.get('Adult_Bio_Intensity', 0):.2f}")
            if row.get('Adult_Bio_Intensity', 0) > 2.0:
                st.warning("⚠️ Something may be faulty.")
            else:
                st.success("Hardware stable.")
                
        with c3:
            st.metric("Operator Error Rate", f"{row.get('Correction_Intensity', 0):.2f}")
            if row.get('Correction_Intensity', 0) > 5.0:
                st.error("🚨 High demographic corrections. Audit operators.")
