import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set visual aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_datasets(base_path="."):
    files_map = {
        'Enrolment': 'api_data_aadhar_enrolment_*.csv',
        'Demographic': 'api_data_aadhar_demographic_*.csv',
        'Biometric': 'api_data_aadhar_biometric_*.csv'
    }
    
    datasets = {}
    print("Loading datasets...")
    for category, pattern in files_map.items():
        full_pattern = os.path.join(base_path, pattern)
        files = glob.glob(full_pattern)
        
        df_list = []
        for file in files:
            try:
                temp = pd.read_csv(file)
                df_list.append(temp)
            except Exception as e:
                print(f"Skipped {file}: {e}")
        
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
                # Date Features
                df['Month'] = df['date'].dt.month_name()
            datasets[category] = df
            print(f"   -> Loaded {category}: {len(df)} records")
        else:
            print(f"   -> No files found for {category}")
            datasets[category] = pd.DataFrame() 

    return datasets['Enrolment'], datasets['Demographic'], datasets['Biometric']

# ==========================================
# 2. COMPREHENSIVE CLEANING FUNCTION
# ==========================================
def clean_data(df):
    if df.empty: return df
    
    # 1. REMOVE NON-ENGLISH CHARACTERS
    if 'district' in df.columns:
        df['district'] = df['district'].astype(str).replace(r'[^\x00-\x7F]+', '', regex=True)
        df['district'] = df['district'].str.replace('*', '', regex=False).str.strip()
        df.loc[df['district'] == 'ManendragarhChirmiriBharatpur', 'district'] = 'Manendragarh-Chirmiri-Bharatpur'

    # 2. STATE MAPPING
    state_map = {
        'WEST BENGAL': 'West Bengal', 'WESTBENGAL': 'West Bengal', 
        'West bengal': 'West Bengal', 'Westbengal': 'West Bengal', 
        'West Bengli': 'West Bengal', 'west Bengal': 'West Bengal',
        'West  Bengal': 'West Bengal', 'West Bangal': 'West Bengal',
        'odisha': 'Odisha', 'ODISHA': 'Odisha', 'Orissa': 'Odisha',
        'andhra pradesh': 'Andhra Pradesh',
        'Andaman & Nicobar Islands': 'Andaman and Nicobar Islands',
        'Dadra & Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Dadra and Nagar Haveli': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman & Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'Daman and Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'The Dadra And Nagar Haveli And Daman And Diu': 'Dadra and Nagar Haveli and Daman and Diu',
        'Pondicherry': 'Puducherry', 'Uttaranchal': 'Uttarakhand',
        'Tamilnadu': 'Tamil Nadu', 'Chhatisgarh': 'Chhattisgarh',
        'Telanana': 'Telangana', 
        '100000': 'Unknown',
        'Jaipur': 'Rajasthan', 'Nagpur': 'Maharashtra', 'Darbhanga': 'Bihar',
        'Madanapalle': 'Andhra Pradesh', 'BALANAGAR': 'Telangana',
        'Puttenahalli': 'Karnataka', 'Raja Annamalai Puram': 'Tamil Nadu'
    }
    
    if 'state' in df.columns:
        df['state'] = df['state'].replace(state_map)

    # 3. STATE RE-ASSIGNMENT
    # Fix Kamrup (Meghalaya -> Assam)
    df.loc[(df['state'] == 'Meghalaya') & (df['district'] == 'Kamrup'), 'state'] = 'Assam'

    # Fix Telangana districts
    ts_districts = [
        'Adilabad', 'Hyderabad', 'K.V RANGAEEDDY', 'Karimnagar', 'Khammam', 
        'Mahabubnagar', 'Medak', 'Nalgonda', 'Nizamabad', 'Rangareddi', 
        'Warangal', 'Warangal(urban)', 'Warangal(rural)', 'Jagitial', 
        'Jangaon', 'Jayashankar Bhupalpally', 'Jogulamba Gadwal', 
        'Kamareddy', 'Komaram Bheem Asifabad', 'Mahabubabad', 'Mancherial', 
        'Medchal–Malkajgiri', 'Mulugu', 'Nagarkurnool', 'Narayanpet', 
        'Nirmal', 'Peddapalli', 'Rajanna Sircilla', 'Sangareddy', 
        'Siddipet', 'Suryapet', 'Vikarabad', 'Wanaparthy', 'Yadadri'
    ]
    df.loc[(df['state'] == 'Andhra Pradesh') & (df['district'].isin(ts_districts)), 'state'] = 'Telangana'
    
    # Fix Others
    df.loc[(df['state'] == 'Chandigarh') & (df['district'].isin(['Mohali', 'Rupnagar'])), 'state'] = 'Punjab'
    df.loc[(df['state'] == 'Puducherry') & (df['district'].isin(['Cuddalore', 'Viluppuram'])), 'state'] = 'Tamil Nadu'
    df.loc[(df['state'] == 'Jammu and Kashmir') & (df['district'].isin(['Leh', 'Kargil'])), 'state'] = 'Ladakh'

    # 4. DISTRICT RENAMING
    district_map = {
        # --- ANDHRA PRADESH ---
        'Anantapur': 'Ananthapuramu', 'Ananthapur': 'Ananthapuramu',
        'Kadiri Road': 'Sri Sathya Sai',
        'Nellore': 'Sri Potti Sriramulu Nellore', 'Spsr Nellore': 'Sri Potti Sriramulu Nellore',
        'Cuddapah': 'YSR Kadapa', 'Chittoor': 'Chittoor', 'Visakhapatnam': 'Visakhapatanam',
        
        # --- WEST BENGAL ---
        'Bardhaman': 'Purba Bardhaman', 'Burdwan': 'Purba Bardhaman',
        'CoochBehar': 'Cooch Behar', 'Coochbehar': 'Cooch Behar',
        'South Dinajpur': 'Dakshin Dinajpur', 'North Dinajpur': 'Uttar Dinajpur',
        'Domjur': 'Howrah', 'East Midnapore': 'Purba Medinipur',
        'West Medinipur': 'Paschim Medinipur', 'Medinipur': 'Paschim Medinipur',
        'Medinipur West': 'Paschim Medinipur', 'Naihati Anandbazar': 'North 24 Parganas',
        'Naihati Anandabazar': 'North 24 Parganas', 'Nortg 24 praganas': 'North 24 Parganas',
        'South 24 praganas': 'South 24 Parganas',
        
        # --- OTHERS ---
        'Nicobars': 'Nicobar', 
        'Shiyomi': 'Shi Yomi', 'Pakke-Kessang': 'Pakke Kessang', 'Kra-Daadi': 'Kra Daadi',
        'Kamrup Metropolitan': 'Kamrup Metro', 'South Salmara-Mankachar': 'South Salmara Mankachar',
        'Chhatrapati Sambhajinagar': 'Aurangabad',
        'Kaimur': 'Kaimur (Bhabhua)', 'Munger': 'Munger', 'Samastipur': 'Samastipur',
        'Purnia': 'Purnia', 'Purba Champaran': 'East Champaran', 
        'Near university': 'Unknown',
        'Dantewada': 'Dantewada', 'Janjgir-Champa': 'Janjgir-Champa',
        'Gaurela-Pendra-Marwahi': 'Gaurella-Pendra-Marwahi', 
        'Mohla-Manpur-Ambagarh Chowki': 'Mohla-Manpur-Ambagarh Chowki',
        'Ahmadabad': 'Ahmedabad', 'Banas Kantha': 'Banaskantha',
        'Panch Mahals': 'Panchmahal', 'Sabar Kantha': 'Sabarkantha',
        'Surendra Nagar': 'Surendranagar', 'Yamuna Nagar': 'Yamunanagar',
        'Lahaul and Spiti': 'Lahaul and Spiti', 'Lahul & Spiti': 'Lahaul and Spiti',
        'Bandipore': 'Bandipora', 'Bandipur': 'Bandipora', 
        'Poonch': 'Poonch', 'Punch': 'Poonch', 'Rajauri': 'Rajouri', 
        'Shopian': 'Shopian', 'Shupiyan': 'Shopian', 'Udhampur': 'Udhampur',
        'East Singhbum': 'East Singhbhum', 'Hazaribag': 'Hazaribagh',
        'Koderma': 'Kodarma', 'Pakaur': 'Pakur', 'Palamau': 'Palamu',
        'Sahebganj': 'Sahibganj', 'Seraikela-kharsawan': 'Seraikela Kharsawan',
        'Pashchimi Singhbhum': 'West Singhbhum',
        '5th cross': 'Bengaluru Urban', 'Bangalore': 'Bengaluru Urban',
        'Bengaluru': 'Bengaluru Urban', 'Bijapur': 'Vijayapura',
        'Chamarajanagar': 'Chamarajanagara', 'Chickmagalur': 'Chikkamagaluru',
        'Chikmagalur': 'Chikkamagaluru', 'Davanagere': 'Davangere',
        'Hasan': 'Hassan', 'Mysore': 'Mysuru', 'Ramanagar': 'Ramanagara',
        'Shimoga': 'Shivamogga', 'Tumkur': 'Tumakuru', 'Yadgir': 'Yadgir',
        'ANGUL': 'Angul', 'Boudh': 'Boudh', 'Jajpur': 'Jajpur',
        'Jagatsinghpur': 'Jagatsinghpur', 'Khordha': 'Khordha', 
        'Nabarangpur': 'Nabarangpur', 'Subarnapur': 'Subarnapur',
        'Muktsar': 'Sri Muktsar Sahib', 'Nawanshahr': 'Shaheed Bhagat Singh Nagar',
        'S.A.S Nagar': 'SAS Nagar', 'Ferozepur': 'Ferozepur',
        'Chittorgarh': 'Chittorgarh', 'Dholpur': 'Dholpur', 'Jalore': 'Jalore',
        'Jhunjhunu': 'Jhunjhunu', 'Near meera hospital': 'Unknown',
        'East Sikkim': 'Gangtok', 'North Sikkim': 'Mangan',
        'South Sikkim': 'Namchi', 'West Sikkim': 'Gyalshing',
        'Near Dhyana Ashram': 'Unknown', 'Thiruvallur': 'Tiruvallur',
        'Thiruvarur': 'Tiruvarur', 'Tuticorin': 'Thoothukkudi',
        'Tirupattur': 'Tirupattur',
        'Aurangabad': 'Chhatrapati Sambhajinagar', 'Osmanabad': 'Dharashiv',
        'Ahmednagar': 'Ahilyanagar', 'Gurgaon': 'Gurugram',
        'Budaun': 'Badaun', 'Bulandshahar': 'Bulandshahr',
        'Mumbai( Sub Urban )': 'Mumbai Suburban',
        'Chhatrapati Sambhaji Nagar': 'Chhatrapati Sambhajinagar'
    }
    
    if 'district' in df.columns:
        df['district'] = df['district'].replace(district_map)
        df = df[df['district'] != 'Unknown']

    return df

# ==========================================
# 3. EXPORT MONTHLY TRENDS (This creates the file)
# ==========================================
def export_monthly_data(enrol, demo, bio):
    print("\n[Action] Generating Monthly Age-Group Time Series...")

    def aggregate_monthly(df, prefix, cols):
        if 'date' not in df.columns: return pd.DataFrame()
        temp = df.copy()
        temp['YearMonth'] = temp['date'].dt.to_period('M').astype(str)
        grouped = temp.groupby(['state', 'district', 'YearMonth'])[cols].sum().reset_index()
        new_cols = {c: f"{prefix}_{c}" for c in cols}
        grouped = grouped.rename(columns=new_cols)
        return grouped

    print("   -> Processing Enrolment trends...")
    e_monthly = aggregate_monthly(enrol, 'Enrol', ['age_0_5', 'age_5_17', 'age_18_greater'])
    
    print("   -> Processing Demographic trends...")
    d_monthly = aggregate_monthly(demo, 'Demo', ['demo_age_5_17', 'demo_age_17_'])
    
    print("   -> Processing Biometric trends...")
    b_monthly = aggregate_monthly(bio, 'Bio', ['bio_age_5_17', 'bio_age_17_'])
    
    master_ts = pd.merge(e_monthly, d_monthly, on=['state', 'district', 'YearMonth'], how='outer').fillna(0)
    master_ts = pd.merge(master_ts, b_monthly, on=['state', 'district', 'YearMonth'], how='outer').fillna(0)
    
    master_ts.to_csv('aadhaar_monthly_district_trends.csv', index=False)
    print(f"   -> Success! Saved monthly trends to 'aadhaar_monthly_district_trends.csv'")
    return master_ts

# ==========================================
# 4. METRIC CALCULATION & AGGREGATION
# ==========================================
def calculate_metrics(enrol, demo, bio):
    print("\nCalculating Analytical Metrics & Merging Districts...")
    
    # Aggregate to District Level (Removes Date)
    e_grp = enrol.groupby(['state', 'district'])[['age_0_5', 'age_5_17', 'age_18_greater']].sum()
    d_grp = demo.groupby(['state', 'district'])[['demo_age_5_17', 'demo_age_17_']].sum()
    b_grp = bio.groupby(['state', 'district'])[['bio_age_5_17', 'bio_age_17_']].sum()
    
    df = e_grp.join([d_grp, b_grp], how='outer').fillna(0).reset_index()
    
    # Calculate Volatility from Enrolment Data (since we have dates there)
    if 'date' in enrol.columns:
        daily = enrol.groupby(['state', 'district', 'date'])['age_5_17'].sum().reset_index()
        vol = daily.groupby(['state', 'district'])['age_5_17'].agg(['mean', 'std'])
        vol['CV_Volatility'] = (vol['std'] / (vol['mean'] + 0.1)).fillna(0)
        df = df.merge(vol['CV_Volatility'], on=['state', 'district'], how='left')

    # Totals
    df['Enrol_Total'] = df['age_0_5'] + df['age_5_17'] + df['age_18_greater']
    df['Update_Total'] = df['demo_age_5_17'] + df['demo_age_17_'] + df['bio_age_5_17'] + df['bio_age_17_']
    df['Grand_Total'] = df['Enrol_Total'] + df['Update_Total']
    
    # Metrics
    df['UER_Score'] = df['Update_Total'] / (df['Enrol_Total'] + 1)
    df['Adult_Entry_Rate'] = df['age_18_greater'] / (df['Enrol_Total'] + 1)
    df['Catch_Up_Index'] = df['age_5_17'] / (df['age_0_5'] + 1)
    
    # Behavioral Metrics
    df['R21_Child_Bio_Intensity'] = df['bio_age_5_17'] / (df['demo_age_5_17'] + 1)
    df['R22_Adult_Bio_Intensity'] = df['bio_age_17_'] / (df['demo_age_17_'] + 1)
    
    return df

# ==========================================
# 5. ML CLUSTERING
# ==========================================
def perform_clustering(df):
    print("Performing ML Clustering...")
    features = ['UER_Score', 'Catch_Up_Index', 'Adult_Entry_Rate', 'CV_Volatility']
    
    # Handle missing cols if volatility calc failed
    if 'CV_Volatility' not in df.columns: df['CV_Volatility'] = 0
    
    X = df[features].replace([np.inf, -np.inf], 0).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster_ID'] = kmeans.fit_predict(X_scaled)
    return df

# ==========================================
# FINAL EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Load Raw Data (This gets the Dates)
    enrol_df, demo_df, bio_df = load_datasets()
    
    # 2. Clean Data (Fixes Names)
    enrol_df = clean_data(enrol_df)
    demo_df = clean_data(demo_df)
    bio_df = clean_data(bio_df)
    
    # 3. Generate Monthly Trend File (CSV)
    export_monthly_data(enrol_df, demo_df, bio_df)
    
    # 4. Generate Aggregated Master File
    master_df = calculate_metrics(enrol_df, demo_df, bio_df)
    
    # 5. Run Machine Learning
    master_df = perform_clustering(master_df)
    
    # 6. Save Final Files
    master_df.to_csv('aadhaar_district_analytics_ML_final.csv', index=False)
    master_df.to_csv('aadhaar_district_analytics_final_cleaned.csv', index=False)
    
    print("\n[DONE] All files generated:")
    print("1. aadhaar_monthly_district_trends.csv (For Monthly Tabs)")
    print("2. aadhaar_district_analytics_final_cleaned.csv (For District Overview)")