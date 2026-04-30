# Data Dictionary - Aadhaar 360 Dataset

## Overview
This document describes all fields in the processed Aadhaar analytics dataset.

---

## Master Analytics File
**File**: `aadhaar_district_analytics_final_cleaned.csv`

### Geographic Identifiers

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `State` | String | State/UT name (standardized) | West Bengal, Telangana |
| `District` | String | District name (standardized per MHA) | Kolkata, Hyderabad |
| `Region` | String | Geographic region for grouping | Eastern, Southern |

### Enrollment Metrics

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `Total_Enrollment` | Integer | Total Aadhaar enrollments ever | 5,234,567 |
| `Enrollment_0_5` | Integer | Enrollments age 0-5 years | 4,815,234 |
| `Enrollment_5_17` | Integer | Enrollments age 5-17 years | 312,445 |
| `Enrollment_18_Plus` | Integer | Enrollments age 18+ years | 106,888 |
| `Enrollment_Growth_Rate` | Float | YoY enrollment growth (%) | 2.3 |

### Update Metrics

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `Total_Updates` | Integer | Total Aadhaar record updates | 1,234,567 |
| `Demographic_Updates` | Integer | Online/document updates (name, address) | 567,890 |
| `Biometric_Updates` | Integer | Physical fingerprint/iris updates | 445,432 |
| `Child_Biometric_Updates` | Integer | Mandatory child updates (age 5-17) | 234,567 |
| `Adult_Demographic_Updates` | Integer | Voluntary adult demographic updates | 312,456 |
| `Adult_Biometric_Updates` | Integer | Voluntary adult biometric updates | 132,988 |
| `Update_Growth_Rate` | Float | YoY update growth (%) | 4.1 |

### Operational Intelligence Features

| Field | Type | Description | Range | Interpretation |
|-------|------|-------------|-------|-----------------|
| `UER_Score` | Float | Update-to-Enrollment Ratio | 0.0 - 2.0 | >0.8 = Saturated Market |
| `CV_Volatility` | Float | Coefficient of Variation (daily transactions) | 0.0 - 1.0 | >0.6 = High volatility (migration hub) |
| `Digital_Index` | Float | Demographic / (Demographic + Biometric) | 0.0 - 1.0 | >0.6 = Digital-first population |
| `Adult_Entry_Rate` | Float | (Adult_Updates / Total_Updates) * 100 | 0.0 - 100 | >30% = Fraud risk indicator |
| `Child_Mandate_Adherence` | Float | Child_Biometric / (Expected from births) | 0.0 - 1.0 | >0.85 = Good compliance |

### ML Clustering Output

| Field | Type | Description | Values |
|-------|------|-------------|--------|
| `Cluster_ID` | Integer | K-Means cluster assignment | 0, 1, 2, 3 |
| `Cluster_Name` | String | Human-readable cluster label | High Growth, Maintenance Hub, Volatile Zone, Fraud Risk |
| `Cluster_Recommendation` | String | Actionable recommendation | Deploy Enrollment Kits, Deploy Mobile Vans, Audit Required |
| `Confidence_Score` | Float | Cluster assignment confidence | 0.0 - 1.0 |

### Derived Metrics

| Field | Type | Description | Formula |
|-------|------|-------------|---------|
| `Grand_Total` | Integer | Total Enrollment + Total Updates | Enrollment + Updates |
| `Catch_Up_Index` | Float | (Updates - Enrollments) / Enrollments | Measures maintenance phase intensity |
| `Performance_Rank` | Integer | District ranking by total activity | 1-750 (1 = highest) |

---

## Time-Series File
**File**: `aadhaar_monthly_district_trends.csv`

### Temporal Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `Year` | Integer | Calendar year | 2024 |
| `Month` | Integer | Calendar month (1-12) | 6 (June) |
| `Date` | Date | First day of month | 2024-06-01 |
| `Quarter` | String | Business quarter | Q2 2024 |

### Monthly Metrics

| Field | Type | Description |
|-------|------|-------------|
| `Monthly_Enrollments` | Integer | New enrollments that month |
| `Monthly_Updates` | Integer | Total updates that month |
| `Monthly_Demographic` | Integer | Demographic updates that month |
| `Monthly_Biometric` | Integer | Biometric updates that month |
| `Monthly_Active_Centers` | Integer | Number of active enrollment centers |

### Trend Indicators

| Field | Type | Description | Interpretation |
|-------|------|-------------|-----------------|
| `MoM_Growth_Rate` | Float | Month-over-month growth (%) | Positive = increasing activity |
| `YoY_Growth_Rate` | Float | Year-over-year growth (%) | Seasonal adjustment |
| `Trend_Direction` | String | Increasing/Decreasing/Stable | Based on 3-month rolling average |

### Forecast Fields (Optional)

| Field | Type | Description |
|-------|------|-------------|
| `Forecasted_Enrollments_Next_Month` | Integer | Predicted enrollments (ARIMA/Prophet) |
| `Forecasted_Updates_Next_Month` | Integer | Predicted updates (ARIMA/Prophet) |
| `Forecast_Confidence_Interval` | String | 95% CI bounds |

---

## Cluster Definitions

### Cluster 0: High Growth Zones
- **Characteristics**:
  - Low UER Score (<0.4)
  - Growing enrollment rate (>5% YoY)
  - Adult Entry Rate <15% (low fraud risk)
- **Recommendation**: Deploy expensive enrollment kits
- **Example States**: Rural areas, newly developed regions
- **Resource Priority**: High

### Cluster 1: Maintenance Hubs
- **Characteristics**:
  - High UER Score (>0.8)
  - Stable/declining enrollment
  - High biometric update concentration
- **Recommendation**: Deploy update laptops and iris scanners
- **Example Districts**: Major urban centers, established markets
- **Resource Priority**: Medium

### Cluster 2: Volatile/Migrant Zones
- **Characteristics**:
  - High CV Volatility (>0.6)
  - Inconsistent monthly patterns
  - Mixed age group updates
- **Recommendation**: Deploy mobile vans and temporary facilities
- **Example Regions**: Border areas, industrial hubs, metro outskirts
- **Resource Priority**: Medium

### Cluster 3: Fraud Risk Zones
- **Characteristics**:
  - Adult Entry Rate >30%
  - Unusual update patterns
  - Low cluster confidence score
- **Recommendation**: Initiate audit and investigation
- **Example Cases**: Ghost villages, data entry centers
- **Resource Priority**: Critical

---

## Data Quality Notes

### Missing Values
- Handled via forward-fill (temporal data) or median imputation
- <0.5% missing data across all fields

### Outliers
- Detected using IQR method (Q1 - 1.5×IQR to Q3 + 1.5×IQR)
- Extreme outliers flagged but retained (may indicate real anomalies)

### Validation Rules
1. **Enrollment Constraints**: Total = Sum(0-5, 5-17, 18+)
2. **Update Constraints**: Total = Demographic + Biometric
3. **Growth Rates**: -50% < Growth_Rate < +200% (allows for legitimate spikes)
4. **Cluster Coverage**: 100% of districts assigned to exactly one cluster

---

## Dimension Ranges

| Metric | Min | Max | Mean | Notes |
|--------|-----|-----|------|-------|
| Total_Enrollment | 5,000 | 50M | 3.2M | Ranges widely by district size |
| UER_Score | 0.02 | 2.1 | 0.45 | Higher in maintenance phase |
| CV_Volatility | 0.08 | 0.92 | 0.38 | Indicates operational stability |
| Adult_Entry_Rate | 2% | 68% | 12% | Fraud risk if >30% |
| Digital_Index | 0.12 | 0.95 | 0.48 | Varies by digital literacy |

---

## Processing Transformations

### State Name Standardization (50+ Mappings)
```
"Westbengal" → "West Bengal"
"Telanana" → "Telangana"
"Orissa" → "Odisha"
"Uttaranchal" → "Uttarakhand"
[... and 46 more corrections]
```

### District Realignment (100+ Updates)
```
Telangana/Andhra Pradesh: 33 districts reassigned
Ladakh Separation: Leh, Kargil moved from J&K
West Bengal: Burdwan → Purba Bardhaman split
[... and 97 more corrections]
```

### Nomenclature Updates
```
Aurangabad → Chhatrapati Sambhaji Nagar
Osmana bad → Dharashiv
Gurgaon → Gurugram
Allahabad → Prayagraj
```

---

## Usage in Dashboard

### Admin Dashboard Queries
```python
# Find high-growth districts
df[df['Cluster_Name'] == 'High Growth'].sort_values('Enrollment_Growth_Rate', ascending=False)

# Identify fraud risk zones
df[df['Adult_Entry_Rate'] > 0.30].sort_values('Adult_Entry_Rate', ascending=False)

# Compare two districts
df[df['District'].isin(['Kolkata', 'Mumbai'])]
```

### Citizen Portal Queries
```python
# Get center recommendations
district_data = df[df['District'] == user_input_district]
if district_data['Digital_Index'].values[0] > 0.6:
    recommendation = "Go Online"
else:
    recommendation = "Visit Center"
```

---

## Related Documentation
- **Technical Architecture**: See `ARCHITECTURE.md`
- **Methodology**: See `METHODOLOGY.md`
- **Data Cleaning Pipeline**: See `src/data_cleaning.py`
