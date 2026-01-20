import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_correlations(filename='updated_pt3_data.csv'):
    print(f"Loading data from {filename}...")
    
    try:
        # 1. Load the Data
        df = pd.read_csv(filename)
        
        # 2. Filter for Numeric Columns Only
        # (Correlation can only be calculated on numbers)
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        
        if numeric_df.empty:
            print("No numeric data found to correlate.")
            return

        # 3. Calculate Correlation Matrix
        corr_matrix = numeric_df.corr()
        
        print("\nCorrelation Matrix (Top 5 Rows):")
        print(corr_matrix.head())

        # 4. Generate Heatmap Visualization
        plt.figure(figsize=(14, 12))
        sns.heatmap(
            corr_matrix, 
            annot=True,       # Show numbers in the boxes
            fmt=".2f",        # Format to 2 decimal places
            cmap='coolwarm',  # Red = Positive, Blue = Negative correlation
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Correlation Matrix: Relationships between Aadhaar Metrics', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 5. Save the Plot
        output_file = 'vis_correlation_heatmap.png'
        plt.savefig(output_file)
        print(f"\n[Success] Correlation heatmap saved as '{output_file}'")
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Execution ---
if __name__ == "__main__":
    analyze_correlations('aadhaar_district_analytics_final_cleaned.csv')