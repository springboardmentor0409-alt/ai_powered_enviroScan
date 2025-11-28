import pandas as pd
import numpy as np

df = pd.read_csv('fixed_dates.csv')

# Replacing the Nan values with the Minimum values
min_value = df['fire_min_dist_km'].min()
df['fire_min_dist_km']=df['fire_min_dist_km'].fillna(min_value)

# Adding another Column Named as Season
df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
def get_season(date):
    month = date.month
    
    if month in [12, 1, 2]:
        return 'Winter'  # Dec, Jan, Feb
    elif month in [3, 4, 5]:
        return 'Spring'  # Mar, Apr, May
    elif month in [6, 7, 8]:
        return 'Summer'  # Jun, Jul, Aug
    else:
        return 'Autumn' # Sep, Oct, Nov
    
df['Season'] = df['date'].apply(get_season)

# Scaling factors (k values)
K_ROAD = 0.3
K_INDUSTRY = 1.5
K_FARM = 4.0

# Confidence Score Weights
W_POLLUTANT = 0.65
W_PROXIMITY = 0.30
W_FIRE = 0.05

def calculate_zscore(P_series):
    std_dev = P_series.std()
    if std_dev == 0 or pd.isna(std_dev): #if std 0 then return 0
        return 0
    return (P_series - P_series.mean()) / std_dev

def exponential_decay_proximity(d, k):
    return np.exp(-d / k)

# Normalization and Proximity Feature Engineering 

# List of columns to normalize 
pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3']
df_norm = df.copy() # Use a copy for intermediate normalized values

for col in pollutants:
    P = df[col]
    P_min = P.min()
    P_max = P.max()
    
    if P_max == P_min:
        # Use Z-score if all values are identical
        df_norm[f'{col}_norm'] = calculate_zscore(P)
    else:
        # Use Min-Max scaling
        df_norm[f'{col}_norm'] = (P - P_min) / (P_max - P_min)

# Calculate proximity scores
df_norm['prox_road'] = exponential_decay_proximity(df['dist_to_road'], K_ROAD)
df_norm['prox_industry'] = exponential_decay_proximity(df['dist_to_industry'], K_INDUSTRY)
df_norm['prox_farm'] = exponential_decay_proximity(df['dist_to_farm'], K_FARM)


# --- STEP 4: Rule-Based Label Assignment (Primary Output) ---

# Calculate the 75th percentile for O3
O3_75th_percentile = df['O3'].quantile(0.75)
temp_mean = df['temperature'].mean()
humidity_mean = df['humidity'].mean()
no2_mean = df['NO2'].mean()


# Define the conditions based on your report rules
conditions = [
    # Vehicular
    (
        (df['PM2.5'] >= 40) | (df['PM10'] >= 90) | (df['NO2'] >= 20)
    ) & (df['dist_to_road'] <= 0.2),

    # Industrial
    (
        (df['SO2'] >= 8) | (df['NO2'] >= 35)
    ) & (df['dist_to_industry'] <= 1.5),

    # Agricultural
    (
        (df['PM10'] >= 80)
    ) & (df['dist_to_farm'] <= 4),

    # Burning
    (
        (df['fire_nearby'] == 1) | (df['fire_count'] >= 1)
    ) & (df['PM2.5'] >= 45),

    # Photochemical
    (
        (df['O3'] >= O3_75th_percentile)
    ) & (
        (df['temperature'] > temp_mean) |
        (df['humidity'] < humidity_mean) |
        (df['NO2'] < no2_mean)
    )
]

# Define the corresponding labels
choices = ['Vehicular', 'Industrial', 'Agricultural', 'Burning', 'Photochemical']

# Assign the 'Label' column. 'Natural' is the default if no rule matches.
df['Source'] = np.select(conditions, choices, default='Natural')


#  Confidence Score Calculation ---

# Pollutant Score: Max normalized value of all key pollutants
df_norm['pollutant_score'] = df_norm[['PM2.5_norm', 'PM10_norm', 'NO2_norm', 'SO2_norm', 'O3_norm']].max(axis=1)

# Proximity Score: Max proximity to any source
df_norm['proximity_score'] = df_norm[['prox_road', 'prox_industry', 'prox_farm']].max(axis=1)

#  Fire Flag: Convert fire_nearby/fire_count to a 0-1 flag
df_norm['fire_flag'] = np.where((df['fire_nearby'] == 1) | (df['fire_count'] >= 1), 1, 0)

# Final Confidence Score Calculation
df['Confidence_Score'] = (
    W_POLLUTANT * df_norm['pollutant_score'] +
    W_PROXIMITY * df_norm['proximity_score'] +
    W_FIRE * df_norm['fire_flag']
)


#  Display/Save Results ---
print("\n--- Final Labeled Data Sample (First 5 Rows) ---")
print(df[['Source', 'Confidence_Score', 'PM2.5', 'dist_to_road']].head())

# To save the results to a new CSV file:
df.to_csv('labeled_pollution_data.csv', index=False)
