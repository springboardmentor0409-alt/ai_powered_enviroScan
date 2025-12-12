def generate_labels(df):
    """Assign pollution source labels based on heuristics."""

    # Thresholds
    p75_NO2 = df['NO2'].quantile(0.75)
    p75_SO2 = df['SO2'].quantile(0.75)
    p75_PM25 = df['PM2.5'].quantile(0.75)

    df['pollution_source'] = 'Natural'

    # Industrial
    df.loc[
        (df['dist_to_industry'] <= 1) & (df['SO2'] >= p75_SO2),
        'pollution_source'
    ] = 'Industrial'

    # Vehicular
    df.loc[
        ((df['dist_to_road'] <= 0.1) & (df['NO2'] >= p75_NO2)) |
        (df['traffic_index'] >= df['traffic_index'].quantile(0.9)),
        'pollution_source'
    ] = 'Vehicular'

    # Burning
    df.loc[
        (df['fire_nearby'] == 1) & (df['PM2.5'] >= p75_PM25),
        'pollution_source'
    ] = 'Burning'

    # Agricultural
    df.loc[
        (df['dist_to_farm'] <= 1) &
        (df['PM2.5'] >= p75_PM25) &
        (df['humidity'] < 40),
        'pollution_source'
    ] = 'Agricultural'

    return df
