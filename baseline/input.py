import pandas as pd

def predict_source_from_partial_input(model, le, mean_defaults, feature_cols, lat, lon, city, season):

    # 1. Start with the saved mean values dictionary (The 17 numerical defaults)
    final_input_row = mean_defaults.copy()

    # 2. Overwrite the features the user actually provided
    final_input_row['latitude'] = lat
    final_input_row['longitude'] = lon
    final_input_row['city'] = city
    final_input_row['Season'] = season

    # 3. Create the DataFrame for prediction.
    # Columns must be in the EXACT order defined by feature_cols.
    df_input = pd.DataFrame([final_input_row], columns=feature_cols)

    # 4. Make prediction using the trained pipeline
    prediction_encoded = model.predict(df_input)

    # 5. Decode and return the result using the LabelEncoder
    predicted_source = le.inverse_transform(prediction_encoded)

    return predicted_source[0]