import sys
import pandas as pd
import matplotlib.pyplot as plt


try:
    from baseline.dataloading import load_data_and_split
    from baseline.preprocessing import define_preprocessing_and_defaults
    from baseline.model_trainning import train_all_models
    from baseline.input import predict_source_from_partial_input
except ImportError as e:
    print("----------------------------------------------------------------------")
    print(f"CRITICAL ERROR: Failed to import baseline modules. {e}")
    print("Please ensure your Python environment recognizes 'baseline' as a package.")
    print("In some environments, you may need to run this script using:")
    print("python -m baseline.run_analysis (if this script was inside 'baseline')")
    print("----------------------------------------------------------------------")
    sys.exit(1)


def main():
    """
    Orchestrates the entire machine learning pipeline.
    """
    print("Starting ML Pipeline Orchestration...")
    print("-" * 50)

    # Global holders for components needed across steps
    X_train, X_test, y_train, y_test, le = None, None, None, None, None
    preprocess, mean_defaults, feature_cols = None, None, None
    final_model = None

    # STEP 1: Load Data and Split
    X_train, X_test, y_train, y_test, le = load_data_and_split()

    # STEP 2: Preprocessing Definitions and Defaults
    preprocess, mean_defaults, feature_cols = define_preprocessing_and_defaults(X_train)

    # STEP 3: Model Training and Evaluation (also displays plots)
    final_model = train_all_models(X_train, X_test, y_train, y_test, preprocess)

    # STEP 4: Interactive Input and Prediction
    print("\n" + "="*80)
    print("4. INTERACTIVE POLLUTION SOURCE PREDICTOR")
    print("="*80)

    try:
        # Prompt user for input
        user_lat = float(input("Enter Latitude (e.g., 34.05): "))
        user_lon = float(input("Enter Longitude (e.g., -118.24): "))
        user_city = input("Enter City name (e.g., London, Shanghai): ")
        user_season = input("Enter Season (e.g., Summer, Winter): ")

        # Make prediction
        predicted_source = predict_source_from_partial_input(
            model=final_model,
            le=le,
            mean_defaults=mean_defaults,
            feature_cols=feature_cols,
            lat=user_lat,
            lon=user_lon,
            city=user_city,
            season=user_season
        )

        print("\n" + "="*80)
        print(f"INPUT: Location ({user_lat}, {user_lon}), City: {user_city}, Season: {user_season}")
        print(f"PREDICTION: The pollution source is most likely: **{predicted_source}**")
        print("="*80)

    except ValueError:
        print("\n[ERROR] Invalid numerical input provided. Please re-run the script and enter numbers for coordinates.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred during prediction: {e}")


if __name__ == "__main__":
    main()