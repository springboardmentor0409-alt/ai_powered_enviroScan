from src.preprocessing.clean_data import load_raw_data, clean_pollution_data, save_clean_data
from src.preprocessing.feature_engineering import apply_feature_engineering
from src.preprocessing.label_data import generate_labels

RAW_PATH = "data/EnviroFinal_final_unlabled.csv"
CLEAN_PATH = "data/processed/cleaned_data.csv"
LABELED_PATH = "data/labeled/labeled_pollution_data.csv"

def run_pipeline():
    print("🔄 Loading raw data...")
    df = load_raw_data(RAW_PATH)
    print("➡ Raw shape:", df.shape)

    print("🧹 Cleaning data...")
    df = clean_pollution_data(df)
    print("➡ Cleaned shape:", df.shape)

    print("⚙ Applying feature engineering...")
    df = apply_feature_engineering(df)

    print("🏷 Generating labels...")
    df = generate_labels(df)
    print(df['pollution_source'].value_counts())

    print("💾 Saving labeled dataset...")
    save_clean_data(df, LABELED_PATH)

    print("🎉 Pipeline execution complete!")

if __name__ == "__main__":
    run_pipeline()
