# data-pipeline-task
ETL data pipeline using Pandas and Scikit-learn
# ETL Pipeline using Pandas and Scikit-learn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -------------------------------
# 1. EXTRACT: Load the data
# -------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# -------------------------------
# 2. TRANSFORM: Clean & preprocess
# -------------------------------
def preprocess_data(df):
    # Drop duplicate rows
    df = df.drop_duplicates()

    # Handle missing values
    df = df.dropna()

    # Separate features by type
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns

    # Pipelines for preprocessing
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Apply transformations
    processed_data = preprocessor.fit_transform(df)

    return processed_data, preprocessor

# -------------------------------
# 3. LOAD: Save processed data
# -------------------------------
def save_data(processed_data, output_file):
    processed_df = pd.DataFrame(processed_data.toarray() 
                                if hasattr(processed_data, "toarray") 
                                else processed_data)
    processed_df.to_csv(output_file, index=False)

# -------------------------------
# MAIN FUNCTION
# -------------------------------
def main():
    input_file = "data.csv"      # Input dataset
    output_file = "processed_data.csv"  # Output dataset

    print("Loading data...")
    df = load_data(input_file)

    print("Preprocessing data...")
    processed_data, _ = preprocess_data(df)

    print("Saving processed data...")
    save_data(processed_data, output_file)

    print("ETL Pipeline completed successfully!")

if __name__ == "__main__":
    main()
