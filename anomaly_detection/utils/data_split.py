import pandas as pd

def prepare_datasets(file_path, api_size=50000, random_state=42):
    print("🔹 Loading scaled dataset...")
    df = pd.read_csv(file_path)

    # Allocate data for API simulation
    print(f"🔹 Reserving {api_size} rows for API simulation...")
    api_simulation_data = df.sample(n=api_size, random_state=random_state)

    # Remaining data for training
    train_data = df.drop(api_simulation_data.index)

    # Save datasets
    api_simulation_data.to_csv('api_simulation_data.csv', index=False)
    train_data.to_csv('creditcard_train.csv', index=False)

    print("\n✅ Data preparation complete!")
    print(f"Training Set Size: {len(train_data)} rows")
    print(f"API Simulation Set Size: {len(api_simulation_data)} rows")

if __name__ == "__main__":
    prepare_datasets('creditcard_scaled.csv')
