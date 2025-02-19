import pandas as pd

def load_dataset(file_path):
    """Load dataset from a CSV file using pandas."""
    df = pd.read_csv(file_path)
    return df.to_dict(orient="records")

def load_all_datasets():
    """Load all datasets from CSV files."""
    bhagavad_gita = load_dataset("data/bhagavad_gita.csv")
    quran = load_dataset("data/quran.csv")
    bible = load_dataset("data/bible.csv")
    return {
        "Bhagavad Gita": bhagavad_gita,
        "Quran": quran,
        "Bible": bible
    }
