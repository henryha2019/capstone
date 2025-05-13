import pandas as pd

# techdebt: replace hardcoded path with option to insert path when starting up dashboard
def load_data(path="../../data/sample_belt_conveyer.csv"):
    df = pd.read_csv(path)

    df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    return df

def get_unique_locations(df):
    return sorted(df["location"].dropna().unique())
