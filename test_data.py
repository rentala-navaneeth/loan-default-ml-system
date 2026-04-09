import pandas as pd

df = pd.read_csv("data/raw/credit_default.csv", header=1)
df = df.rename(columns={"default payment next month": "target"})
df = df.drop(columns=["ID"])
print(df.head())
print(df.columns)