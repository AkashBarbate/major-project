import pandas as pd
import numpy as np

df=pd.read_csv("raw_scraped_articles.csv")
print(f"Total: {len(df)}")
print()
print(df['source_section'].value_counts())
print()
print(df['source_type'].value_counts())
print(df.head())
# get duplicates and null values
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.info())