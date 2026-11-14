"""
pre_training_cleanup.py
-----------------------
Two quick fixes before training:
1. Remove the 50 most obviously mislabeled articles
   (non-sports articles in Sports source with no Sports label,
    non-entertainment in Entertainment with no Entertainment label)
2. Save cleaned file ready for train_improved.py
"""
import pandas as pd

df = pd.read_csv('gemini_labeled_articles.csv')

CATEGORIES = [
    'Politics', 'Science and Technology', 'Business and Economy',
    'International Affairs', 'Education', 'Health', 'Sports', 'Entertainment'
]

print(f"Before cleanup: {len(df)} articles")

# Remove articles that are clearly from wrong RSS source
# AND have zero label match — these are scraping errors, not labeling errors
# e.g. "AP confirms Myanmar mass graves" in Sports source → not sports at all
clearly_wrong = (
    ((df['source_section'] == 'Sports')         & (df['Sports'] == 0)         & (df['Business and Economy'] == 1)) |
    ((df['source_section'] == 'Entertainment')  & (df['Entertainment'] == 0)  & (df['Business and Economy'] == 1)) |
    ((df['source_section'] == 'Health')         & (df['Health'] == 0)         & (df['Sports'] == 1))
)

print(f"Removing {clearly_wrong.sum()} clearly mislabeled articles")
df = df[~clearly_wrong].reset_index(drop=True)
print(f"After cleanup: {len(df)} articles")

# Final label counts
print("\nFinal label distribution:")
counts = df[CATEGORIES].sum().sort_values(ascending=False)
for cat, count in counts.items():
    pct = count / len(df) * 100
    print(f"  {cat:<28} {count:>4}  ({pct:.1f}%)")

print(f"\nImbalance ratio: {counts.max()/counts.min():.2f}x")
df.to_csv('gemini_labeled_articles_clean.csv', index=False, encoding='utf-8')
print("\nSaved → gemini_labeled_articles_clean.csv")
print("Ready for: python train_improved.py")