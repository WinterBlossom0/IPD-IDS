import pandas as pd

latent_cols = ['latent_0', 'latent_1', 'latent_2', 'latent_3', 'latent_4','label']

# df_train
print("=" * 50)
print("df_train")
print("=" * 50)
df_train = pd.read_csv("df_train.csv")
print(f"Shape before: {df_train.shape}")
df_train_deduped = df_train.drop_duplicates(subset=latent_cols, keep='first')
df_train_deduped.reset_index(drop=True, inplace=True)
print(f"Shape after:  {df_train_deduped.shape}")
print(f"Rows removed: {df_train.shape[0] - df_train_deduped.shape[0]}")
df_train_deduped.to_csv("df_train.csv", index=False)
print("✓ Saved to df_train.csv")

# df_test
print("\n" + "=" * 50)
print("df_test")
print("=" * 50)
df_test = pd.read_csv("df_test.csv")
print(f"Shape before: {df_test.shape}")
df_test_deduped = df_test.drop_duplicates(subset=latent_cols, keep='first')
df_test_deduped.reset_index(drop=True, inplace=True)
print(f"Shape after:  {df_test_deduped.shape}")
print(f"Rows removed: {df_test.shape[0] - df_test_deduped.shape[0]}")
df_test_deduped.to_csv("df_test.csv", index=False)
print("✓ Saved to df_test.csv")
