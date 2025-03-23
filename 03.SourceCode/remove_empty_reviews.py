import pandas as pd

file_path = "D:\\DoAnTotNghiep\\02.Dataset\\labeling_reviews.xlsx"
df = pd.read_excel(file_path)

df_cleaned = df[df["Comment"].notna() & (df["Comment"] != "")]

output_file = "D:\\DoAnTotNghiep\\02.Dataset\\reviews_not_empty.xlsx"
df_cleaned.to_excel(output_file, index=False)

print(f"File đã được lưu: {output_file}")
