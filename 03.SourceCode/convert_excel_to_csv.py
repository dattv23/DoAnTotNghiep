import pandas as pd
import os


def excel_to_csv(excel_file, output_folder, sheet_name):
    xls = pd.ExcelFile(excel_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_excel(xls, sheet_name=sheet_name)

    csv_file = os.path.join(output_folder, f"{sheet_name.lower()}.csv")

    df.to_csv(csv_file, index=False, encoding="utf-8-sig")
    print(f"Đã lưu: {csv_file}")


excel_file = "C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\02.Dataset\\Tiki.xlsx"
output_folder = "C:\\Users\\LT64\\Desktop\\DoAnTotNghiep\\02.Dataset"
sheet_name = "Reviews"
excel_to_csv(excel_file, output_folder, sheet_name)
