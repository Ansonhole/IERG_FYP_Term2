import pandas as pd
import os

print("=== Excel 重複資料清理工具 ===\n")

def clean_file(filename):
    if not os.path.exists(filename):
        print(f"❌ 找不到檔案：{filename}")
        return False
    
    try:
        df = pd.read_excel(filename)
        original_count = len(df)
        
        if df.empty:
            print(f"✅ {filename} 是空的，無需清理")
            return True

        # 1. 刪除完全相同的重複行
        df = df.drop_duplicates()

        # 2. 以 job_title + company_name 判斷重複（最合理的方式）
        key_columns = ['job_title', 'company_name']
        if all(col in df.columns for col in key_columns):
            df = df.drop_duplicates(subset=key_columns, keep='first')
        
        # 3. 刪除完全空白的行
        df = df.dropna(how='all')
        
        # 4. 重設索引（讓 Excel 沒有空白列）
        df = df.reset_index(drop=True)
        
        new_count = len(df)
        removed = original_count - new_count
        
        # 儲存回原本檔案
        df.to_excel(filename, index=False)
        
        print(f"✅ 已清理 {filename}")
        print(f"   原有 {original_count} 筆資料")
        print(f"   清理後 {new_count} 筆資料")
        print(f"   共刪除 {removed} 筆重複或空白資料\n")
        return True
        
    except Exception as e:
        print(f"❌ 清理 {filename} 時發生錯誤：{e}")
        return False


# === 主程式 ===
print("開始清理兩個 Excel 檔案...\n")

success1 = clean_file("data/graduate_jobs.xlsx")
success2 = clean_file("data/internships.xlsx")

if success1 or success2:
    print("="*60)
    print("🎉 所有 Excel 檔案已清理完成！")
    print("現在你可以執行以下指令重新建立索引：")
    print("   python build_index.py")
    print("   python -m src.main")
else:
    print("❌ 清理失敗，請確認 data/ 資料夾內有 Excel 檔案。")

input("\n按 Enter 鍵結束...")