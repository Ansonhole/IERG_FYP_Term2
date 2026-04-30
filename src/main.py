from src.recommender import generate_recommendation
from src.user_profile import UserProfile
import os
from dotenv import load_dotenv

load_dotenv()

def main():
    print("\n" + "="*80)
    print("🚀 CUHK 工程系個人化推薦系統")
    print("="*80)
    print("請選擇模式：")
    print("1. 工作推薦模式（實習 / 畢業職位）")
    print("2. 學院推薦模式（幫助你選擇最適合的學院）")
    print("輸入 'exit' 結束\n")

    while True:
        mode = input("請輸入選擇 (1 或 2): ").strip()

        if mode == 'exit':
            print("感謝使用，再見！")
            break

        if mode == '1':
            # 工作模式（維持不變）
            print("\n【工作推薦模式】")
            year = input("年級（例如 2、3、4）： ").strip()
            skills = input("主要技能（用逗號分隔）： ").strip()
            interests = input("興趣領域（用逗號分隔）： ").strip()
            gpa = input("GPA（例如 3.5）： ").strip()

            user_profile = UserProfile(year=year, skills=[s.strip() for s in skills.split(',') if s.strip()], 
                                     interests=interests, gpa=float(gpa) if gpa else 0.0)
            print("✅ 個人資料已儲存！\n")

        elif mode == '2':
            # === 優化後的學院推薦模式 ===
            print("\n【學院推薦模式】")
            print("請回答以下問題，我會為你推薦最適合的學院：\n")
            
            residential = input("1. 你希望住學院宿舍嗎？ (是 / 否 / 無所謂): ").strip()
            christian = input("2. 你對基督教氛圍和價值觀的重視程度？ (高 / 中 / 低 / 無所謂): ").strip()
            ge_interest = input("3. 你對通識教育（GE課程）的興趣程度？ (高 / 中 / 低): ").strip()
            activity = input("4. 你喜歡參加學院活動、領導培訓、交流計劃嗎？ (是 / 否 / 無所謂): ").strip()
            academic = input("5. 你比較重視學院的學術氛圍還是生活體驗？ (學術 / 生活 / 兩者都要): ").strip()
            personality = input("6. 你的性格比較內向還是外向？ (內向 / 外向 / 都可以): ").strip()

            user_profile = {
                "mode": "college",
                "residential": residential,
                "christian": christian,
                "ge_interest": ge_interest,
                "activity": activity,
                "academic": academic,
                "personality": personality
            }
            print("✅ 偏好已記錄！現在你可以輸入問題，或直接輸入 '推薦學院' / '推薦最適合我的學院'。\n")

        else:
            print("❌ 請輸入 1 或 2")
            continue

        # 對話循環
        while True:
            query = input("\n你： ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                continue

            try:
                answer, results = generate_recommendation(query, user_profile)
            except Exception as e:
                if "503" in str(e) or "UNAVAILABLE" in str(e):
                    print("❌ Gemini 目前流量過高，請稍等 10-30 秒後再試。")
                else:
                    print(f"❌ 發生錯誤: {e}")

        print("\n" + "-"*60)

if __name__ == "__main__":
    main()