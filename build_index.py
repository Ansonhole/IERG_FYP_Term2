# build_index.py
from src.embeddings import build_and_save_embeddings

if __name__ == "__main__":
    print("🚀 開始建立向量索引（包含學院 Q&A）...\n")
    df, embeddings = build_and_save_embeddings()
    print("\n🎉 索引建立完成！你現在可以執行主程式了。")
    print("   執行指令: python -m src.main")