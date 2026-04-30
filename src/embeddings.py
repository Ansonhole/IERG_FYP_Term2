from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os
from src.data_loader import load_all_data

def create_text_for_embedding(row):
    if row.get('type') == "college_choice":
        # 針對 Q&A 格式優化
        return f"College: {row.get('college', '')} | Q: {row.get('question', '')} | A: {row.get('answer', '')}"
    else:
        # 職位格式
        skills = str(row.get('required_skills', '')) + " " + str(row.get('preferred_skills', ''))
        return f"Title: {row.get('job_title', row.get('title', ''))}. " \
               f"Company: {row.get('company_name', '')}. " \
               f"Skills: {skills}. " \
               f"Description: {row.get('description', '')}"

def build_and_save_embeddings():
    """建立 embeddings 並儲存索引"""
    print("📚 載入所有資料...")
    df = load_all_data()
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("🔄 正在生成向量嵌入（這可能需要一點時間）...")
    texts = df.apply(create_text_for_embedding, axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype('float32')
    
    # 儲存 embeddings 與 metadata
    np.save("data/embeddings.npy", embeddings)
    df.to_pickle("data/metadata.pkl")
    
    # 建立 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, "data/faiss_index.bin")
    
    print(f"✅ 向量索引建立成功！")
    print(f"   - 總資料筆數: {len(df)} rows")
    print(f"   - 其中學院 Q&A: {len(df[df['type']=='college_choice'])} 筆")
    print(f"   - Embeddings 形狀: {embeddings.shape}")
    
    return df, embeddings