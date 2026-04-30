import pandas as pd
import os
from typing import List

def clean_skill_list(skill_str) -> List[str]:
    """自動清理 required_skills 和 preferred_skills"""
    if pd.isna(skill_str) or str(skill_str).strip() in ['nan', 'None', '']:
        return []
    skills = str(skill_str).split(',')
    cleaned = [s.strip() for s in skills if s.strip() and s.strip().lower() != 'nan']
    return cleaned

def load_all_data() -> pd.DataFrame:
    data = []
    base_path = "data"
    
    # Load internships
    try:
        df = pd.read_excel(os.path.join(base_path, "internships.xlsx"))
        df["type"] = "internship"
        data.append(df)
        print(f"✅ Loaded {len(df)} internships")
    except Exception as e:
        print(f"Warning internships: {e}")

    # Load graduate jobs
    try:
        df = pd.read_excel(os.path.join(base_path, "graduate_jobs.xlsx"))
        df["type"] = "graduate_job"
        data.append(df)
        print(f"✅ Loaded {len(df)} graduate jobs")
    except Exception as e:
        print(f"Warning graduate_jobs: {e}")

    # Load college Q&A (新格式)
    try:
        df = pd.read_excel(os.path.join(base_path, "college_choices.xlsx"))
        df["type"] = "college_choice"
        data.append(df)
        print(f"✅ Loaded {len(df)} college Q&A pairs")
    except Exception as e:
        print(f"Warning college_choices: {e}")

    if not data:
        raise FileNotFoundError("No data files found in data/ folder!")

    final_df = pd.concat(data, ignore_index=True)
    print(f"\n🎉 Total data loaded: {len(final_df)} rows\n")
    return final_df