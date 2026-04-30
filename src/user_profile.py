from typing import List

class UserProfile:
    def __init__(self, year: int, skills: List[str], interests: List[str], gpa: float = 0.0):
        self.year = year
        self.skills = [s.strip().lower() for s in skills if s.strip()]
        self.interests = [i.strip().lower() for i in interests if i.strip()]
        self.gpa = gpa

    def to_prompt(self) -> str:
        return f"""年級：{self.year} 年
主要技能：{', '.join(self.skills) if self.skills else '未提供'}
興趣領域：{', '.join(self.interests) if self.interests else '未提供'}
GPA：{self.gpa}"""