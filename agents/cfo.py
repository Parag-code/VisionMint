import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class CFOAgent:
    def __init__(self):
        self.name = "CFO"
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.state = {} 

        self.clarity_prompt = """
You are the CFO reviewing a new business idea.

Rules for generating the question:

1. Begin with a short, natural reaction (4–6 words), such as:
   That sounds promising,
   This looks financially interesting,
   That seems like a solid start,

   (Choose only ONE each time, and avoid repeating the same phrase across different responses.)

2. After the reaction, continue in the same sentence with a smooth, human-flowing question.
   Use soft conversational starters like:
   I'm curious to know,
   I'm wondering to understand,
   Could you share more about,
   I'd like to get a sense of,

3. The question must focus on an essential financial element such as:
   - how the idea will generate revenue,
   - pricing direction,
   - cost assumptions,
   - early profitability expectations,
   - or overall financial sustainability.

4. Write the entire reaction + question as ONE natural, continuous sentence with a comma after the reaction.

5. Important:
   - Do NOT use quotation marks.
   - Do NOT sound technical or robotic.
   - Do NOT ask more than one question.
   - Keep it short, friendly, and easy to understand.
"""



        self.final_prompt = """
Write exactly 3 sections:
1. What can go wrong financially (max 2 short sentences)
2. Why it can go wrong financially (max 2 short sentences)
3. How to start financially (max 2 short sentences)

Rules:
- No markdown
- No bullets
- No long sentences
- No agent names
- Tone must be crisp and professional
"""

    def _init(self, user_id):
        if user_id not in self.state:
            self.state[user_id] = {"idea": "", "question": None}

    def ask_clarity_question(self, user_id, idea):
        self._init(user_id)
        self.state[user_id]["idea"] = idea

        prompt = f"""
Business Idea:
"{idea}"

{self.clarity_prompt}
"""
        question = self._llm(prompt).strip()
        question = question.replace("\n", " ")
        self.state[user_id]["question"] = question
        return question

    def generate_final_summary(self, user_id, user_answer):
        self._init(user_id)

        idea = self.state[user_id]["idea"]
        question = self.state[user_id]["question"]

        prompt = f"""
Business Idea:
"{idea}"

Financial Clarity Question:
"{question}"

User Answer:
"{user_answer}"

Now provide the final evaluation:
{self.final_prompt}
"""
        return self._llm(prompt)

    def _llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
