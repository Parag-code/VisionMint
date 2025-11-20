import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class CEOAgent:
    def __init__(self):
        self.name = "CEO"
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.state = {} 

        self.clarity_prompt = """
You are the CEO reviewing a startup idea.

Rules for how you generate the question:

1. Start with a short natural reaction (4–6 words) such as:
   That sounds interesting,
   This looks promising,
   That's an intriguing direction,

   (Choose only ONE, and do NOT repeat the same phrase every time.)

2. After the reaction, continue in the same sentence with a smooth, human-flowing question.
   Examples of soft question starters:
   I'm curious to know,
   I'm wondering to understand,
   Could you share,
   I'd like to know,

3. The question must go deeper into one important strategic aspect:
   - the core problem being solved,
   - the target user segment,
   - how the idea creates value,
   - what makes the approach different.

4. Important:
   - Use one single flowing sentence with a comma after the reaction.
   - Do NOT use quotation marks.
   - Do NOT be robotic.
   - Do NOT ask generic questions like "How does it work?".
   - Keep it short, sharp, and conversational.
"""



        self.final_prompt = """
You are a CEO giving a final evaluation.

Write exactly 3 sections:
1. What can go wrong (max 2 short sentences)
2. Why it can go wrong (max 2 short sentences)
3. How to start (max 2 short sentences)

Rules:
- No markdown
- No bullet points
- No long sentences
- No agent names
- Keep it crisp, professional, and very short
"""

    def _init(self, user_id):
        if user_id not in self.state:
            self.state[user_id] = {"idea": "", "question": None}

    def ask_clarity_question(self, user_id, idea):
        self._init(user_id)
        self.state[user_id]["idea"] = idea

        prompt = f"""
Startup Idea:
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
Idea:
"{idea}"

Clarity Question:
"{question}"

User Answer:
"{user_answer}"

Now produce the final evaluation:
{self.final_prompt}
"""
        return self._llm(prompt)

    def _llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
