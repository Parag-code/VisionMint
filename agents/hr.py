import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class HRAgent:
    def __init__(self):
        self.name = "HR Officer"
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.state = {}

        self.clarity_prompt = """
You are the HR Head evaluating a new startup idea.

Rules for generating the question:

1. Begin with a warm, friendly reaction (4–6 words), such as:
   That sounds wonderful,
   This feels like a solid start,
   That’s a thoughtful direction,

   (Choose only ONE, and avoid repeating the same phrase across responses.)

2. Continue directly in the same sentence with a smooth, supportive question.
   Use soft starters like:
   I'm curious to know,
   I'm wondering to understand,
   Could you share more about,
   I'd love to understand,

3. The question must focus on a core HR aspect such as:
   - required team roles,
   - essential skills needed,
   - collaboration or execution capacity,
   - early hiring priorities.

4. Write it as ONE natural sentence with a comma after the reaction.

5. Important:
   - Do NOT use quotation marks.
   - Do NOT sound formal or technical.
   - Do NOT ask more than one question.
   - Keep it warm, supportive, and human.
"""



        self.final_prompt = """
Write exactly 3 sections:
1. What HR challenges exist (max 2 sentences)
2. Why these challenges matter (max 2 sentences)
3. How to begin building the right team (max 2 sentences)

Rules:
- No bullets
- No markdown
- No agent names
- Keep tone concise, clear, and professional
"""

    def _init(self, user_id):
        if user_id not in self.state:
            self.state[user_id] = {
                "idea": "",
                "question": None
            }

    def ask_clarity_question(self, user_id, idea):
        self._init(user_id)
        self.state[user_id]["idea"] = idea

        prompt = f"""
User Idea:
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
User Idea:
"{idea}"

HR Clarification Question:
"{question}"

User Answer:
"{user_answer}"

Final HR Evaluation:
{self.final_prompt}
"""
        return self._llm(prompt)

    def _llm(self, prompt):
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return result.choices[0].message.content
