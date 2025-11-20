import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class RiskAgent:
    def __init__(self):
        self.name = "Risk Officer"
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.state = {}   
        
        self.clarity_prompt = """
You are the Chief Risk & Compliance Officer reviewing a new business idea.

Rules for generating the question:

1. Begin with a short, natural reaction (4–6 words), such as:
   That seems quite promising,
   This looks like a solid start,
   That’s an interesting direction so far,

   (Choose only ONE each time, and avoid repeating the same phrase across responses.)

2. Continue directly in the same sentence with a smooth, friendly risk-related question.
   Use soft openers like:
   I'm curious to know,
   I'm wondering to understand,
   Could you share more about,
   I'd like to get a sense of,

3. The question must focus on a meaningful risk aspect such as:
   - legal exposure,
   - data privacy handling,
   - compliance or regulatory obligations,
   - operational vulnerabilities or weaknesses.

4. Write the entire reaction + question as ONE natural, continuous sentence with a comma after the reaction.

5. Important:
   - Do NOT use quotation marks.
   - Do NOT be technical, harsh, or formal.
   - Do NOT ask more than one question.
   - Keep it light, simple, and conversational.
"""




        self.final_prompt = """
Write exactly 3 sections:
1. What risk issues can go wrong (max 2 sentences)
2. Why it can go wrong (max 2 sentences)
3. How to start safely (max 2 sentences)

Rules:
- No markdown
- No bullets
- No agent names
- Keep tone concise and professional
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

Risk Clarification Question:
"{question}"

User Answer:
"{user_answer}"

Now provide the final risk evaluation:
{self.final_prompt}
"""
        return self._llm(prompt)


    def _llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
