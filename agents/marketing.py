import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class MarketingAgent:
    def __init__(self):
        self.name = "Marketing"
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.state = {}

        self.clarity_prompt = """
You are the Marketing Head reviewing a new business idea.

Rules for generating the question:

1. Begin with a short, natural reaction (4–6 words), such as:
   That looks quite interesting,
   This seems like a solid idea,
   That’s a promising direction,

   (Choose only ONE, and avoid repeating the same phrase across responses.)

2. Continue in the same sentence with a smooth, friendly marketing question.
   Use soft conversational starters like:
   I'm curious to know,
   I'm wondering to understand,
   Could you share more about,
   I'd like to get a sense of,

3. The question must focus on an important marketing aspect such as:
   - who the target users are,
   - what the value proposition is,
   - how the idea stands out in the market,
   - what unique need it fulfills.

4. Write it as ONE natural, continuous sentence with a comma after the reaction.

5. Important:
   - Do NOT use quotation marks.
   - Do NOT be technical or formal.
   - Do NOT ask more than one question.
   - Keep it conversational and thoughtful.
"""



        self.final_prompt = """
Write exactly 3 sections:
1. What can go wrong in the market (max 2 short sentences)
2. Why it can go wrong (max 2 short sentences)
3. How to start in the market (max 2 short sentences)

Rules:
- No markdown
- No bullets
- No agent names
- Tone must be concise and professional
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

Marketing Clarity Question:
"{question}"

User Answer:
"{user_answer}"

Now provide the final marketing evaluation:
{self.final_prompt}
"""
        return self._llm(prompt)

    def _llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
