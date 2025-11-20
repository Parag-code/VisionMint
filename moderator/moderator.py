import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class ModeratorAgent:
    def __init__(self):
        self.name = "Moderator"
        self.model = "llama-3.3-70b-versatile"

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.state = {}

    def _init_user(self, user_id):
        if user_id not in self.state:
            self.state[user_id] = {
                "agent_outputs": {
                    "CEO": None,
                    "CFO": None,
                    "Marketing": None,
                    "Risk": None,
                    "HR": None
                },
                "final_report": None
            }

    def store_agent_output(self, user_id, agent_name, output):
        self._init_user(user_id)
        self.state[user_id]["agent_outputs"][agent_name] = output

    def generate_final_report(self, user_id, agent_outputs=None):

        self._init_user(user_id)

        if agent_outputs:
            self.state[user_id]["agent_outputs"] = agent_outputs

        data = self.state[user_id]["agent_outputs"]

        combined_text = (
            f"Strategy:\n{data['CEO']}\n\n"
            f"Finance:\n{data['CFO']}\n\n"
            f"Marketing:\n{data['Marketing']}\n\n"
            f"Risk:\n{data['Risk']}\n\n"
            f"Team:\n{data['HR']}"
        )

        prompt = f"""
You are the Moderator responsible for creating a polished, board-ready evaluation.

You MUST output EXACTLY this JSON structure:

{{
  "strengths": "",
  "opportunities": "",
  "risks": "",
  "recommendations": ""
}}

STRICT RULES:
- Write in a professional, human, consultant-like tone.
- Keep each field 2–3 sentences, concise but meaningful.
- Insights must feel deep, well-considered, and business-ready.
- Do NOT write long paragraphs.
- Do NOT add bullets, markdown, or extra fields.
- Do NOT output anything before or after the dictionary.
- Reflect all aspects: strategy, financial viability, market potential, data/privacy risks, team capability.

Rewrite, refine, and elevate the following agent insights into a clean, polished final report:
{combined_text}

Return ONLY the dictionary.
"""

        response = self._llm(prompt).strip()


        import json

        try:
            parsed_once = json.loads(response)
        except:
            parsed_once = response

        if isinstance(parsed_once, str):
            try:
                parsed_twice = json.loads(parsed_once)
            except:
                parsed_twice = parsed_once
        else:
            parsed_twice = parsed_once

        self.state[user_id]["final_report"] = parsed_twice
        return parsed_twice

    def answer_followup(self, user_id, question):
        self._init_user(user_id)
        final_report = self.state[user_id]["final_report"]

        prompt = f"""
You are answering a follow-up question about the evaluation.

Final Report:
{final_report}

User question: "{question}"

Respond with a short, clear, professional explanation.
No markdown. No bullets.
"""

        return self._llm(prompt)

    def _llm(self, prompt):
        res = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content
 