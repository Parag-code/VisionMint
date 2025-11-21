from flask import Flask, request, jsonify
from state import reset_state

from agents.ceo import CEOAgent
from agents.cfo import CFOAgent
from agents.marketing import MarketingAgent
from agents.risk import RiskAgent
from agents.hr import HRAgent
from moderator.moderator import ModeratorAgent
from flask_cors import CORS


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

agent_sequence = ["CEO", "CFO", "Marketing", "Risk", "HR"]

agents = {
    "CEO": CEOAgent(),
    "CFO": CFOAgent(),
    "Marketing": MarketingAgent(),
    "Risk": RiskAgent(),
    "HR": HRAgent()
}

moderator = ModeratorAgent()

@app.route("/awake", methods=["GET"])
def health():
    return "OK", 200



@app.route("/chat", methods=["POST"])
def chat():
    data = request.json

    if "user_id" not in data:
        return jsonify({"error": "user_id required"}), 400
    if "message" not in data:
        return jsonify({"error": "message required"}), 400

    user_id = data["user_id"]
    msg = data["message"].strip()

    if moderator.state.get(user_id) is None:
        idea = msg.strip()

        reset_state(user_id)
        moderator._init_user(user_id)

        moderator.state[user_id]["idea"] = idea
        moderator.state[user_id]["current_agent"] = "CEO"
        moderator.state[user_id]["agent_outputs"] = {}

        q = agents["CEO"].ask_clarity_question(user_id, idea)

        return jsonify({
            "agent": "CEO",
            "question": q,
            "done": False
        })

    if moderator.state[user_id]["current_agent"] is not None:

        user_answer = msg.strip()

        current_agent = moderator.state[user_id]["current_agent"]
        idea = moderator.state[user_id]["idea"]

        final_summary = agents[current_agent].generate_final_summary(
            user_id, user_answer
        )

        moderator.state[user_id]["agent_outputs"][current_agent] = final_summary

        idx = agent_sequence.index(current_agent)

        if idx < len(agent_sequence) - 1:
            next_agent = agent_sequence[idx + 1]
            moderator.state[user_id]["current_agent"] = next_agent

            q = agents[next_agent].ask_clarity_question(user_id, idea)

            return jsonify({
                "agent": next_agent,
                "question": q,
                "done": False
            })

        final_report = moderator.generate_final_report(
            user_id, moderator.state[user_id]["agent_outputs"]
        )
        
        moderator.state[user_id]["current_agent"] = None


        return jsonify({
            "agent": "Moderator",
            "final_report": final_report,
            "done": True,
            "message": "You’ve got a solid idea brewing here. If anything feels unclear or you’d like to discuss more, I’m here to help!"
        })

    followup = moderator.answer_followup(user_id, msg)

    return jsonify({
        "agent": "Moderator",
        "answer": followup,
        "done": False
    })


if __name__ == "__main__":
    app.run(debug=True)
