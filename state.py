user_states = {}

def reset_state(user_id):
    """Reset user state for new idea processing"""
    user_states[user_id] = {
        "idea_processed": False
    }
