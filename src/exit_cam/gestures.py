def hand_gestures(result):
    """The top gesture of every detected hand, as a list of name and score."""
    return [
        {"name": hand[0].category_name, "score": hand[0].score}
        for hand in result.gestures
        if hand
    ]


def is_trigger_sign(gestures, name, threshold):
    """True when either hand makes the sign called name with a score of at
    least threshold."""
    return any(
        gesture["name"] == name and gesture["score"] >= threshold
        for gesture in gestures
    )
