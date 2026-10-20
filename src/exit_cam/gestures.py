TRIGGER_GESTURE = "Victory"
TRIGGER_THRESHOLD = 0.8  # confidence threshold for the trigger gesture


def hand_gestures(result):
    """The top gesture of every detected hand, as a list of name and score."""
    return [
        {"name": hand[0].category_name, "score": hand[0].score}
        for hand in result.gestures
        if hand
    ]


def is_trigger_sign(gestures):
    """True when either hand makes the trigger sign with enough confidence."""
    return any(
        gesture["name"] == TRIGGER_GESTURE and gesture["score"] >= TRIGGER_THRESHOLD
        for gesture in gestures
    )
