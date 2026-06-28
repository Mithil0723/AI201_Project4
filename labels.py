"""
labels.py — maps a confidence score to the verdict + the plain-language
transparency label a reader sees on the platform.

`confidence` is the system's estimated probability the text is AI-generated:
    0.0  = we're sure a human wrote it
    0.5  = genuinely undecided
    1.0  = we're sure it's AI-generated

Thresholds (deliberately asymmetric to protect human creators — see README):
    confidence  < 0.40            -> likely_human
    0.40 <= confidence <= 0.70    -> uncertain
    confidence  > 0.70            -> likely_ai

The "likely AI" bar is set high (0.70) and the uncertain band is wide on
purpose: on a writing platform, wrongly flagging a real person's work as AI is
the more damaging error, so we'd rather say "uncertain" than accuse.
"""

LIKELY_HUMAN_MAX = 0.40
LIKELY_AI_MIN = 0.70


def classify(confidence):
    """Return the attribution string for a given confidence."""
    if confidence < LIKELY_HUMAN_MAX:
        return "likely_human"
    if confidence > LIKELY_AI_MIN:
        return "likely_ai"
    return "uncertain"


def make_label(confidence):
    """
    Return (attribution, label_text). The label is what a non-technical reader
    sees. It states the verdict, shows a percentage they can interpret, and is
    honest that this is an automated estimate with an appeal path.
    """
    attribution = classify(confidence)
    ai_pct = round(confidence * 100)
    human_pct = round((1 - confidence) * 100)

    if attribution == "likely_ai":
        text = (
            f"🤖 Likely AI-generated. Our automated analysis estimates a "
            f"{ai_pct}% chance this text was produced with generative AI. "
            f"This is an estimate, not a verdict — the creator can appeal if "
            f"they believe it's wrong."
        )
    elif attribution == "likely_human":
        text = (
            f"✍️ Likely human-written. Our automated analysis found no strong "
            f"signs of AI generation (estimated {human_pct}% human-written). "
            f"This is an estimate, not a guarantee."
        )
    else:  # uncertain
        text = (
            f"❓ Uncertain origin. Our analysis couldn't confidently tell "
            f"whether a human or AI wrote this (roughly {ai_pct}% toward AI, "
            f"{human_pct}% toward human). We're flagging it as inconclusive "
            f"rather than guessing."
        )

    return attribution, text
