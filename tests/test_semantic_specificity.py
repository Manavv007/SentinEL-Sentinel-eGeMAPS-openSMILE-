"""
Tests for the broadened, domain-agnostic semantic specificity.

The point of the change: detect scripted/essay prose and genuine personal answers from
STRUCTURE (first-person experience vs impersonal/societal/prescriptive prose), not from
hardcoded AWS/WebSocket/Instagram keyword lists. So a script in any topic is flagged and
a genuine answer in any topic is recognized.
"""

from __future__ import annotations

from engine.semantic_specificity import (
    compute_semantic_specificity,
    is_personal_natural_answer,
)


def _spec(text: str) -> dict:
    return compute_semantic_specificity({"transcript": text})


# --- Domain-agnostic scripted essay detection (NO tech/social keywords) --------------

def test_keyword_free_essay_flagged_generic():
    """A platitude essay on a topic with none of the hardcoded phrases is still flagged."""
    text = (
        "Leadership is about inspiring others. A good leader listens carefully and "
        "motivates the team toward a shared vision. Communication and empathy are "
        "essential qualities that everyone should develop in order to succeed."
    )
    spec = _spec(text)
    assert spec["generic_script_likelihood"] >= 0.55, spec
    assert spec["personal_experiential_score"] < 0.35
    assert not is_personal_natural_answer(spec)


def test_another_keyword_free_essay_flagged():
    text = (
        "Education plays a vital role in shaping society. It is important to encourage "
        "curiosity and critical thinking. People who keep learning throughout their lives "
        "tend to adapt better to change and contribute more to their communities."
    )
    spec = _spec(text)
    assert spec["generic_script_likelihood"] >= 0.55, spec
    assert spec["essay_generic_score"] >= 0.4


# --- Domain-agnostic personal recognition (NON-demo domains) -------------------------

def test_non_tech_personal_answer_recognized():
    """A genuine first-person answer in a non-tech domain reads as personal, not generic."""
    text = (
        "In my last job at a small bakery, I redesigned the morning prep schedule and "
        "I cut ingredient waste by about 15 percent over three months. I also trained "
        "two new hires on the new process."
    )
    spec = _spec(text)
    assert spec["personal_experiential_score"] >= 0.5, spec
    assert spec["generic_script_likelihood"] <= 0.5, spec
    assert is_personal_natural_answer(spec)


# --- Real "Harsh" interview examples -------------------------------------------------

def test_harsh_intro_recognized_as_personal():
    text = (
        "My name is Harsh, and I was highly attracted to this role because of the job "
        "description, which is machine learning intern, as I'm highly interested in this "
        "role, and especially behind the mathematical intuition of machine learning."
    )
    spec = _spec(text)
    # First-person experiential lifts personal narrative (previously missed by the regex).
    assert spec["personal_experiential_score"] >= 0.5, spec
    assert spec["personal_narrative_score"] >= 0.35, spec
    assert spec["generic_script_likelihood"] <= 0.5, spec


def test_harsh_project_answer_personal():
    text = (
        "So the features I used for the plant disease detection was, I trained it on "
        "around 54,000 images using a transfer learning approach. I used VGGNet, where I "
        "freeze around 297 layers, and we trained around 10 layers of the model."
    )
    spec = _spec(text)
    assert spec["personal_experiential_score"] >= 0.5, spec
    assert is_personal_natural_answer(spec)


def test_harsh_scripted_essays_flagged():
    essay_a = (
        "So the technology has changed the way we communicate, learn, and work in our "
        "daily lives. A few years ago, people depended heavily on books and physical "
        "classrooms for learning. But now anyone with an Internet connection can access courses."
    )
    essay_b = (
        "Spending too much time on screens can reduce productivity and affect mental health. "
        "That is why it is important to use technology wisely, not just for entertainment, "
        "but for personal growth, creativity, and meaningful connection with others."
    )
    sa, sb = _spec(essay_a), _spec(essay_b)
    assert sa["generic_script_likelihood"] >= 0.55, sa
    assert sb["generic_script_likelihood"] >= 0.55, sb
    assert not is_personal_natural_answer(sa)
    assert not is_personal_natural_answer(sb)


# --- Regression: previously-working keyword cases unchanged --------------------------

def test_influencer_personal_still_recognized():
    text = "I have been a social media influencer for the past five years. I use Instagram metrics to see what performs well, and I create content with AI tools."
    spec = _spec(text)
    assert is_personal_natural_answer(spec)


def test_genuine_concept_explanation_not_over_flagged():
    """A genuine (if abstract) technical explanation should not score like a platitude essay
    — domain-agnostic essay weighting favors societal/prescriptive markers, which this lacks."""
    text = (
        "Overfitting happens when a model memorizes the training data instead of learning "
        "the general pattern, so it performs well on training but poorly on new data."
    )
    spec = _spec(text)
    # It may be uncertain, but should not be a confident essay-level generic script.
    assert spec["essay_generic_score"] < 0.55, spec
