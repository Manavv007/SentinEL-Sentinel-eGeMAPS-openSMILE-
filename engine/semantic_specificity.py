"""
Semantic specificity scoring — person-independent transcript analysis.

Generic scripted answers (essay prose, platitudes) score low specificity.
Personal spontaneous answers (names, numbers, project details, hedges) score high.
"""

from __future__ import annotations

import math
import re
from typing import Any

import config

# Common sentence starters — not treated as proper nouns
_SENTENCE_STARTERS = frozenset(
    {
        "So",
        "The",
        "This",
        "That",
        "It",
        "I",
        "We",
        "They",
        "There",
        "When",
        "If",
        "But",
        "And",
        "Because",
        "However",
        "Therefore",
        "Also",
        "Well",
        "Yeah",
        "Yes",
        "No",
    }
)

_HEDGE_PATTERN = re.compile(
    r"\b(uh|um|erm|ah|like|kind of|sort of|you know|i think|i guess|"
    r"maybe|approximately|around|roughly|about)\b",
    re.IGNORECASE,
)

_NAME_INTRO = re.compile(
    r"\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b", re.IGNORECASE
)
_ROLE_TERMS = re.compile(
    r"\b(intern|engineer|developer|researcher|scientist|analyst|manager|"
    r"machine learning|deep learning|data science|software|backend|frontend)\b",
    re.IGNORECASE,
)

_FIRST_PERSON_ACTION = re.compile(
    r"\bI\s+(used|built|worked|trained|found|developed|implemented|created|"
    r"designed|deployed|wrote|learned|studied|applied|optimized|fine-tuned|"
    r"experimented|researched|analyzed|engineered|was|am|have|had|did|"
    r"attracted|chose|selected|joined|led|managed|presented|published)\b",
    re.IGNORECASE,
)

_GENERIC_PHRASES = (
    "technology has changed",
    "in today's world",
    "it is important to",
    "personal growth",
    "meaningful connection",
    "use technology wisely",
    "not just for entertainment",
    "mental health",
    "reduce productivity",
    "few years ago",
    "people depended heavily",
    "anyone with an internet connection",
    "the way we communicate",
    "daily lives",
)

_NUMBER_PATTERN = re.compile(
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?|\b\d+(?:\.\d+)?%|\bv?\d+(?:\.\d+)?\b",
)

# Textbook / memorized technical script patterns (not personal project narrative)
_TECH_DEFINITION_PHRASES = (
    "unlike traditional",
    "this allowed",
    "persistent connection",
    "persistent full",
    "request response",
    "request driven",
    "stateless",
    "full-duplex",
    "full-reflex",
    "between client and server",
    "without reopening",
    "broadcast events",
    "web-circuit",
    "websocket",
    "web socket",
    "lambda functions",
    "http is",
    "implemented using",
    "two-way communication",
    "front-end and back-end",
    "frontend and backend",
    "client connection",
    "dynamically",
)

_PERSONAL_NARRATIVE = re.compile(
    r"\b(I|we|my|our)\s+(use|used|have been|try to|tried to|repeat|create|creating|"
    r"post|posts|stream|streams|analyze|analyse|content|influencer|team|metrics of)\b",
    re.IGNORECASE,
)

_PERSONAL_PROJECT_ANCHOR = re.compile(
    r"\b(I|we)\s+(built|created|implemented|developed|designed|deployed|architected|"
    r"faced|struggled|learned|worked on|made|chose)\b",
    re.IGNORECASE,
)

_DEFINitional_PASSIVE = re.compile(
    r"\b(is|are|was|were)\s+(stateless|persistent|implemented|used to|designed to)\b",
    re.IGNORECASE,
)

_TECH_PROPER = re.compile(
    r"\b(AWS|HTTP|HTTPS|WebSocket|WebSockets|Lambda|API|REST|TCP|UDP|RespoNet|Canva|"
    r"Instagram|Facebook)\b",
    re.IGNORECASE,
)

# Generic systems/architecture concept vocabulary. A dense enumeration of these with
# no personal narrative or specifics signals memorized/textbook technical prose
# (e.g. "We implemented microservices ... authentication, real-time communication ...").
_TECH_CONCEPT_TERMS = re.compile(
    r"\b(microservices?|monolith\w*|scalab\w+|latency|throughput|load[ -]?balanc\w*|"
    r"fault[ -]?toleran\w*|high availability|authentication|authorization|real-?time|"
    r"asynchronous|synchronous|persistent|stateless|full[ -]?duplex|half[ -]?duplex|"
    r"broadcast|distributed|concurren\w+|middleware|caching|in-memory|encryption|"
    r"websockets?|rest api|api gateway|message queue|pub[ -]?sub|independent services?|"
    r"event[ -]?driven|orchestrat\w+|containeriz\w+|incident management|media handling)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Domain-agnostic content features (no hardcoded tech/domain vocabulary).
#
# These generalize the "scripted essay vs genuine personal answer" signal beyond the
# AWS/WebSocket/Instagram keyword lists above, so a scripted answer in ANY domain
# (and a genuine answer in any domain) is scored on structure, not topic words.
# ---------------------------------------------------------------------------

# First-person reference (speaker talking about themselves).
_FP_PRONOUN = re.compile(r"\b(i|we|my|our|me|us|mine|ours|i'm|i've|i'd|i'll|we're|we've|we'll)\b", re.IGNORECASE)

# First-person + experiential/action verb = genuine lived experience (domain-agnostic;
# broad verb morphology rather than a fixed project vocabulary).
_FP_EXPERIENTIAL = re.compile(
    r"\b(i|we)\s+(?:\w+\s+){0,2}?"
    r"(did|do|does|made|make|built|build|used|use|created|create|worked|work|trained|train|"
    r"designed|design|implemented|implement|developed|develop|chose|choose|led|lead|managed|"
    r"manage|tried|try|found|find|learned|learn|faced|face|struggled|struggle|decided|decide|"
    r"tested|test|deployed|deploy|wrote|write|fixed|fix|improved|improve|reduced|reduce|"
    r"increased|increase|started|start|joined|join|studied|study|experimented|analyzed|"
    r"noticed|realized|handled|solved|presented|published|set|ran|run|took|take|got|gave|"
    r"was|were|am|have|had|spent|chose|picked|added|removed)\b",
    re.IGNORECASE,
)

# Generic, non-personal societal subjects — the hallmark of platitude/essay prose.
_GENERIC_SUBJECT = re.compile(
    r"\b(technology|technologies|people|society|everyone|everybody|anyone|anybody|someone|"
    r"somebody|humanity|mankind|the world|the internet|users?|individuals?|students?|children|"
    r"nowadays|these days|today's world|the way we|we all|one's|human beings?|the future)\b",
    re.IGNORECASE,
)

# Prescriptive / generalizing essay register.
_PRESCRIPTIVE = re.compile(
    r"\b(it is important|it's important|we should|you should|one should|everyone should|"
    r"we must|it is essential|it's essential|it is necessary|we need to|important to|"
    r"that is why|this is why|in conclusion|to sum up|overall|in today's|in our daily|"
    r"should always|should never|\bshould\b)\b",
    re.IGNORECASE,
)

# Generic copula+predicate definitions ("X is essential", "Y is a ...") — no domain words.
_GENERIC_DEFINITIONAL = re.compile(
    r"\b(is|are|was|were)\s+(a|an|the|one|able to|used to|known as|defined as|considered|"
    r"essential|important|necessary|responsible for|about)\b",
    re.IGNORECASE,
)

# Abstract nouns / nominalizations by morphology (essays are nominalization-heavy).
_ABSTRACT_NOUN = re.compile(
    r"\b[a-z]{4,}(?:tion|sion|ment|ness|ity|ism|ance|ence|ship|hood)\b", re.IGNORECASE
)

# Generic second-person ("you can ...", "your ...") used for advice/essay prose.
_SECOND_PERSON_GENERIC = re.compile(r"\b(you can|you should|you will|you must|you are|your)\b", re.IGNORECASE)


def _domain_agnostic_signals(
    text: str, words: list[str], sentences: list[str]
) -> dict[str, float]:
    """
    Structure-based (topic-independent) features that separate scripted essay prose
    from genuine first-person experience.

    personal_experiential_score : high => speaker recounting their own actions/experience
    essay_generic_score         : high => impersonal platitude/definition prose

    essay_generic_score is intentionally weighted toward *societal subjects* and
    *prescriptive register* (the platitude signature) rather than abstract/definitional
    structure alone — so a genuine technical *explanation* (abstract but not preachy)
    is not mistaken for a memorized essay.
    """
    n_words = max(len(words), 1)
    n_sent = max(len(sentences), 1)

    fp_experiential = len(_FP_EXPERIENTIAL.findall(text))
    fp_sentences = sum(1 for s in sentences if _FP_PRONOUN.search(s))
    first_person_clause_ratio = fp_sentences / n_sent
    experiential_density = min(1.0, fp_experiential / max(n_sent / 1.5, 1.0))
    personal_experiential_score = min(
        1.0, 0.4 * first_person_clause_ratio + 0.6 * experiential_density
    )

    generic_subject_density = min(1.0, len(_GENERIC_SUBJECT.findall(text)) / max(n_words / 18.0, 1.0))
    prescriptive_density = min(1.0, len(_PRESCRIPTIVE.findall(text)) / 2.0)
    abstract_noun_ratio = min(1.0, len(_ABSTRACT_NOUN.findall(text)) / max(n_words / 9.0, 1.0))
    second_person_density = min(1.0, len(_SECOND_PERSON_GENERIC.findall(text)) / 2.0)
    definitional_density = min(1.0, len(_GENERIC_DEFINITIONAL.findall(text)) / max(n_sent / 1.5, 1.0))

    essay_generic_score = min(
        1.0,
        0.40 * generic_subject_density
        + 0.22 * prescriptive_density
        + 0.16 * abstract_noun_ratio
        + 0.12 * second_person_density
        + 0.10 * definitional_density,
    )
    # Genuine first-person experience strongly suppresses the essay signal.
    essay_generic_score *= 1.0 - 0.6 * personal_experiential_score

    return {
        "first_person_clause_ratio": round(first_person_clause_ratio, 4),
        "experiential_density": round(experiential_density, 4),
        "personal_experiential_score": round(personal_experiential_score, 4),
        "generic_subject_density": round(generic_subject_density, 4),
        "prescriptive_density": round(prescriptive_density, 4),
        "abstract_noun_ratio": round(abstract_noun_ratio, 4),
        "second_person_density": round(second_person_density, 4),
        "definitional_density": round(definitional_density, 4),
        "essay_generic_score": round(essay_generic_score, 4),
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text or "")
def _sentences(text: str) -> list[str]:
    parts = re.split(r"[.!?]+", text or "")
    return [p.strip() for p in parts if p.strip()]


def compute_semantic_specificity(transcript: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based specificity bundle from WhisperX transcript text.

    specificity_score: 0 = generic scripted prose, 1 = specific personal content
    generic_script_likelihood: inverse — for fusion as suspicion channel
    """
    text = str(transcript.get("transcript", "") or "").strip()
    words = _tokenize(text)
    n_words = max(len(words), 1)
    sentences = _sentences(text)
    n_sent = max(len(sentences), 1)

    # Domain-agnostic structural signals (topic-independent personal-vs-essay).
    da = _domain_agnostic_signals(text, words, sentences)

    # Proper nouns / named entities (capitalized mid-sentence or known tokens)
    proper = 0
    if _NAME_INTRO.search(text):
        proper += 2
    if _ROLE_TERMS.search(text):
        proper += 1
    for i, w in enumerate(words):
        if len(w) < 2:
            continue
        if w[0].isupper() and w not in _SENTENCE_STARTERS:
            if i == 0 or (i > 0 and not words[i - 1].endswith(".")):
                proper += 1
        elif w.lower() in ("vggnet", "vgg", "tensorflow", "pytorch", "keras", "whisper"):
            proper += 1

    proper_noun_density = min(1.0, proper / max(n_words / 12.0, 1.0))

    numbers = _NUMBER_PATTERN.findall(text)
    numeric_detail_density = min(1.0, len(numbers) / max(n_words / 15.0, 1.0))

    fp_action = len(_FIRST_PERSON_ACTION.findall(text))
    first_person_specificity = min(1.0, fp_action / max(n_sent / 2.0, 1.0))

    hedges = len(_HEDGE_PATTERN.findall(text))
    hedging_presence = min(1.0, hedges / max(n_sent / 1.5, 1.0))

    sent_lengths = [len(_tokenize(s)) for s in sentences if s]
    if len(sent_lengths) >= 2:
        mean_len = sum(sent_lengths) / len(sent_lengths)
        var_len = sum((x - mean_len) ** 2 for x in sent_lengths) / len(sent_lengths)
        cv = math.sqrt(var_len) / max(mean_len, 1.0)
        sentence_structure_variance = min(1.0, cv / 0.55)
    else:
        sentence_structure_variance = 0.35

    text_lower = text.lower()
    generic_hits = sum(1 for p in _GENERIC_PHRASES if p in text_lower)
    generic_phrase_density = min(1.0, generic_hits / 3.0)

    tech_def_hits = sum(1 for p in _TECH_DEFINITION_PHRASES if p in text_lower)
    tech_proper_hits = len(_TECH_PROPER.findall(text))
    tech_concept_hits = len(_TECH_CONCEPT_TERMS.findall(text))
    tech_concept_density = min(1.0, tech_concept_hits / max(n_words / 12.0, 1.0))
    definitional_passive = len(_DEFINitional_PASSIVE.findall(text))
    personal_narrative_hits = len(_PERSONAL_NARRATIVE.findall(text))
    personal_project_hits = len(_PERSONAL_PROJECT_ANCHOR.findall(text))

    personal_narrative_score = min(
        1.0,
        personal_narrative_hits / max(n_sent / 2.0, 1.0)
        + (0.25 if "I have been" in text or "I use" in text else 0.0),
    )
    # Generalize beyond the keyword regex: genuine first-person experience in ANY domain
    # (e.g. a self-introduction or a non-tech project) lifts the personal-narrative score.
    personal_narrative_score = min(
        1.0, max(personal_narrative_score, 0.85 * da["personal_experiential_score"])
    )

    # Dense generic-tech enumeration without personal narrative = memorized technical
    # script (catches first-person-framed recitations like "We implemented microservices
    # by separating ... authentication, real-time communication ..." that carry no
    # personal specifics). Self-suppresses for genuine personal answers.
    tech_enumeration_term = tech_concept_density * (1.0 - personal_narrative_score)

    memorized_technical_script_score = max(
        0.0,
        min(
            1.0,
            tech_def_hits * 0.22
            + min(1.0, tech_proper_hits / 3.0) * 0.28
            + definitional_passive * 0.15
            + (1.0 - min(1.0, personal_project_hits / 1.5)) * 0.20
            + (1.0 - personal_narrative_score) * 0.15
            + tech_enumeration_term * 0.30,
        ),
    )
    # Penalize genuine personal workflow answers ("I use Instagram metrics...")
    if personal_narrative_score >= 0.5 and personal_project_hits == 0:
        memorized_technical_script_score *= max(0.25, 1.0 - personal_narrative_score)

    specificity_score = (
        0.22 * proper_noun_density
        + 0.20 * numeric_detail_density
        + 0.18 * first_person_specificity
        + 0.14 * hedging_presence
        + 0.14 * sentence_structure_variance
        + 0.12 * personal_narrative_score
    )
    specificity_score = max(0.0, min(1.0, specificity_score - 0.18 * generic_phrase_density))

    generic_script_likelihood = max(
        0.0,
        min(
            1.0,
            (1.0 - specificity_score) * 0.40
            + generic_phrase_density * 0.25
            + memorized_technical_script_score * 0.35
            + (1.0 - hedging_presence) * 0.08,
        ),
    )

    # Domain-agnostic path: catch scripted essay/platitude prose in ANY topic, even when
    # none of the hardcoded phrase/keyword lists match. Combined with max() so this only
    # ADDS recall for non-keyword scripts and never lowers the keyword-based score.
    domain_agnostic_generic = max(
        0.0,
        min(
            1.0,
            (1.0 - specificity_score) * 0.28
            + da["essay_generic_score"] * 0.55
            + (1.0 - da["personal_experiential_score"]) * 0.10
            + (1.0 - hedging_presence) * 0.05,
        ),
    )
    generic_script_likelihood = max(generic_script_likelihood, domain_agnostic_generic)

    reasons: list[str] = []
    if proper_noun_density >= 0.35:
        reasons.append(f"proper nouns / named details ({proper} tokens)")
    if numeric_detail_density >= 0.35:
        reasons.append(f"numeric specificity ({len(numbers)} values)")
    if hedging_presence >= 0.25:
        reasons.append(f"spontaneous hedging ({hedges} markers)")
    if generic_phrase_density >= 0.35:
        reasons.append("generic essay / platitude phrasing detected")
    if da["essay_generic_score"] >= 0.45 and da["personal_experiential_score"] < 0.35:
        reasons.append(
            f"domain-agnostic essay structure {da['essay_generic_score']:.2f} — "
            f"impersonal/societal subjects, low first-person experience (topic-independent)"
        )
    if memorized_technical_script_score >= 0.45:
        reasons.append(
            f"memorized technical script patterns {memorized_technical_script_score:.2f} "
            f"(definition-style prose, low personal project anchor)"
        )
    if personal_narrative_score >= 0.45:
        reasons.append(
            f"personal narrative {personal_narrative_score:.2f} — first-person workflow/experience"
        )
    if specificity_score >= 0.48:
        reasons.append(f"specificity {specificity_score:.2f} — personal/detailed content")
    elif generic_script_likelihood >= 0.55:
        reasons.append(
            f"generic script likelihood {generic_script_likelihood:.2f} — lacks personal specifics"
        )

    return {
        "specificity_score": round(specificity_score, 4),
        "generic_script_likelihood": round(generic_script_likelihood, 4),
        "proper_noun_density": round(proper_noun_density, 4),
        "numeric_detail_density": round(numeric_detail_density, 4),
        "first_person_specificity": round(first_person_specificity, 4),
        "hedging_presence": round(hedging_presence, 4),
        "sentence_structure_variance": round(sentence_structure_variance, 4),
        "generic_phrase_density": round(generic_phrase_density, 4),
        "personal_narrative_score": round(personal_narrative_score, 4),
        "memorized_technical_script_score": round(memorized_technical_script_score, 4),
        "tech_definition_hits": tech_def_hits,
        "technical_concept_density": round(tech_concept_density, 4),
        # Domain-agnostic (topic-independent) structural features:
        "personal_experiential_score": da["personal_experiential_score"],
        "essay_generic_score": da["essay_generic_score"],
        "first_person_clause_ratio": da["first_person_clause_ratio"],
        "generic_subject_density": da["generic_subject_density"],
        "abstract_noun_ratio": da["abstract_noun_ratio"],
        "prescriptive_density": da["prescriptive_density"],
        "word_count": n_words,
        "reasons": reasons,
    }


def is_personal_natural_answer(spec: dict[str, Any]) -> bool:
    """True when transcript reads like genuine personal experience, not memorized script."""
    s = float(spec.get("specificity_score", 0.0))
    g = float(spec.get("generic_script_likelihood", 1.0))
    m = float(spec.get("memorized_technical_script_score", 0.0))
    p = float(spec.get("personal_narrative_score", 0.0))
    return (
        m < config.MEMORIZED_TECHNICAL_PROBABLE_MIN
        and p >= config.PERSONAL_NARRATIVE_CLEAR_MIN
        and s >= config.SPECIFICITY_CLEAR_MIN * 0.85
        and g <= 0.48
    )


def apply_specificity_to_status(
    status: str,
    spec: dict[str, Any],
    *,
    session_external_prior: float = 0.5,
    content_uniformity: float = 0.0,
    answer_index: int = 0,
    contrastive_external: float = 0.0,
    weighted_evidence: float = 0.0,
) -> tuple[str, list[str]]:
    """
    Person-independent verdict adjustment from transcript specificity.
    Does not replace contrastive — refines when acoustic signals are unreliable.
    """
    reasons: list[str] = []
    s = float(spec.get("specificity_score", 0.5))
    g = float(spec.get("generic_script_likelihood", 0.5))
    m = float(spec.get("memorized_technical_script_score", 0.0))
    p = float(spec.get("personal_narrative_score", 0.0))

    clear_min = config.SPECIFICITY_CLEAR_MIN
    probable_max = config.SPECIFICITY_GENERIC_PROBABLE_MIN
    mem_min = config.MEMORIZED_TECHNICAL_PROBABLE_MIN

    # Memorized technical script (AWS/WebSocket definition prose) — person-independent
    if m >= mem_min and p < config.PERSONAL_NARRATIVE_CLEAR_MIN:
        promote = status in ("CLEAR", "AMBIGUOUS")
        if (
            contrastive_external >= config.SOURCING_EXTERNAL_PROMOTE_MIN * 0.85
            or weighted_evidence >= config.AMBIGUOUS_MIN_WEIGHTED_EVIDENCE
            or g >= 0.40
        ):
            if promote:
                reasons.append(
                    f"memorized technical script {m:.2f} — textbook definition prose "
                    f"(external {contrastive_external:.2f}, evidence {weighted_evidence:.2f})"
                )
                return "PROBABLE_SCRIPT_READING", reasons

    if is_personal_natural_answer(spec):
        if status in ("PROBABLE_SCRIPT_READING", "AMBIGUOUS"):
            reasons.append(
                f"personal narrative {p:.2f} + specificity {s:.2f} — genuine experience/workflow"
            )
            return "CLEAR", reasons

    if s >= clear_min and g <= 0.42 and m < mem_min:
        if status == "PROBABLE_SCRIPT_READING":
            reasons.append(
                f"semantic specificity {s:.2f} — personal/named/numeric content overrides acoustic false positive"
            )
            return "CLEAR", reasons
        if status == "AMBIGUOUS":
            reasons.append(f"semantic specificity {s:.2f} supports CLEAR")
            return "CLEAR", reasons

    if g >= probable_max and s <= 0.32:
        if status in ("CLEAR", "AMBIGUOUS"):
            reasons.append(
                f"generic script likelihood {g:.2f} — essay/platitude vocabulary without personal specifics"
            )
            new_status = "PROBABLE_SCRIPT_READING"
            if (
                answer_index >= 2
                and session_external_prior >= config.SESSION_FEEDFORWARD_P_MIN
            ):
                reasons.append(
                    f"session prior P(external)={session_external_prior:.2f} reinforces generic-content suspicion"
                )
            if content_uniformity >= config.CONTENT_UNIFORMITY_SUSPICIOUS_MIN:
                reasons.append(
                    f"cross-answer content uniformity {content_uniformity:.2f} — similar vocabulary across questions"
                )
            return new_status, reasons

    return status, reasons
