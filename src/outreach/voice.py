"""
Voice & anti-template layer for outreach emails.

- Persona-aware focus and asks (recruiters are asked about the process, not for referrals; founders get short notes).
- Opening styles chosen from the research that actually exists, rotated so consecutive emails don't start alike.
- Deterministic checks that trigger a rewrite: stock phrases repeated within an email or across recent emails,
  openings/asks copied from recent emails, unsupported claims about what the company needs/values/focuses on,
  invented job openings, invented relationships, and assumed recipient responsibilities.
"""

import re
from collections import Counter
from dataclasses import dataclass

from src.outreach.personas import html_to_plain

# ---------------------------------------------------------------------------
# Persona focus & asks
# ---------------------------------------------------------------------------

PERSONA_FOCUS: dict[str, str] = {
    "recruiter": "The recipient is a recruiter: focus on the role, your fit, internship availability and the "
    "application process. Keep technical detail light.",
    "generic_inbox": "This goes to a careers inbox: name the internship role clearly and keep it short and plain.",
    "product_manager": "The recipient works in product: focus on product analytics, user/product problems, "
    "experimentation and metrics, and your analytical experience only where the resume supports it.",
    "product_lead": "The recipient leads product: focus on product analytics, user/product problems, experimentation "
    "and metrics, and your analytical experience only where the resume supports it.",
    "data_lead": "The recipient leads data/analytics: focus on data work, analysis and the systems behind it.",
    "qa_lead": "The recipient leads quality/testing: focus on testing, reliability and debugging work you actually did.",
    "engineering_manager": "The recipient is an engineering manager: focus on technical systems, data and engineering "
    "work, and how it relates to the product or engineering area in the context.",
    "hiring_manager": "The recipient is hiring for this area: focus on the concrete work that matches the role.",
    "tech_lead": "The recipient is a senior engineer: peer-to-peer and technical, focused on systems you built.",
    "vp_engineering": "The recipient leads engineering: focus on technical systems and engineering work.",
    "cto": "The recipient is the CTO: technical and brief; focus on the engineering work most related to theirs.",
    "founder": "The recipient is a founder: be shorter and direct. Say why their problem/product is interesting to you "
    "and what you could contribute.",
    "executive": "The recipient is a senior executive: be brief and plain.",
}
DEFAULT_FOCUS = PERSONA_FOCUS["engineering_manager"]

ASK_BY_PERSONA: dict[str, str] = {
    "recruiter": "Ask whether they are hiring interns for this kind of role and how to apply or be considered. "
    "Do not ask a recruiter for a referral.",
    "generic_inbox": "Ask for the note to be passed to the relevant team, or how to apply.",
    "founder": "Ask directly whether they are taking interns, and who you should talk to if it isn't them. "
    "No referral request.",
    "cto": "Ask whether the team is taking interns and who you should talk to. No referral request.",
    "executive": "Ask to be pointed to the right person for internships. No referral request.",
}
DEFAULT_ASK = (
    "Ask whether their team takes interns in this area. You may ask them to point you to the right person or process, "
    "or for a referral if they seem well placed to give one. One simple sentence."
)

SHORT_PERSONAS = {"founder", "cto", "executive", "generic_inbox"}


def persona_focus(persona: str | None) -> str:
    return PERSONA_FOCUS.get(persona or "", DEFAULT_FOCUS)


def persona_ask(persona: str | None) -> str:
    return ASK_BY_PERSONA.get(persona or "", DEFAULT_ASK)


# Company-level (speculative) inquiries: no posting to discuss, so they are shorter.
COMPANY_OUTREACH_WORDS = (100, 150)
_COMPANY_ASK_TAIL: dict[str, str] = {
    "recruiter": " Ask how to be considered. Do not ask a recruiter for a referral.",
    "generic_inbox": " Ask for the note to be passed to the relevant team.",
    "founder": " If it isn't them, ask who you should talk to. No referral request.",
    "cto": " If it isn't them, ask who you should talk to. No referral request.",
    "executive": " Ask to be pointed to the right person. No referral request.",
}


def company_outreach_ask(persona: str | None, ask_about_openings: bool = True) -> str:
    """The ask for a company-level inquiry: a genuine question about current/upcoming internships, never an assumption."""
    if not ask_about_openings:
        return persona_ask(persona)
    return (
        "Ask, as a genuine question, whether they have any current or upcoming internship opportunities in these areas. "
        "Don't assume an opening exists." + _COMPANY_ASK_TAIL.get(persona or "", " If they aren't the right person, ask who is.")
    )


def word_target(persona: str | None, min_words: int, max_words: int) -> tuple[int, int]:
    """Founders and executives get the short end of the range."""
    if persona in SHORT_PERSONAS:
        return min_words, min(max_words, min_words + 35)
    return min_words, max_words


# ---------------------------------------------------------------------------
# Opening styles
# ---------------------------------------------------------------------------

OPENING_STYLES: dict[str, str] = {
    "engineering_post": 'Open by referring to their engineering or product post/blog topic from the context, e.g. '
    '"I saw your post on ...".',
    "news": 'Open with the recent development from the context, e.g. "I was reading about ..." or "I noticed that ...".',
    "job_posting": "Open with the specific internship posting from the context and what it involves.",
    "product_direct": "Open by naming their product and what it does, with no generic introduction.",
    "caught_attention": 'Open with the specific thing that caught your attention, e.g. "Your work on ... caught my '
    'attention."',
    "reason_first": 'Open by saying why you are writing, e.g. "I wanted to reach out after seeing ...".',
    "plain": "The research is thin: open with one plain sentence about what the company does. Do not embellish.",
}


@dataclass
class ResearchSignals:
    has_post: bool
    has_news: bool
    has_real_job: bool
    has_product: bool

    @property
    def thin(self) -> bool:
        return not (self.has_post or self.has_news or self.has_real_job or self.has_product)


def research_signals(research: dict[str, object], recent_news: object, job_source: str | None, description: str | None) -> ResearchSignals:
    culture = str(research.get("culture") or "").lower()
    news_titles = " ".join(str(n.get("title", "")) for n in recent_news if isinstance(n, dict)) if isinstance(recent_news, list) else ""
    has_post = bool(re.search(r"\b(blog|post|engineering|tech talk|article)\b", culture + " " + news_titles.lower()))
    has_news = bool(news_titles.strip())
    products = research.get("products")
    has_product = bool(products) or bool(description and len(description) > 40)
    has_real_job = job_source not in (None, "speculative", "company_outreach", "pasted")
    return ResearchSignals(has_post, has_news, has_real_job, has_product)


def choose_opening(signals: ResearchSignals, recent_styles: list[str]) -> str:
    """Picks an opening style supported by the research, avoiding the styles used in the last few emails."""
    if signals.thin:
        return "plain"
    candidates: list[str] = []
    if signals.has_real_job:
        candidates.append("job_posting")
    if signals.has_post:
        candidates.append("engineering_post")
    if signals.has_news:
        candidates += ["news", "reason_first"]
    if signals.has_product:
        candidates += ["product_direct", "caught_attention"]
    recent = [s for s in recent_styles if s][:3]
    for style in candidates:
        if style not in recent:
            return style
    usage = Counter(recent_styles)
    return min(candidates, key=lambda s: (usage[s], candidates.index(s)))


# ---------------------------------------------------------------------------
# Repetition & template detection
# ---------------------------------------------------------------------------

WATCHED_PHRASES: dict[str, str] = {
    "which aligns with": r"\bwhich (?:\w+ )?aligns? with\b",
    "aligns with": r"\baligns? (?:well |perfectly |closely |nicely )?with\b",
    "I am keen to": r"\b(?:i am|i'm) (?:really |very )?keen to\b",
    "support X's growth": r"\bsupport\b[^.]{0,40}\bgrowth\b",
    "data-driven": r"\bdata[- ]driven\b",
    "scalable": r"\bscalab(?:le|ility)\b",
    "I was impressed by": r"\b(?:i was|i'm|i am) (?:really |very |quite )?impressed\b",
    "I would appreciate any referral or guidance": r"\bappreciate any (?:referral|guidance)\b",
    "I came across": r"\bi came across\b",
    "caught my attention": r"\bcaught my (?:attention|eye)\b",
    "is why I am interested": r"\bis why (?:i am|i'm) (?:particularly |especially |really )?interested\b",
    "interested in contributing": r"\binterested in contributing\b",
    "matches my": r"\b(?:matches|match) my\b",
    "the kind of X that": r"\bthe kind of\b",
}
STRICT_PHRASES = {
    "which aligns with", "aligns with", "support X's growth", "is why I am interested", "interested in contributing",
    "matches my",
}  # avoid if used in any of the last 3 emails

PRAISE_WORDS = [
    "amazing", "incredible", "remarkable", "fantastic", "groundbreaking", "visionary", "best-in-class",
    "aligns perfectly", "align perfectly", "perfectly aligned", "passion", "truly", "dream company", "inspiring",
]
FABRICATED_RELATIONSHIP = re.compile(
    r"\b(?:referred (?:me|by)|suggested (?:that )?i (?:reach out|contact|email|write)|we met|as we discussed|"
    r"mutual (?:friend|connection)|you (?:may )?remember me|following up on our|our (?:recent )?conversation)\b",
    re.IGNORECASE,
)
JOB_OPENING_CLAIM = re.compile(
    r"\b(?:your|the) (?:\w+ ){0,4}(?:opening|posting|job post|listing|vacancy)\b|"
    r"\b(?:role|position) (?:you|your team) (?:posted|advertised|listed)\b|"
    r"\b(?:you(?:'re| are)|your team is|the team is|you are currently) (?:actively )?hiring\b",
    re.IGNORECASE,
)
# Only checked for company-level inquiries, where no opening is known at all.
COMPANY_LEVEL_CLAIM = re.compile(
    r"\b(?:apply(?:ing)?|application) (?:for|to) (?:the|your) (?:[\w/-]+ ){0,4}(?:role|position|internship|opening)\b|"
    r"\b(?:your|the) open (?:[\w/-]+ ){0,3}(?:roles?|positions?|internships?)\b|"
    r"\b(?:you(?:'re| are)|your team is|the team is) (?:currently )?(?:growing|expanding|scaling|building out|looking for)\b|"
    r"\b(?:plans?|planning) to hire\b",
    re.IGNORECASE,
)
INQUIRY_ASK = re.compile(
    r"\?|\b(?:whether|if)\b[^.]{0,100}\b(?:intern|interns|internship|internships|opportunit\w*|openings?)\b",
    re.IGNORECASE,
)
COMPANY_INTENT = re.compile(
    r"\b(?:your|the (?:team|company)'s|[a-z0-9&.]+'s) (?:focus|commitment|emphasis|priority|priorities|mission|"
    r"need|needs|vision|values|culture of)\s+(?:on|to|for|of|around)?\s*(?P<obj>[^.,;:!?]{3,60})|"
    r"\b(?:you|your team|the company|the team) (?:values?|focus(?:es)? on|prioriti[sz]es?|needs?|is committed to|"
    r"cares? about)\s+(?P<obj2>[^.,;:!?]{3,60})",
    re.IGNORECASE,
)
RESPONSIBILITY_CLAIM = re.compile(
    r"\b(?:you (?:lead|run|manage|head|own|oversee|built)|your team (?:builds|owns|runs|works on|is building))\s+"
    r"(?P<obj>[^.,;:!?]{3,50})",
    re.IGNORECASE,
)
_STOP = {
    "the", "a", "an", "and", "of", "to", "for", "in", "on", "with", "your", "their", "its", "our", "that", "this",
    "is", "are", "at", "by", "as", "be", "it", "how", "what", "which", "more", "most", "new", "team", "company",
}


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9+#-]{2,}", text.lower()) if w not in _STOP]


def _supported(phrase: str, source: str, min_share: float = 0.5) -> bool:
    words = _content_words(phrase)
    if not words:
        return True
    src = source.lower()
    hits = sum(1 for w in words if w in src or w.rstrip("s") in src)
    return hits / len(words) >= min_share


def _body_text(body_html: str, user_name: str = "") -> str:
    text = html_to_plain(body_html)
    for ch in ("‐", "‑", "‒", "–", "—"):
        text = text.replace(ch, "-")
    text = text.replace("’", "'")
    if user_name and user_name in text:
        text = text[: text.rfind(user_name)]
    text = re.sub(
        r"\b(best regards|kind regards|warm regards|regards|best|thanks|thank you|sincerely|cheers)[,.!]?\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    return re.sub(r"^\s*(hi|hello|dear)\b[^,\n]*,?\s*", "", text.strip(), flags=re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 3]


def _skeleton(sentence: str, words: int = 4) -> str:
    return " ".join(re.findall(r"[a-z']+", sentence.lower())[:words])


def _paragraph_leads(body_html: str) -> list[str]:
    paragraphs = [html_to_plain(p) for p in re.findall(r"<p[^>]*>(.*?)</p>", body_html or "", flags=re.I | re.S)]
    return [_skeleton(p, 2) for p in paragraphs if p.strip() and not re.match(r"^(hi|hello|best|thanks|regards)\b", p.strip(), re.I)]


TECH_TERMS = {
    "daemon", "ingestion", "zero-allocation", "udp", "tcp", "observability", "pipeline", "pipelines", "inference",
    "topology", "topologies", "throughput", "high-throughput", "low-latency", "latency", "microservices", "rust",
    "python", "prometheus", "grafana", "kafka", "kubernetes", "docker", "llm", "llms", "ollama", "toml", "neo4j",
    "fastapi", "streamlit", "gnn", "gnns", "graph", "telemetry", "pyside6", "api", "apis", "rest", "sql", "aws",
    "gcp", "react", "node.js", "typescript", "java", "c++", "redis", "postgresql", "eks", "spot", "gemini",
    "root-cause", "deterministic", "real-time", "machine-readable", "etl", "embedding", "embeddings", "rag",
}
GENERIC_SUBJECT = re.compile(
    r"^\s*(re:\s*)?((software|backend|data|product)\s+)?(engineer(ing)?\s+)?(intern(ship)?)\s*"
    r"(inquiry|enquiry|opportunit(y|ies)|application|request|interest)?\s*$",
    re.IGNORECASE,
)


LINKING_SENTENCE = re.compile(
    r"\b(reminded me of|is why|which is why|that's why|that is why|seems (directly )?(useful|relevant)|"
    r"relevant to|similar to (what|the)|relates to|connects to|close to (what|the)|could be useful for|"
    r"would be useful for|could help (with|your)|fits (well )?with|in line with)\b",
    re.IGNORECASE,
)


def linking_sentences(text: str) -> list[str]:
    return [s for s in _sentences(text) if LINKING_SENTENCE.search(s)]


def jargon_sentences(text: str, limit: int = 3) -> list[str]:
    """Sentences that read like a tech-stack list (limit+ technical terms)."""
    flagged = []
    for sentence in _sentences(text):
        words = re.findall(r"[a-z0-9+.#-]+", sentence.lower().replace("/", " "))
        hits = {w for w in words if w in TECH_TERMS}
        if len(hits) >= limit:
            flagged.append(sentence)
    return flagged


def check_subject(subject: str, recent_subjects: list[str]) -> list[str]:
    issues = []
    if GENERIC_SUBJECT.match(subject or ""):
        issues.append(f"subject '{subject}' is generic; mention the specific topic, product or team")
    lowered = (subject or "").strip().lower()
    if lowered and any(lowered == (s or "").strip().lower() for s in recent_subjects[:5]):
        issues.append(f"subject '{subject}' repeats a recent email's subject")
    return issues


def phrase_counts(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {name: len(re.findall(pattern, lowered)) for name, pattern in WATCHED_PHRASES.items()}


def overused_recent_phrases(recent_bodies: list[str]) -> list[str]:
    """Stock phrases used in 2+ of the recent emails (or strict ones used in any of the last 3)."""
    usage: Counter[str] = Counter()
    strict_recent: set[str] = set()
    for i, body in enumerate(recent_bodies[:5]):
        for name, count in phrase_counts(html_to_plain(body)).items():
            if count:
                usage[name] += 1
                if i < 3 and name in STRICT_PHRASES:
                    strict_recent.add(name)
    return sorted({name for name, n in usage.items() if n >= 2} | strict_recent)


def recent_patterns_summary(recent_bodies: list[str], user_name: str) -> str:
    """Short description of how the last emails opened and asked, so the writer can avoid repeating them."""
    lines = []
    for i, body in enumerate(recent_bodies[:3], start=1):
        sentences = _sentences(_body_text(body, user_name))
        if not sentences:
            continue
        lines.append(f'{i}. opened: "{sentences[0][:90]}" | asked: "{sentences[-1][:90]}"')
    return "\n".join(lines)


def recent_sentence_starts(recent_bodies: list[str], user_name: str, limit: int = 8) -> list[str]:
    """Opening words of the middle sentences of recent emails (to be avoided, not imitated)."""
    starts: list[str] = []
    for body in recent_bodies[:3]:
        for sentence in _sentences(_body_text(body, user_name))[1:-1]:
            words = sentence.split()[:6]
            if len(words) >= 5:
                start = " ".join(words) + " ..."
                if start not in starts:
                    starts.append(start)
    return starts[:limit]


def check_voice(body_html: str, recent_bodies: list[str], user_name: str = "") -> list[str]:
    """Template-like writing: stock phrases, praise, and openings/asks/structure copied from recent emails."""
    issues: list[str] = []
    text = _body_text(body_html, user_name)
    counts = phrase_counts(text)
    for name, count in counts.items():
        if count >= 2:
            issues.append(f"'{name}' is used {count} times in this email; say it differently")
    for name in overused_recent_phrases(recent_bodies):
        if counts.get(name):
            issues.append(f"'{name}' was used in recent emails; rephrase without it")
    lowered = text.lower()
    for word in PRAISE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            issues.append(f"exaggerated praise ('{word}'); state the fact plainly")

    links = linking_sentences(text)
    if len(links) >= 2:
        issues.append(
            f"{len(links)} sentences connect your experience to the company; keep at most one natural link "
            f"(e.g. drop \"{links[-1][:70]}...\")"
        )
    for sentence in jargon_sentences(text):
        issues.append(
            f"reads like a tech-stack list: \"{sentence[:90]}...\"; say what you did in plain words "
            "(what the system did and for whom), with one technology name at most"
        )

    sentences = _sentences(text)
    if sentences and recent_bodies:
        recent_skeletons = {
            _skeleton(s, 5) for body in recent_bodies[:3] for s in _sentences(_body_text(body, user_name))[1:-1]
        }
        for sentence in sentences[1:-1]:
            if re.search(r"\bstudent\b|\bb\.?tech\b|\bdegree\b", sentence, re.IGNORECASE):
                continue  # the who-I-am sentence carries fixed facts; the prompt asks to vary it, but don't force a rewrite
            skeleton = _skeleton(sentence, 5)
            if len(skeleton.split()) >= 5 and skeleton in recent_skeletons:
                issues.append(f"the sentence \"{sentence[:70]}...\" is built like one in a recent email; rephrase it")
        opening, ask = _skeleton(sentences[0]), _skeleton(sentences[-1], 5)
        leads = _paragraph_leads(body_html)
        for body in recent_bodies[:3]:
            other = _sentences(_body_text(body, user_name))
            if not other:
                continue
            if opening and opening == _skeleton(other[0]):
                issues.append(f"the opening ('{sentences[0][:60]}...') copies a recent email; open differently")
                break
        for body in recent_bodies[:2]:
            other = _sentences(_body_text(body, user_name))
            if other and ask and ask == _skeleton(other[-1], 5):
                issues.append("the closing ask is worded exactly like a recent email; vary it")
                break
        previous_leads = _paragraph_leads(recent_bodies[0])
        if len(leads) >= 3 and leads == previous_leads:
            issues.append("the paragraph structure is identical to the previous email; vary the flow")
    return list(dict.fromkeys(issues))


def check_company_inquiry(body_html: str, user_name: str = "") -> list[str]:
    """A company-level inquiry must actually ask about current/upcoming internship opportunities."""
    if INQUIRY_ASK.search(_body_text(body_html, user_name)):
        return []
    return ["company-level inquiry must ask whether there are current or upcoming internship opportunities"]


def check_grounding(
    body_html: str,
    research_text: str,
    recipient_text: str,
    has_real_job: bool,
    user_name: str = "",
    company_level: bool = False,
) -> list[str]:
    """
    Company/recipient statements that the research does not support. `company_level` (no known opening) also rejects
    "applying for the X role", "your open positions" and stated hiring plans.
    """
    issues: list[str] = []
    text = _body_text(body_html, user_name)
    for match in COMPANY_INTENT.finditer(text):
        obj = match.group("obj") or match.group("obj2") or ""
        if not _supported(obj, research_text):
            issues.append(f"unsupported claim about what the company focuses on/needs: \"{match.group(0).strip()[:80]}\"")
    for match in RESPONSIBILITY_CLAIM.finditer(text):
        if not _supported(match.group("obj"), recipient_text + " " + research_text):
            issues.append(f"assumes the recipient's responsibilities: \"{match.group(0).strip()[:80]}\"")
    if not has_real_job:
        for match in JOB_OPENING_CLAIM.finditer(text):
            before = text[max(0, match.start() - 12) : match.start()].lower()
            if re.search(r"\b(if|whether)\b", before):
                continue  # "if your team is hiring" is a question, not a claim
            issues.append(f"mentions a job opening that the research does not show: \"{match.group(0).strip()}\"")
    if company_level:
        for match in COMPANY_LEVEL_CLAIM.finditer(text):
            before = text[max(0, match.start() - 12) : match.start()].lower()
            if re.search(r"\b(if|whether)\b", before):
                continue
            issues.append(
                "company-level inquiry claims a specific opening or hiring plan that is not known: "
                f"\"{match.group(0).strip()}\""
            )
    rel = FABRICATED_RELATIONSHIP.search(text)
    if rel:
        issues.append(f"implies a relationship that does not exist: \"{rel.group(0)}\"")
    return list(dict.fromkeys(issues))
