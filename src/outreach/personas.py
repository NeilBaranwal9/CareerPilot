"""Persona-specific writing guidelines, tones, prompt formatting and fallback templates."""

import html
import re
from typing import Any

PERSONA_GUIDELINES: dict[str, str] = {
    "founder": (
        "Recipient is a FOUNDER. Aim for the short end (80-110 words). The observation should be about their product "
        "or business, and the connection sentence about the problem they are solving."
    ),
    "cto": (
        "Recipient is the CTO. Make the observation about an engineering or AI initiative if one is known, and connect "
        "your experience to that kind of work in plain words (outcomes, not tool lists)."
    ),
    "vp_engineering": (
        "Recipient leads engineering. Make the observation about their engineering or product direction and connect "
        "your experience to the problems their teams work on."
    ),
    "engineering_manager": (
        "Recipient is an ENGINEERING MANAGER. Make the observation about their product or engineering work and connect "
        "your most relevant experience to what their team builds."
    ),
    "hiring_manager": (
        "Recipient is a HIRING MANAGER. If a specific internship role is known, name it and connect your experience "
        "to what that role needs."
    ),
    "tech_lead": (
        "Recipient is a senior engineer. Peer-to-peer and plain; the ask is for guidance or a referral to the right person."
    ),
    "recruiter": (
        "Recipient is a RECRUITER. Name the internship role or role family clearly and state your degree and year "
        "exactly as the resume states them; mention the resume is attached."
    ),
    "generic_inbox": (
        "Recipient is the company's careers inbox. Name the internship role family, note the attached resume, and ask "
        "for it to be forwarded to the right team."
    ),
    "executive": (
        "Recipient is a senior executive. Keep to the short end; ask to be pointed to the right person for internships."
    ),
}
DEFAULT_GUIDELINE = PERSONA_GUIDELINES["engineering_manager"]

TONE_DESCRIPTIONS: dict[str, str] = {
    "formal": "formal and polished, no slang or exclamation marks",
    "warm": "warm, genuine and professional; conversational but not casual",
    "concise": "extremely concise and direct; every sentence earns its place",
    "enthusiastic": "energetic and enthusiastic about their product, without sounding over the top",
    "casual": "friendly and casual, like a message to a peer",
}

DEFAULT_FOLLOWUP_PROMPT = """Write {followup_count} short follow-up emails for a cold outreach that received no reply yet.
They are sent as replies in the same email thread to {contact_name} ({contact_role}) at {company_name} about {role_name}.

Original email:
{initial_email}

Context you may use (do not invent anything beyond it):
- Company: {company_description}
- Recent news: {recent_news}
- My highlights, most relevant first:
{highlights}
- My resume (the ONLY source of facts about me):
{resume_summary}

Rules:
- Every statement about me must be supported by my resume; never attribute a project to the wrong event or employer,
  and never add location, relocation or availability claims.
- Tone: {tone}.
- Follow-up #1 (sent {followup_1_days} days later): 40-70 words; politely bump the thread and add ONE different,
  relevant item from my resume that was not in the original email, described by its outcome (not its tech stack).
- Follow-up #2 (sent {followup_2_days} days later): 30-50 words; close the loop gracefully and ask whether someone
  else is the right person to ask about internships.
- Never ask for a call or meeting. No buzzwords (passionate, leveraging, cutting-edge, synergy, innovative).
- Plain, friendly HTML using <p> tags. Sign off with my name: {user_name}. No placeholders, no subject lines.
"""


class _SafeDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def safe_format(template: str, **values: Any) -> str:
    """str.format that leaves unknown placeholders untouched (keeps older config templates working)."""
    return template.format_map(_SafeDict(values))


def guideline_for(persona: str | None) -> str:
    return PERSONA_GUIDELINES.get(persona or "", DEFAULT_GUIDELINE)


def tone_for(persona: str | None, default_tone: str, persona_tones: dict[str, str]) -> str:
    tone = persona_tones.get(persona or "", default_tone)
    if persona == "founder" and persona not in persona_tones and default_tone == "warm":
        tone = "concise"
    return tone


def describe_tone(tone: str) -> str:
    return TONE_DESCRIPTIONS.get(tone, tone)


def first_name(full_name: str | None) -> str:
    if not full_name:
        return "there"
    token = full_name.strip().split()[0]
    if token.lower() in ("hiring", "recruiting", "talent", "hr", "team", "careers"):
        return "Hiring Team"
    return token


def html_to_plain(body_html: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", body_html or "", flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def fallback_followups(
    contact_name: str | None, company_name: str, role_name: str, user_name: str, highlight: str | None, count: int
) -> list[str]:
    greeting = f"<p>Hi {html.escape(first_name(contact_name))},</p>"
    extra = f" One more relevant note: {html.escape(highlight)}." if highlight else ""
    templates = [
        f"{greeting}<p>Just bumping this up in case it got buried.{extra} If there is a suitable internship at "
        f"{html.escape(company_name)}, I'd be grateful for any guidance or a referral.</p>"
        f"<p>Best,<br>{html.escape(user_name)}</p>",
        f"{greeting}<p>I'll close the loop here so I don't crowd your inbox. If someone else is the right person to "
        f"talk to about {html.escape(role_name)} roles at {html.escape(company_name)}, I'd be grateful for a pointer. "
        f"Thanks for your time!</p><p>Best,<br>{html.escape(user_name)}</p>",
    ]
    return templates[:count] if count <= len(templates) else templates + templates[-1:] * (count - len(templates))


PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:your|insert|candidate|company|recipient|name|my|contact|role)[^\]]{0,40}\]|\{[a-z_]{3,30}\}|<(?:company|name)>",
    re.IGNORECASE,
)


def find_placeholders(text: str) -> list[str]:
    return sorted(set(PLACEHOLDER_PATTERN.findall(text or "")))


# ---------------------------------------------------------------------------
# Reply-rate style rules (enforced by prompt and checked deterministically)
# ---------------------------------------------------------------------------

EMAIL_STYLE_RULES = """

Writing style (optimise for a reply, not for showing off technology):
- 80-180 words in the body. Simple, professional English from a student writing to a professional. Not a cover
  letter, not a sales email. Concrete statements over adjectives.
- Structure, one short paragraph each:
  1. Hi {first_name}, then one company-specific observation that explains why I am writing (a product, launch,
     engineering or AI initiative, or business development from the context; never invent news).
  2. One sentence about me.
  3. One sentence connecting my experience to their company, in terms of impact/outcomes.
  4. My interest in relevant internship opportunities.
  5. A simple guidance or referral request, e.g. "If there's a suitable opening, I'd be grateful for a referral." or
     "I'd appreciate any guidance on relevant opportunities."
- Mention only the 1-2 most relevant items from my profile (e.g. Software Engineering Intern at Jio Platforms,
  Goldman Sachs India Hackathon finalist, Team Ignition rocket telemetry software, AI/ML specialisation at VIT Chennai,
  a relevant project). Describe them by what they achieved, not by tool lists.
  Bad: "I built a zero-allocation Rust UDP telemetry daemon with Prometheus/Grafana observability."
  Better: "At Jio Platforms, I worked on systems for large-scale telemetry collection and analysis."
- Never use: passionate, leveraging, cutting-edge, synergy, revolutionary, innovative, "eager to contribute".
- Never ask for a call or meeting, never ask when they are free, never say "please review my resume".
- Every statement about me must be supported by my resume. Company facts must come from the context above.
- Return a short, plain subject line and an HTML body using <p> tags only.
"""

BANNED_PHRASES = [
    "passionate", "leverag", "cutting-edge", "cutting edge", "synergy", "synergies", "revolutionary", "innovative",
    "eager to contribute",
]
MEETING_ASKS = re.compile(
    r"schedule (a |an )?(quick |brief |short )?(call|meeting|chat)|when are you (free|available)|hop on a call|"
    r"\b1[05][- ]minute|quick call|brief call|set up a (call|meeting)|please review my resume|book a (call|slot)",
    re.IGNORECASE,
)


def email_body_word_count(body_html: str, user_name: str) -> int:
    """Words in the body, excluding the sign-off/signature (everything from the sender's name on)."""
    text = html_to_plain(body_html)
    if user_name and user_name in text:
        text = text[: text.rfind(user_name)]
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b(best|regards|best regards|thanks|thank you|sincerely|cheers)[,.!]?\s*$", " ", text.strip(), flags=re.I)
    return len(re.findall(r"[A-Za-z0-9][\w'’.-]*", text))


def check_email_style(body_html: str, user_name: str, min_words: int = 80, max_words: int = 180) -> list[str]:
    """Deterministic reply-rate style violations (empty list = OK)."""
    issues: list[str] = []
    plain = html_to_plain(body_html)
    lowered = plain.lower()
    words = email_body_word_count(body_html, user_name)
    if words < min_words or words > max_words:
        issues.append(f"body is {words} words; keep it between {min_words} and {max_words}")
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            issues.append(f"avoid the buzzword '{phrase}'")
    match = MEETING_ASKS.search(plain)
    if match:
        issues.append(f"do not ask for a meeting/call ('{match.group(0)}'); ask for guidance or a referral instead")
    return issues
