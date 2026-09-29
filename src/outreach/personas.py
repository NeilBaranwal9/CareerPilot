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
- Never ask for a call or meeting. No buzzwords (passionate, leveraging, cutting-edge, synergy, innovative), no
  "which aligns with", no "just checking in" filler, and don't repeat sentences from the original email.
- Sound like the same student who wrote the original: plain, specific, slightly informal is fine.
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

How to write this email (follow silently; never mention these instructions in the email):
- Write like a technically strong student who personally spent five minutes researching the company: natural,
  concise, specific, professional but conversational, confident without sounding entitled. Slightly informal is fine.
  Not a sales template, not a cover letter, no marketing language, no emojis, no "Dear Sir/Madam".
- About {min_words}-{max_words} words. Do not add length just to add personalization.
- Start the body with "<p>Hi {first_name},</p>". Then a flexible flow (vary it; it is not a template):
  1. Opening: ONE specific, verifiable fact from the context (a product, an engineering or product post, a recent
     development, the job posting, or a technology they mention). {opening_instruction}
  2. Who I am: briefly, my degree, year and college exactly as my profile/resume state them, and that I'm looking for
     a {target_role} internship.
  3. The single most relevant experience or project from my resume, in plain words: what the system did and for
     whom, with at most ONE technology name. Do not paste resume bullets or list tools.
     Too technical: "I built a zero-allocation Rust UDP telemetry ingestion daemon with Prometheus/Grafana."
     Plain: "At Jio Platforms I worked on the system that collects network telemetry and traces faults in real time."
  4. At most ONE sentence linking that experience to this company or role, only if the link is natural. Never an
     "aligns with" / "matches my" sentence, and never a second sentence about how the company fits me.
  5. The ask: {ask_instruction}
  (The example sentences above only show the level of detail. Write your own wording; don't reuse theirs.)
- {persona_focus} Don't pretend to know the recipient's responsibilities beyond their title.
- Never claim what the company needs, values, prioritises or focuses on unless the context says so explicitly. Never
  turn a technical fact into praise. Bad: "Your innovative infrastructure aligns perfectly with my passion for scalable
  technology." Better: "I saw your engineering post on using EKS and Spot Instances. I've worked on high-volume
  telemetry systems at Jio, so the infrastructure side of that caught my attention."
- Never invent projects, technologies, company initiatives, recipient responsibilities, relationships, job openings,
  metrics or company priorities. If the context is thin, write a simpler email rather than inventing personalization.
- Avoid stock constructions such as "which aligns with", "I am keen to", "support X's growth", "data-driven",
  "scalable", "I was impressed by", "I would appreciate any referral or guidance". Never use: passionate, leveraging,
  cutting-edge, synergy, innovative, revolutionary, world-class, "excited to apply". No exaggerated praise.
- Never ask for a call or meeting, never ask when they are free, never say "please review my resume".
{avoid_block}- Before answering, check: would this sound normal if a VIT student personally wrote it after five minutes
  of research? If not, rewrite it simpler and more conversational.
- Use contractions where natural (I'm, I've, I'd). Keep sentences short.
- Subject: short and specific to this email (the product, topic or team), never "Internship Inquiry" or similar.
- End with a short sign-off line followed by my name, {user_name}.
- Return the subject line and an HTML body using <p> tags only.
"""

BANNED_PHRASES = [
    "passionate", "leverag", "cutting-edge", "cutting edge", "synergy", "synergies", "revolutionary", "innovative",
    "eager to contribute", "i am passionate", "i'm passionate", "excited to apply", "thrilled", "dream company",
    "game-changing", "game changer", "world-class", "delighted to",
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


def check_email_style(
    body_html: str, user_name: str, min_words: int = 120, max_words: int = 170, tolerance: int = 15
) -> list[str]:
    """Deterministic reply-rate style violations (empty list = OK). Length allows a small tolerance around the target."""
    issues: list[str] = []
    plain = html_to_plain(body_html)
    lowered = plain.lower()
    words = email_body_word_count(body_html, user_name)
    if words < min_words - tolerance or words > max_words + tolerance:
        issues.append(f"body is {words} words; aim for about {min_words}-{max_words}")
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            issues.append(f"avoid the buzzword '{phrase}'")
    match = MEETING_ASKS.search(plain)
    if match:
        issues.append(f"do not ask for a meeting/call ('{match.group(0)}'); ask for guidance or a referral instead")
    return issues
