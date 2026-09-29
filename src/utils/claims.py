"""
Deterministic hallucination guard for resumes and emails.

`find_unsupported_claims(text, source)` flags statements in `text` that introduce facts absent from `source`:
awards / rankings / hackathon results, year of study or graduation year, CGPA/GPA, numbers, and technologies.
For resumes, `source` is the original resume. For emails it is the resume plus the company research, job
description and your own profile notes (so company facts and your configured highlights are allowed).
"""

import re

from src.intel.classify import keyword_in

AWARD_PATTERNS: list[tuple[str, str]] = [
    (r"\bwinners?\b", "winner"),
    (r"\bwon\b(?!')", "won"),
    (r"\bwinning\b", "winning"),
    (r"\b(first|second|third) (place|prize|position|runner[- ]up)\b", "placement"),
    (r"\b(1st|2nd|3rd|4th|5th)\b", "ordinal placement"),
    (r"\bchampions?(hip)?\b", "champion"),
    (r"\branked\b|\brank(ing)? \d+|\b(air|all india rank)\b", "ranking"),
    (r"\btop \d+%?", "top-N claim"),
    (r"\bawards?\b|\bawarded\b", "award"),
    (r"\bmedal(ist)?s?\b", "medal"),
    (r"\bfinalists?\b", "finalist"),
    (r"\b(gold|silver|bronze)\b", "gold/silver/bronze"),
    (r"\bhonou?rs?\b", "honours"),
    (r"\bscholarships?\b|\bscholars?\b", "scholarship"),
    (r"\btopper\b", "topper"),
    (r"\bpodium\b", "podium"),
    (r"\brunner[- ]up\b", "runner-up"),
    (r"\bsuperday\b", "superday"),
]

YEAR_OF_STUDY = re.compile(
    r"\b(pre[- ]?final|final|first|second|third|fourth|1st|2nd|3rd|4th)[- ]year\b|\bsophomore\b|\bfreshman\b|"
    r"\bclass of 20\d\d\b|\bgraduat\w*\s+(?:in\s+)?20\d\d\b|\bbatch of 20\d\d\b|\bgraduating\b",
    re.IGNORECASE,
)
GPA = re.compile(r"\b(c?gpa|cpi|sgpa)\b", re.IGNORECASE)

TECH_LEXICON = [
    "python", "java", "javascript", "typescript", "golang", "go", "rust", "c++", "c#", "ruby", "php", "scala",
    "kotlin", "swift", "r", "sql", "nosql", "html", "css", "react", "angular", "vue", "next.js", "node.js", "nodejs",
    "express", "django", "flask", "fastapi", "spring", "spring boot", "rails", ".net", "graphql", "rest", "grpc",
    "kafka", "rabbitmq", "redis", "postgresql", "postgres", "mysql", "mongodb", "cassandra", "dynamodb",
    "elasticsearch", "neo4j", "snowflake", "bigquery", "spark", "hadoop", "airflow", "dbt", "tableau", "power bi",
    "looker", "excel", "docker", "kubernetes", "terraform", "ansible", "jenkins", "github actions", "aws", "gcp",
    "azure", "linux", "prometheus", "grafana", "datadog", "splunk", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "langchain", "llamaindex", "openai", "hugging face", "transformers",
    "opencv", "selenium", "cypress", "playwright", "pytest", "junit", "jmeter", "postman", "figma", "jira",
    "solidity", "blockchain", "flutter", "react native", "android", "ios", "streamlit", "ollama", "gemini",
    "pyside6", "qt", "matlab", "simulink", "arduino", "ros",
]

_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?%?(?![\w])")
_ACHIEVEMENT_NUMBER = re.compile(r"(?<![\w.])(\d+\.\d+|\d+(?:\.\d+)?%|\d+(?:st|nd|rd|th))(?![\w])", re.IGNORECASE)


def _normalize(text: str) -> str:
    text = text or ""
    for ch in ("‐", "‑", "‒", "–", "—", "−"):
        text = text.replace(ch, "-")
    text = text.replace(" ", " ")
    return re.sub(r"\s+", " ", text).lower()


def _sentence_of(text: str, start: int) -> str:
    boundaries = [m.end() for m in re.finditer(r"[.!?](?:\s|$)|\n", text)]
    left = max([b for b in boundaries if b <= start], default=0)
    right = min([b for b in boundaries if b > start], default=len(text))
    return text[left:right].strip()[:140]


def find_unsupported_claims(text: str, source: str, numbers: str = "achievements", check_tech: bool = True) -> list[str]:
    """
    numbers: "all" (every number must be in the source; for resumes), "achievements" (only decimals, percentages
    and ordinals; for emails that may quote company figures), or "none".
    """
    issues: list[str] = []
    body = _normalize(text)
    src = _normalize(source)

    for pattern, label in AWARD_PATTERNS:
        match = re.search(pattern, body, re.IGNORECASE)
        if match and not re.search(pattern, src, re.IGNORECASE):
            issues.append(f"unsupported {label} claim: \"{_sentence_of(body, match.start())}\"")

    for match in YEAR_OF_STUDY.finditer(body):
        phrase = match.group(0)
        if phrase not in src:
            issues.append(f"unsupported year-of-study/graduation claim '{phrase}'")
    if GPA.search(body) and not GPA.search(src):
        issues.append("mentions CGPA/GPA, which is not in the resume")

    if numbers != "none":
        regex = _NUMBER if numbers == "all" else _ACHIEVEMENT_NUMBER
        src_numbers = {m.group(0).rstrip("%").replace(",", "") for m in _NUMBER.finditer(src)}
        src_numbers |= {m.group(0) for m in _ACHIEVEMENT_NUMBER.finditer(src)}
        for match in regex.finditer(body):
            token = match.group(0)
            plain = token.rstrip("%").replace(",", "")
            if token not in src_numbers and plain not in src_numbers:
                issues.append(f"number '{token}' not found in the source material")

    if check_tech:
        for tech in TECH_LEXICON:
            if len(tech) <= 2 and tech not in ("go", "r", "qt"):
                continue
            if tech in ("go", "r", "rest", "ios", "express"):
                continue  # too ambiguous as plain English words
            if keyword_in(body, tech) and not keyword_in(src, tech):
                issues.append(f"skill/technology '{tech}' is not in the resume")

    return list(dict.fromkeys(issues))
