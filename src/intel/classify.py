"""Deterministic classifiers: sector, funding stage, contact role, headcount and salary parsing."""

import re
from functools import lru_cache

# Canonical sector keys -> indicative keywords. Multi-word keywords count double.
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "fintech": [
        "fintech", "payments", "payment gateway", "payment", "lending", "loans", "credit", "neobank", "banking",
        "wealth", "investing", "investment", "brokerage", "upi", "cards", "wallet", "remittance", "accounting",
        "invoicing", "treasury", "financial services", "nbfc", "bnpl", "buy now pay later", "mutual fund",
        "stock broking", "personal finance", "expense management", "payroll", "financial infrastructure",
    ],
    "trading": [
        "trading", "hft", "high-frequency", "high frequency", "quant", "quantitative", "market making",
        "market maker", "algorithmic trading", "algo trading", "proprietary trading", "prop trading", "derivatives",
        "options trading", "hedge fund",
    ],
    "crypto": ["crypto", "cryptocurrency", "blockchain", "web3", "defi", "bitcoin", "ethereum", "nft", "stablecoin"],
    "insurtech": ["insurtech", "insurance", "claims", "underwriting"],
    "healthtech": [
        "healthtech", "health tech", "healthcare", "health", "medical", "clinic", "hospital", "patient", "pharma",
        "diagnostics", "telemedicine", "wellness", "medtech", "biotech", "doctor", "mental health", "fitness",
    ],
    "edtech": [
        "edtech", "education", "learning platform", "e-learning", "students", "courses", "upskilling", "tutoring",
        "school", "university", "exam prep", "lms", "learning",
    ],
    "ai": [
        "artificial intelligence", "ai", "machine learning", "ml", "llm", "llms", "generative ai", "genai",
        "deep learning", "computer vision", "nlp", "ai-powered", "ai agents", "ai-native", "foundation model",
        "conversational ai", "speech recognition", "mlops",
    ],
    "devtools": [
        "developer tools", "devtools", "developer platform", "api platform", "sdk", "observability", "ci/cd",
        "devops", "infrastructure software", "open source", "database", "cloud infrastructure", "developer",
    ],
    "cybersecurity": ["cybersecurity", "security", "identity", "fraud detection", "threat", "zero trust", "siem"],
    "saas": ["saas", "b2b software", "enterprise software", "crm", "workflow", "productivity", "collaboration"],
    "ecommerce": ["ecommerce", "e-commerce", "d2c", "marketplace", "retail", "shopping", "quick commerce", "q-commerce"],
    "logistics": ["logistics", "supply chain", "shipping", "delivery", "fleet", "freight", "warehouse"],
    "mobility": ["mobility", "electric vehicle", "ev", "automotive", "ride-hailing", "autonomous driving"],
    "climate": ["climate", "clean energy", "renewable", "solar", "carbon", "sustainability", "energy"],
    "hrtech": ["hrtech", "hr tech", "recruiting", "hiring platform", "talent", "workforce", "hris"],
    "proptech": ["proptech", "real estate", "property", "rental", "housing"],
    "agritech": ["agritech", "agtech", "agriculture", "farmers", "farming"],
    "gaming": ["gaming", "games", "esports", "game studio"],
    "media": ["media", "content", "streaming", "social media", "creator", "news"],
    "foodtech": ["foodtech", "food delivery", "restaurant", "cloud kitchen"],
    "travel": ["travel", "hospitality", "booking", "hotels", "flights"],
    "legaltech": ["legaltech", "legal", "compliance", "contracts"],
    "martech": ["martech", "marketing", "advertising", "adtech", "customer engagement"],
    "data": ["data platform", "analytics", "data infrastructure", "data engineering", "business intelligence"],
    "robotics": ["robotics", "robots", "drones", "hardware", "iot", "semiconductor"],
}

# Tie-break priority when two sectors score equally (more specific first).
SECTOR_PRIORITY = [
    "trading", "crypto", "insurtech", "fintech", "healthtech", "edtech", "cybersecurity", "devtools", "ai",
    "hrtech", "proptech", "agritech", "logistics", "mobility", "climate", "gaming", "foodtech", "travel",
    "legaltech", "martech", "ecommerce", "media", "data", "robotics", "saas",
]

SECTOR_ALIASES: dict[str, str] = {
    "financial technology": "fintech",
    "finance": "fintech",
    "financial services": "fintech",
    "banking": "fintech",
    "payments": "fintech",
    "artificial intelligence": "ai",
    "machine learning": "ai",
    "generative ai": "ai",
    "health": "healthtech",
    "healthcare": "healthtech",
    "education": "edtech",
    "e-commerce": "ecommerce",
    "security": "cybersecurity",
    "software": "saas",
    "b2b saas": "saas",
    "enterprise software": "saas",
    "developer tools": "devtools",
    "quant": "trading",
    "quantitative trading": "trading",
    "capital markets": "trading",
    "web3": "crypto",
    "blockchain": "crypto",
}


@lru_cache(maxsize=512)
def _keyword_regex(keyword: str) -> re.Pattern[str]:
    return re.compile(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])")


def keyword_in(text: str, keyword: str) -> bool:
    """Whole-word, case-insensitive keyword match (so 'ai' does not match 'email')."""
    return bool(_keyword_regex(keyword.lower()).search((text or "").lower()))


def sector_scores(text: str) -> dict[str, float]:
    lowered = (text or "").lower()
    scores: dict[str, float] = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            hits = len(_keyword_regex(kw).findall(lowered))
            if hits:
                score += min(hits, 3) * (2.0 if " " in kw or "-" in kw else 1.0)
        if score:
            scores[sector] = score
    return scores


def classify_sector(text: str) -> tuple[str, list[str]]:
    """Returns (primary_sector, all_matching_sectors_ranked). 'generic' when nothing matches."""
    scores = sector_scores(text)
    if not scores:
        return "generic", []
    ranked = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], SECTOR_PRIORITY.index(kv[0]) if kv[0] in SECTOR_PRIORITY else 99),
    )
    return ranked[0][0], [s for s, _ in ranked]


def normalize_sector(label: str | None) -> str | None:
    """Maps free-text industry labels ('FinTech', 'Financial Services', 'AI/ML') to canonical sector keys."""
    if not label:
        return None
    lowered = label.strip().lower()
    if lowered in SECTOR_KEYWORDS:
        return lowered
    if lowered in SECTOR_ALIASES:
        return SECTOR_ALIASES[lowered]
    primary, _ = classify_sector(lowered)
    return None if primary == "generic" else primary


FUNDING_ORDER = ["pre_seed", "seed", "series_a", "series_b", "series_c", "series_d_plus", "public", "acquired"]

_FUNDING_PATTERNS: list[tuple[str, str]] = [
    (r"pre[- ]?seed", "pre_seed"),
    (r"(?<!pre[- ])(?<!pre)\bseed\b", "seed"),
    (r"series[- ]a\b", "series_a"),
    (r"series[- ]b\b", "series_b"),
    (r"series[- ]c\b", "series_c"),
    (r"series[- ][d-j]\b", "series_d_plus"),
    (r"\bipo\b|publicly traded|listed on (?:the )?(?:nse|bse|nasdaq|nyse)|public company", "public"),
    (r"acquired by", "acquired"),
]


def normalize_funding_stage(text: str | None) -> str | None:
    """Extracts the latest funding stage mentioned in text ('raised Seed in 2021 and Series B in 2024' -> series_b)."""
    if not text:
        return None
    lowered = text.lower()
    if lowered.strip() in FUNDING_ORDER or lowered.strip() == "bootstrapped":
        return lowered.strip()
    found = [stage for pattern, stage in _FUNDING_PATTERNS if re.search(pattern, lowered)]
    if found:
        return max(found, key=FUNDING_ORDER.index)
    if re.search(r"bootstrapped|self[- ]funded|profitable without", lowered):
        return "bootstrapped"
    return None


# ---------------------------------------------------------------------------
# Contact roles
# ---------------------------------------------------------------------------

_ROLE_RULES: list[tuple[str, list[str]]] = [
    ("recruiter", [
        "recruiter", "recruiting", "talent acquisition", "talent partner", "technical recruiter", "sourcer",
        "people partner", "hr business partner", "human resources", "hr manager", "hr executive", "head of people",
        "people operations", "talent lead", "hiring partner", "campus hiring", "university recruiting", "hr",
    ]),
    ("founder", ["founder", "co-founder", "cofounder"]),
    ("cto", ["cto", "chief technology officer", "chief technical officer"]),
    ("vp_engineering", [
        "vp engineering", "vp of engineering", "vice president engineering", "vice president of engineering",
        "head of engineering", "director of engineering", "engineering director", "head of technology",
        "svp engineering", "head of tech",
    ]),
    ("hiring_manager", ["hiring manager"]),
    ("engineering_manager", [
        "engineering manager", "software engineering manager", "manager, engineering", "manager - engineering",
        "development manager", "sde manager", "manager of engineering", "em ", "tech manager", "technical manager",
    ]),
    ("tech_lead", [
        "tech lead", "technical lead", "lead engineer", "engineering lead", "staff engineer", "principal engineer",
        "architect", "team lead",
    ]),
    ("executive", ["ceo", "chief executive", "coo", "chief operating", "president", "managing director", "cpo"]),
    ("engineer", ["software engineer", "developer", "sde", "engineer", "programmer", "data scientist"]),
    ("generic_inbox", ["generic inbox", "hiring team", "careers inbox", "recruiting (generic"]),
]


def classify_role(title: str | None) -> tuple[str, str]:
    """Maps a job title to (role_category, seniority)."""
    lowered = f" {(title or '').lower()} "
    category = "other"
    for cat, keywords in _ROLE_RULES:
        if any(keyword_in(lowered, kw.strip()) for kw in keywords):
            category = cat
            break

    if category in ("founder", "cto", "executive", "vp_engineering"):
        seniority = "executive"
    elif category in ("engineering_manager", "hiring_manager") or keyword_in(lowered, "manager"):
        seniority = "manager"
    elif keyword_in(lowered, "senior") or keyword_in(lowered, "lead") or keyword_in(lowered, "staff"):
        seniority = "senior"
    else:
        seniority = "individual"
    return category, seniority


# ---------------------------------------------------------------------------
# Headcount & salary parsing
# ---------------------------------------------------------------------------


def parse_headcount(value: str | int | None) -> int | None:
    """'51-200 employees' -> 125, '1,200+' -> 1200, 350 -> 350."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    numbers = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", str(value))]
    if not numbers:
        return None
    if len(numbers) >= 2:
        return (numbers[0] + numbers[1]) // 2
    return numbers[0]


_STAGE_HEADCOUNT = {
    "pre_seed": 8, "seed": 20, "series_a": 60, "series_b": 200, "series_c": 500,
    "series_d_plus": 1500, "public": 3000, "acquired": 1000,
}


def effective_headcount(employee_count: int | None, funding_stage: str | None) -> int | None:
    """Known headcount, or a rough estimate from the funding stage when headcount is unknown."""
    if employee_count is not None:
        return employee_count
    return _STAGE_HEADCOUNT.get(funding_stage or "")


def size_bucket(employee_count: int | None) -> str:
    if employee_count is None:
        return "unknown"
    if employee_count <= 50:
        return "1-50"
    if employee_count <= 200:
        return "51-200"
    if employee_count <= 1000:
        return "201-1000"
    return "1000+"


def parse_salary_lpa(text: str | None) -> tuple[float | None, float | None]:
    """
    Parses Indian (LPA / lakh / ₹) and USD salary strings into (min_lpa, max_lpa).
    '10-20 LPA' -> (10, 20); '₹12,00,000' -> (12, 12); '$120k - $150k' -> (~102, ~128).
    """
    if not text:
        return None, None
    lowered = text.lower().replace(",", "")
    usd = "$" in lowered or "usd" in lowered
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", lowered)]
    if not numbers:
        return None, None
    values: list[float] = []
    for n in numbers[:2]:
        if usd:
            amount = n * 1000 if ("k" in lowered and n < 10000) else n
            values.append(round(amount * 85 / 100000, 1))  # USD -> INR lakhs
        elif n >= 100000:
            values.append(round(n / 100000, 1))  # raw rupees
        elif re.search(r"\bcr\b|crore", lowered) and n < 100:
            values.append(n * 100)
        else:
            values.append(n)
    return min(values), max(values)
