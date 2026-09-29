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
        "ta", "ta manager", "ta lead", "ta partner", "talent", "campus recruiter",
        "people operations", "talent lead", "hiring partner", "campus hiring", "university recruiting", "hr",
    ]),
    ("founder", ["founder", "co-founder", "cofounder"]),
    ("cto", ["cto", "chief technology officer", "chief technical officer"]),
    ("product_lead", [
        "head of product", "vp product", "vp of product", "vice president of product", "director of product",
        "product director", "chief product officer", "cpo", "product lead", "group product manager",
        "principal product manager", "lead product manager",
    ]),
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
    ("product_manager", [
        "product manager", "associate product manager", "apm", "product owner", "technical product manager",
        "product management",
    ]),
    ("data_lead", [
        "head of data", "data lead", "analytics lead", "head of analytics", "data science manager",
        "analytics manager", "director of data", "lead data scientist", "director of analytics",
    ]),
    ("qa_lead", [
        "qa lead", "qa manager", "test lead", "head of quality", "quality assurance manager", "sdet lead",
        "test manager", "quality engineering manager", "head of qa",
    ]),
    ("tech_lead", [
        "tech lead", "technical lead", "lead engineer", "engineering lead", "staff engineer", "staff software engineer",
        "principal engineer", "architect", "team lead",
    ]),
    ("executive", [
        "ceo", "chief executive", "coo", "chief operating", "president", "managing director", "cfo",
        "chief financial officer", "chief business officer", "cbo", "cro", "chief revenue officer", "cmo",
        "chief marketing officer", "chief", "vice president", "vp",
    ]),
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
# Target role families (job-search queries, title matching, company-level outreach)
# ---------------------------------------------------------------------------

# Checked in order: "QA Engineer" is qa (not software), "AI Engineer" is ai_ml, "Business Intelligence" is data.
_ROLE_FAMILY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("qa", re.compile(r"\b(qa|sdet|test|testing|tester|quality)\b")),
    ("ai_ml", re.compile(r"\b(ai|ml|machine learning|deep learning|nlp|llm|computer vision|genai)\b")),
    ("product", re.compile(r"\b(product|apm)\b")),
    ("data", re.compile(r"\b(data|analytics|business intelligence|bi)\b")),
    ("business", re.compile(r"\b(business|strategy|operations|consulting|consultant)\b")),
    ("software", re.compile(
        r"\b(software|sde|swe|backend|back-end|frontend|front-end|full[ -]?stack|developer|engineer|engineering|"
        r"programmer|devops|mobile|android|ios|python|java|cloud)\b"
    )),
]
ROLE_FAMILY_LABELS: dict[str, str] = {
    "product": "product",
    "data": "data/analytics",
    "business": "business analysis",
    "qa": "QA/testing",
    "software": "software engineering",
    "ai_ml": "AI/ML",
}
# Words that describe seniority/season rather than the role itself.
_ROLE_NOISE = re.compile(
    r"\b(intern|interns|internship|trainee|summer|winter|graduate|new grad|entry level|junior|jr|associate)\b|\(.*?\)"
)


def role_family(title: str | None) -> str | None:
    """'QA Engineer Intern' -> 'qa', 'Product Analyst Intern' -> 'product'; None for e.g. 'Summer Intern'."""
    lowered = (title or "").lower()
    for family, pattern in _ROLE_FAMILY_RULES:
        if pattern.search(lowered):
            return family
    return None


def role_families(roles: list[str]) -> list[str]:
    """Distinct families of the configured roles, in the order the roles are listed (first = most preferred)."""
    families: list[str] = []
    for role in roles:
        family = role_family(role)
        if family and family not in families:
            families.append(family)
    return families


def role_search_terms(roles: list[str], limit: int = 2) -> list[str]:
    """
    Short job-search terms taken from the configured roles: the first role of each of the first `limit` families,
    without intern/seniority words. ['Product Analyst Intern', 'APM Intern', 'Data Analyst Intern'] ->
    ['product analyst', 'data analyst']. Empty when no role has a recognisable family (never defaults to engineer).
    """
    terms: list[str] = []
    seen: set[str] = set()
    for role in roles:
        family = role_family(role)
        if not family or family in seen:
            continue
        term = " ".join(_ROLE_NOISE.sub(" ", role.lower()).split()).strip(" -,/")
        if term:
            seen.add(family)
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def describe_role_families(families: list[str], limit: int = 3) -> str:
    """['product', 'data', 'qa'] -> 'product, data/analytics or QA/testing'."""
    labels = [ROLE_FAMILY_LABELS.get(f, f) for f in families[:limit]]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " or " + labels[-1]


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
