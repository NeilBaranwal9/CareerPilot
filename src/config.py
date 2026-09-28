import os

import yaml
from pydantic import BaseModel, Field


class SalaryRange(BaseModel):
    min_lpa: float
    max_lpa: float


class CompanySize(BaseModel):
    min_employees: int
    max_employees: int


class JobPreferences(BaseModel):
    roles: list[str]
    geographies: list[str]
    remote_only: bool
    salary_range: SalaryRange
    company_size: CompanySize
    experience_years_max: float
    allow_speculative_outreach: bool = False


class Exclusions(BaseModel):
    companies: list[str]
    keywords: list[str]
    emails: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)


class PipelineSettings(BaseModel):
    daily_draft_limit: int
    research_depth: str
    cache_lifetime_seconds: int
    retry_limits: int
    generate_resume: bool = False
    # attach_base = always attach base_resume_path unchanged; variants = attach the rule-selected variant unchanged;
    # tailor = LLM-tailored resume per role. Empty = derived from generate_resume (tailor if true, else variants).
    resume_mode: str = ""
    ai_resume_path: str = "resumes/resume_vineet_kushwaha_ai.typ"
    dev_resume_path: str = "resumes/resume_vineet_kushwaha_dev.typ"
    base_resume_path: str = "resumes/resume_vineet_kushwaha.typ"
    generated_resumes_dir: str
    db_path: str
    automation: bool = True

    def effective_resume_mode(self) -> str:
        mode = (self.resume_mode or "").strip().lower()
        if mode in ("attach_base", "variants", "tailor"):
            return mode
        return "tailor" if self.generate_resume else "variants"


# Environment variables consulted when an API key is left blank in config.yaml
_PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: str = ""
    api_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 1000
    # Cheaper/faster model used for extraction & classification calls (e.g. "llama-3.1-8b-instant" on Groq).
    fast_model: str = ""
    # Models tried in order if the primary model is rate limited, decommissioned or unavailable.
    fallback_models: list[str] = Field(default_factory=list)
    max_retries: int = 4
    timeout_seconds: float = 60.0

    def resolved_api_key(self) -> str:
        """Returns the configured API key, falling back to the provider's standard environment variable."""
        if self.api_key:
            return self.api_key
        env_name = _PROVIDER_ENV_KEYS.get(self.provider.lower())
        return os.environ.get(env_name, "") if env_name else ""


class GmailConfig(BaseModel):
    credentials_file: str
    token_file: str
    scopes: list[str]


class ScoringWeights(BaseModel):
    role_match: float
    tech_stack: float
    salary: float
    company_quality: float
    growth: float
    confidence: float


class ScoringThresholds(BaseModel):
    minimum_score: float


class ScoringConfig(BaseModel):
    weights: ScoringWeights
    thresholds: ScoringThresholds
    # "hybrid" blends deterministic rules with the LLM judgement, "llm" and "rules" use one side only.
    mode: str = "hybrid"
    llm_weight: float = 0.5


class PromptTemplates(BaseModel):
    company_research: str
    resume_tailoring: str
    email_generation: str
    followup_generation: str = ""
    # Used when the base resume is a PDF/DOCX/text file (the LLM returns a structured resume rendered to PDF).
    resume_tailoring_structured: str = ""


class UserIdentity(BaseModel):
    name: str = "Your Name"
    email: str = ""
    linkedin_url: str = ""
    github_url: str = ""
    website_url: str = ""


class TargetProfile(BaseModel):
    """What a good-fit company looks like for you. Drives company fit scoring and rejection."""

    # Relative preference per sector (1.0 = ideal). Sectors not listed fall back to "generic".
    sector_weights: dict[str, float] = Field(default_factory=lambda: {"generic": 0.5})
    # If non-empty, companies outside these sectors are rejected outright.
    allowed_sectors: list[str] = Field(default_factory=list)
    # If non-empty, preferred funding stages (pre_seed, seed, series_a, series_b, series_c, series_d_plus, public, bootstrapped).
    funding_stages: list[str] = Field(default_factory=list)
    # Your core skills; used to measure tech-stack overlap.
    skills: list[str] = Field(default_factory=list)
    min_company_fit: float = 0.45
    reject_below_fit: bool = True


class DiscoveryConfig(BaseModel):
    # Standing natural-language queries re-run by the daily scheduler, e.g. "Fintech companies in India".
    queries: list[str] = Field(default_factory=list)
    # Company sources: llm, web_search, yc, linkedin_jobs, wellfound, ats_boards
    sources: list[str] = Field(default_factory=lambda: ["llm", "web_search", "yc", "linkedin_jobs", "wellfound"])
    # Job sources: ats, linkedin, career_page, wellfound, indeed, web_search
    job_sources: list[str] = Field(
        default_factory=lambda: ["ats", "linkedin", "career_page", "wellfound", "indeed", "web_search"]
    )
    companies_per_run: int = 20
    max_jobs_per_company: int = 3
    # Known ATS board tokens, e.g. {"greenhouse": ["razorpay"], "lever": ["cred"]}
    ats_boards: dict[str, list[str]] = Field(default_factory=dict)


class ContactDiscoveryConfig(BaseModel):
    # linkedin_search, team_page, press, github, hunter, apollo
    sources: list[str] = Field(
        default_factory=lambda: ["linkedin_search", "team_page", "press", "github", "hunter", "apollo"]
    )
    # Preferred personas, most preferred first.
    personas: list[str] = Field(
        default_factory=lambda: ["engineering_manager", "hiring_manager", "recruiter", "founder", "cto", "tech_lead"]
    )
    # "auto" adapts to company size (founders at small startups, EMs/recruiters at larger ones); "ordered" follows personas strictly.
    persona_strategy: str = "auto"
    max_contacts_per_company: int = 1
    allow_generic_inbox: bool = True


class EmailVerificationConfig(BaseModel):
    smtp_check: bool = True
    smtp_timeout: float = 8.0
    smtp_from: str = ""
    helo_host: str = ""
    allow_catch_all: bool = True
    # Accept the best evidence-backed guess when SMTP verification is unavailable (e.g. ISP blocks port 25).
    allow_unverified: bool = True
    min_confidence: float = 0.3
    max_smtp_probes: int = 6


class ApiKeys(BaseModel):
    """Optional third-party API keys. Blank values fall back to environment variables."""

    hunter: str = ""
    apollo: str = ""
    github: str = ""
    serper: str = ""
    brave: str = ""

    def get(self, name: str) -> str:
        value = getattr(self, name, "") or ""
        if value:
            return str(value)
        return os.environ.get(f"{name.upper()}_API_KEY", "") or os.environ.get(f"{name.upper()}_TOKEN", "")


class SearchConfig(BaseModel):
    # auto (serper > brave > duckduckgo), duckduckgo, serper, brave
    provider: str = "auto"
    min_interval_seconds: float = 2.0


class VariantRule(BaseModel):
    """
    A rule matches when EVERY condition that is set matches (each list is "any of").
    Example: {sectors: [fintech], title_any: [backend, sde]} = fintech company AND backend/SDE title.
    """

    sectors: list[str] = Field(default_factory=list)  # company primary sector (e.g. fintech)
    title_any: list[str] = Field(default_factory=list)  # whole-word match in the job title
    description_any: list[str] = Field(default_factory=list)  # whole-word match in the job description
    stack_any: list[str] = Field(default_factory=list)  # company tech stack items
    personas: list[str] = Field(default_factory=list)  # contact persona, e.g. founder, recruiter
    title_none: list[str] = Field(default_factory=list)  # exclude when any of these appear in the title


class ResumeVariant(BaseModel):
    name: str
    path: str  # .pdf, .typ, .docx, .md or .txt; several variants may share one file (see `focus`)
    tags: list[str] = Field(default_factory=list)  # used only for score-based selection (no rules anywhere)
    sectors: list[str] = Field(default_factory=list)
    rules: list[VariantRule] = Field(default_factory=list)
    priority: int = 0  # higher wins when several variants match their rules
    default: bool = False  # used when no rule matches
    focus: str = ""  # tailoring instructions, e.g. "Lead with payments/fraud work; emphasise Python & APIs"


class Highlight(BaseModel):
    name: str
    summary: str = ""
    tags: list[str] = Field(default_factory=list)


class FollowUpStep(BaseModel):
    after_days: int


class SendWindow(BaseModel):
    timezone: str = "Asia/Kolkata"
    start_hour: int = 9
    end_hour: int = 12
    weekdays_only: bool = True


class OutreachConfig(BaseModel):
    # formal | warm | concise | enthusiastic | casual
    tone: str = "warm"
    # Per-persona tone overrides, e.g. {"founder": "concise"}
    persona_tones: dict[str, str] = Field(default_factory=dict)
    auto_send: bool = False
    daily_send_limit: int = 20
    min_minutes_between_sends: int = 4
    send_window: SendWindow = Field(default_factory=SendWindow)
    followups: list[FollowUpStep] = Field(
        default_factory=lambda: [FollowUpStep(after_days=4), FollowUpStep(after_days=7)]
    )
    stop_on_reply: bool = True
    company_cooldown_days: int = 60
    check_gmail_history: bool = True
    classify_replies: bool = True
    retry_on_bounce: bool = True
    # Days after the last follow-up with no reply before an application is marked "no_response".
    no_response_after_days: int = 10


class LearningConfig(BaseModel):
    enabled: bool = True
    prior_reply_rate: float = 0.08
    prior_strength: float = 10.0
    min_samples: int = 5


class AppConfig(BaseModel):
    job_preferences: JobPreferences
    exclusions: Exclusions
    pipeline: PipelineSettings
    llm: LLMConfig
    gmail: GmailConfig
    scoring: ScoringConfig
    prompts: PromptTemplates
    user_identity: UserIdentity = UserIdentity()
    target_profile: TargetProfile = Field(default_factory=TargetProfile)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    contacts: ContactDiscoveryConfig = Field(default_factory=ContactDiscoveryConfig)
    email_verification: EmailVerificationConfig = Field(default_factory=EmailVerificationConfig)
    api_keys: ApiKeys = Field(default_factory=ApiKeys)
    search: SearchConfig = Field(default_factory=SearchConfig)
    resumes: list[ResumeVariant] = Field(default_factory=list)
    highlights: list[Highlight] = Field(default_factory=list)
    outreach: OutreachConfig = Field(default_factory=OutreachConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Loads configuration from a YAML file and validates it using Pydantic."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return AppConfig(**config_data)
