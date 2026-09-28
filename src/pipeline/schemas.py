"""Pydantic schemas for structured LLM responses used across the pipeline."""

from pydantic import BaseModel, Field


class DiscoveredCompany(BaseModel):
    name: str = Field(description="Name of the company")
    domain: str | None = Field(description="Primary domain of the company, e.g. company.com")
    employee_count: int | None = Field(description="Estimated number of employees")
    industry: str | None = Field(description="Industry vertical, e.g. SaaS, Fintech")
    sector: str | None = Field(default=None, description="Canonical sector, e.g. fintech, healthtech, edtech, ai")
    funding_stage: str | None = Field(default=None, description="Latest funding stage, e.g. seed, series_a, series_b")
    description: str | None = Field(default=None, description="One-line description of what the company does")
    location: str | None = Field(default=None, description="Headquarters city and country")


class CompanyListResponse(BaseModel):
    companies: list[DiscoveredCompany]


class DiscoveredJob(BaseModel):
    title: str = Field(description="Title of the job role")
    url: str | None = Field(description="URL to the job listing or company careers page")
    location: str | None = Field(description="Job location details")
    salary: str | None = Field(description="Salary range or package details")
    experience_years: float | None = Field(description="Required experience years, if mentioned")
    description: str | None = Field(description="Brief summary of requirements or job description")


class JobListResponse(BaseModel):
    jobs: list[DiscoveredJob]


class NewsItem(BaseModel):
    title: str = Field(description="Headline of the news item, launch or announcement")
    date: str | None = Field(default=None, description="Date or month/year if known")
    summary: str | None = Field(default=None, description="One-sentence summary")
    url: str | None = Field(default=None, description="Source URL if present in the context")


class CompanyResearchResponse(BaseModel):
    business_model: str = Field(description="Business model and core product description")
    funding: str = Field(description="Details of latest funding or launches")
    tech_stack: list[str] = Field(description="List of core languages, frameworks, or databases used")
    culture: str = Field(description="Engineering culture or notable developer initiatives")
    leadership: list[str] = Field(description="Founders, CEO, CTO, or VP Engineering details")
    description: str | None = Field(default=None, description="Two-sentence description of the company and product")
    sector: str | None = Field(
        default=None,
        description="Primary sector: fintech, trading, crypto, insurtech, healthtech, edtech, ai, devtools, "
        "cybersecurity, saas, ecommerce, logistics, mobility, climate, hrtech, proptech, agritech, gaming, media, "
        "foodtech, travel, legaltech, martech, data, robotics or generic",
    )
    sub_sectors: list[str] = Field(default_factory=list, description="Other sectors from the same list that apply")
    funding_stage: str | None = Field(
        default=None,
        description="Latest stage: pre_seed, seed, series_a, series_b, series_c, series_d_plus, public, bootstrapped, acquired",
    )
    headcount: int | None = Field(default=None, description="Approximate number of employees")
    hiring_signals: str | None = Field(default=None, description="Evidence the company is (or is not) hiring engineers")
    recent_news: list[NewsItem] = Field(
        default_factory=list, description="Recent launches, funding rounds or announcements found in the context"
    )
    products: list[str] = Field(default_factory=list, description="Named products or features")
    location: str | None = Field(default=None, description="Headquarters city and country")


class DiscoveredContact(BaseModel):
    name: str = Field(description="Full name of the contact person")
    role: str = Field(description="Role or title of the contact")
    email_pattern: str | None = Field(description="Observed email format, e.g., first.last@company.com")
    linkedin_url: str | None = Field(description="LinkedIn profile link if available")
    background: str | None = Field(
        default=None, description="One-line professional background (prior companies, talks, projects) if present"
    )


class ContactListResponse(BaseModel):
    contacts: list[DiscoveredContact]


class EmailDiscoveryResponse(BaseModel):
    email: str | None = Field(description="Discovered professional email address")
    pattern_used: str | None = Field(description="The pattern or source used to find/verify this email")


class OpportunityScoreResponse(BaseModel):
    role_match: float = Field(description="Score between 0.0 and 1.0 representing how well the role fits")
    tech_stack: float = Field(description="Score between 0.0 and 1.0 representing tech alignment")
    salary: float = Field(description="Score between 0.0 and 1.0 representing salary alignment")
    company_quality: float = Field(description="Score between 0.0 and 1.0 representing company status/domain")
    growth: float = Field(description="Score between 0.0 and 1.0 representing growth potential")
    confidence: float = Field(description="Score between 0.0 and 1.0 representing confidence in data quality")
    reasoning: str = Field(description="Brief summary of scoring rationale")


class ResumeTailorResponse(BaseModel):
    tailored_typst_content: str = Field(description="The complete modified Typst code for the resume")
    keywords_added: list[str] = Field(description="List of ATS keywords or skills added")
    reasoning: str = Field(description="Explanation of modifications and section ordering decisions")


class ResumeEntry(BaseModel):
    title: str = Field(description="Role, project name, degree or award title, exactly as in the original resume")
    organization: str | None = Field(default=None, description="Employer, institution or event, exactly as in the original")
    location: str | None = Field(default=None, description="Location if present in the original")
    dates: str | None = Field(default=None, description="Dates exactly as in the original, e.g. 'May 2026 - Jun 2026'")
    subtitle: str | None = Field(default=None, description="Optional line such as the tech stack or GPA, from the original")
    bullets: list[str] = Field(default_factory=list, description="Bullet points (may be reworded, never invented)")


class ResumeSection(BaseModel):
    heading: str = Field(description="Section heading, e.g. Experience, Projects, Education, Skills, Achievements")
    entries: list[ResumeEntry] = Field(default_factory=list, description="Entries for Experience/Projects/Education")
    items: list[str] = Field(
        default_factory=list, description="Plain lines for list sections such as Skills or Achievements"
    )


class StructuredResumeSchema(BaseModel):
    name: str = Field(description="Candidate's full name, exactly as in the original")
    contact_line: str = Field(description="Phone | email | LinkedIn | GitHub line, exactly as in the original")
    sections: list[ResumeSection] = Field(description="Resume sections in the tailored order")
    keywords_added: list[str] = Field(default_factory=list, description="Skills/keywords emphasised for this role")
    reasoning: str = Field(default="", description="What was reordered/emphasised and why")


class EmailGenResponse(BaseModel):
    subject: str = Field(description="Subject line for the outreach email")
    body_html: str = Field(description="HTML formatted email body")


class FollowUpDraft(BaseModel):
    body_html: str = Field(description="HTML body of the follow-up (it is sent as a reply in the same thread)")


class FollowUpSequenceSchema(BaseModel):
    followups: list[FollowUpDraft] = Field(description="Follow-up emails in sending order")


class ValidationResponse(BaseModel):
    is_valid: bool = Field(description="True if the resume and email contain no placeholders or hallucinated facts")
    errors: list[str] = Field(description="List of validation errors found")


class ReplyClassificationSchema(BaseModel):
    category: str = Field(
        description="One of: interview, positive, referral, neutral, not_interested, out_of_office, bounce, other"
    )
    summary: str = Field(description="One-sentence summary of the reply")
    interview_requested: bool = Field(default=False, description="True if they propose a call/interview/next step")


class CampaignSpecSchema(BaseModel):
    sectors: list[str] = Field(default_factory=list, description="Canonical sectors, e.g. fintech, ai, trading")
    geographies: list[str] = Field(default_factory=list, description="Countries/cities, e.g. India, Bengaluru, Remote")
    funding_stages: list[str] = Field(
        default_factory=list, description="Stages like seed, series_a, series_b (empty if not specified)"
    )
    min_employees: int | None = Field(default=None, description="Minimum employees if specified")
    max_employees: int | None = Field(default=None, description="Maximum employees if specified")
    company_count: int | None = Field(default=None, description="How many companies the user wants")
    personas: list[str] = Field(
        default_factory=list,
        description="Contacts to reach: engineering_manager, recruiter, founder, cto, hiring_manager, tech_lead",
    )
    keywords: list[str] = Field(default_factory=list, description="Other descriptive keywords, e.g. 'B2B', 'payments'")
    role_titles: list[str] = Field(default_factory=list, description="Job titles to target if mentioned")
