from src.config import load_config
from src.db.models import Company
from src.sources.ats import (
    candidate_tokens,
    detect_ats_from_html,
    fetch_ats_jobs,
    filter_relevant_jobs,
    parse_experience_years,
    probe_ats,
)
from src.sources.companies import CompanyCandidate, merge_candidates, parse_spec_fallback
from src.sources.contacts import ContactCandidate, merge_contact_candidates, parse_linkedin_result, rank_contacts
from src.sources.job_boards import parse_linkedin_job_cards, search_indeed_jobs, search_wellfound_companies


class FakeJsonBrowser:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []

    def fetch_json(self, url, method="GET", headers=None, params=None, json_body=None, timeout=20.0):
        self.requests.append(url)
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                return payload
        raise RuntimeError(f"404 {url}")


GREENHOUSE = {
    "jobs": [
        {"title": "Software Engineer I", "absolute_url": "https://boards.greenhouse.io/payco/jobs/1",
         "location": {"name": "Bengaluru, India"}, "content": "&lt;p&gt;0-2 years of experience with Python&lt;/p&gt;"},
        {"title": "Senior Staff Engineer", "absolute_url": "https://boards.greenhouse.io/payco/jobs/2",
         "location": {"name": "Bengaluru, India"}, "content": ""},
        {"title": "Account Executive", "absolute_url": "https://boards.greenhouse.io/payco/jobs/3",
         "location": {"name": "Mumbai"}, "content": ""},
        {"title": "Backend Engineer", "absolute_url": "https://boards.greenhouse.io/payco/jobs/4",
         "location": {"name": "Berlin, Germany"}, "content": ""},
    ]
}
LEVER = [
    {"text": "Backend Developer", "hostedUrl": "https://jobs.lever.co/cred/abc", "categories": {"location": "Remote"},
     "descriptionPlain": "1+ years building APIs"}
]


def test_ats_detection_and_tokens():
    html = '<a href="https://boards.greenhouse.io/payco/jobs/123">Jobs</a>'
    assert detect_ats_from_html(html) == ("greenhouse", "payco")
    assert detect_ats_from_html('<iframe src="https://jobs.lever.co/cred"></iframe>') == ("lever", "cred")
    assert detect_ats_from_html('<script src="https://jobs.ashbyhq.com/sarvam/embed"></script>') == ("ashby", "sarvam")
    assert candidate_tokens("Razorpay Software Pvt Ltd", "razorpay.com")[0] == "razorpay"


def test_greenhouse_and_lever_parsing_and_filtering():
    browser = FakeJsonBrowser({
        "https://boards-api.greenhouse.io/v1/boards/payco": GREENHOUSE,
        "https://api.lever.co/v0/postings/cred": LEVER,
    })
    gh = fetch_ats_jobs(browser, "greenhouse", "payco")
    assert len(gh) == 4 and gh[0].description.startswith("0-2 years")
    relevant = filter_relevant_jobs(gh, ["Software Engineer", "Backend Engineer"], ["India", "Remote"], 1.0)
    assert [j.title for j in relevant] == ["Software Engineer I"]  # senior, sales and Berlin roles removed
    assert parse_experience_years(gh[0].description) == 0.0

    lever = fetch_ats_jobs(browser, "lever", "cred")
    assert lever[0].title == "Backend Developer" and lever[0].source == "lever"


def test_probe_ats_finds_board_by_token():
    browser = FakeJsonBrowser({"https://boards-api.greenhouse.io/v1/boards/payco": GREENHOUSE})
    found = probe_ats(browser, "PayCo", "payco.in")
    assert found is not None and found[0] == "greenhouse" and found[1] == "payco"


LINKEDIN_HTML = """
<ul>
 <li><div class="base-card">
   <a class="base-card__full-link" href="https://in.linkedin.com/jobs/view/software-engineer-at-payco-3999999999?refId=x">x</a>
   <h3 class="base-search-card__title">Software Engineer</h3>
   <h4 class="base-search-card__subtitle"><a href="https://in.linkedin.com/company/payco?trk=x">PayCo</a></h4>
   <span class="job-search-card__location">Bengaluru, Karnataka, India</span>
   <time datetime="2026-09-20">1 week ago</time>
 </div></li>
</ul>
"""


def test_linkedin_job_card_parsing():
    jobs = parse_linkedin_job_cards(LINKEDIN_HTML)
    assert jobs == [{
        "title": "Software Engineer",
        "company": "PayCo",
        "company_linkedin_url": "https://in.linkedin.com/company/payco",
        "location": "Bengaluru, Karnataka, India",
        "url": "https://in.linkedin.com/jobs/view/software-engineer-at-payco-3999999999",
        "posted_at": "2026-09-20",
        "source": "linkedin",
    }]


def test_linkedin_profile_result_parsing():
    result = {
        "title": "Priya Sharma - Engineering Manager - PayCo | LinkedIn",
        "url": "https://in.linkedin.com/in/priya-sharma-123?trk=x",
        "snippet": "Engineering Manager at PayCo. Previously Flipkart.",
    }
    cand = parse_linkedin_result(result, "PayCo")
    assert cand is not None and cand.name == "Priya Sharma" and cand.title == "Engineering Manager"
    assert cand.linkedin_url == "https://in.linkedin.com/in/priya-sharma-123"
    other_company = {**result, "title": "Priya Sharma - Engineering Manager - OtherCo | LinkedIn", "snippet": ""}
    assert parse_linkedin_result(other_company, "PayCo") is None


class FakeSearchBrowser:
    def __init__(self, results):
        self.results = results

    def search_google(self, query, num_results=5, include_blocked=False):
        return self.results


def test_wellfound_and_indeed_from_search_results():
    wf = FakeSearchBrowser([
        {"title": "PayCo Careers, Funding, and Management Team | Wellfound", "url": "https://wellfound.com/company/payco", "snippet": "Payments"},
        {"title": "Unrelated", "url": "https://example.org", "snippet": ""},
    ])
    assert search_wellfound_companies(wf, "fintech India")[0]["name"] == "PayCo"
    indeed = FakeSearchBrowser([
        {"title": "Software Engineer - PayCo - Bengaluru, Karnataka - Indeed.com", "url": "https://in.indeed.com/viewjob?jk=1", "snippet": "Python"},
        {"title": "Software Engineer - OtherCo - Pune - Indeed.com", "url": "https://in.indeed.com/viewjob?jk=2", "snippet": ""},
    ])
    jobs = search_indeed_jobs(indeed, "PayCo", "software engineer")
    assert len(jobs) == 1 and jobs[0]["location"] == "Bengaluru, Karnataka"


def test_goal_parsing_fallback():
    spec = parse_spec_fallback("Find 200 fintech companies in India")
    assert spec.count == 200 and spec.sectors == ["fintech"] and spec.geographies == ["India"]
    spec = parse_spec_fallback("Series A/B AI startups")
    assert spec.funding_stages == ["series_a", "series_b"] and "ai" in spec.sectors
    spec = parse_spec_fallback("Trading companies, reach engineering managers or recruiters")
    assert "trading" in spec.sectors and spec.personas == ["engineering_manager", "recruiter"]


def test_candidate_merge_dedupes_and_combines_sources():
    merged = merge_candidates([
        CompanyCandidate(name="PayCo Pvt Ltd", domain="https://www.payco.in/", source="llm"),
        CompanyCandidate(name="Payco", sector="fintech", hiring=True, source="linkedin_jobs"),
        CompanyCandidate(name="Other", domain="linkedin.com", source="web_search"),
    ])
    assert len(merged) == 2
    payco = merged[0]
    assert payco.domain == "payco.in" and payco.sector == "fintech" and payco.hiring is True
    assert payco.sources == ["llm", "linkedin_jobs"]
    assert merged[1].domain is None  # aggregator domains are not company domains


def test_contact_ranking_is_size_aware():
    cfg = load_config("config.example.yaml")
    people = merge_contact_candidates([
        ContactCandidate(name="Asha Rao", title="Co-founder & CEO", source="team_page"),
        ContactCandidate(name="Vikram Iyer", title="Engineering Manager", source="linkedin_search"),
        ContactCandidate(name="Neha Gupta", title="Technical Recruiter", source="linkedin_search"),
        ContactCandidate(name="Vikram Iyer", title="Engineering Manager", source="apollo", email="vikram@payco.in", email_confidence=0.8),
    ])
    assert len(people) == 3
    vikram = next(p for p in people if p.name == "Vikram Iyer")
    assert vikram.sources == ["linkedin_search", "apollo"] and vikram.email == "vikram@payco.in"

    small = rank_contacts(list(people), Company(name="Tiny", employee_count=15), cfg)
    assert small[0].role_category == "founder"
    large = rank_contacts(list(people), Company(name="Big", employee_count=800), cfg)
    assert large[0].role_category == "engineering_manager"

    cfg.contacts.persona_strategy = "ordered"
    cfg.contacts.personas = ["recruiter", "engineering_manager"]
    ordered = rank_contacts(list(people), Company(name="Big", employee_count=800), cfg)
    assert ordered[0].role_category == "recruiter"
