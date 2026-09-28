"""Optional enrichment APIs: Hunter.io, Apollo.io and GitHub (all network I/O goes through BrowserProvider)."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from src.providers.browser import BrowserProvider

logger = logging.getLogger("recruiting-platform.sources.enrichment")


@dataclass
class PersonRecord:
    name: str
    title: str = ""
    email: str | None = None
    email_confidence: float = 0.0
    linkedin_url: str | None = None
    github_url: str | None = None
    background: str | None = None
    source: str = ""


@dataclass
class DomainEmailIntel:
    pattern: str | None = None
    accept_all: bool | None = None
    people: list[PersonRecord] = field(default_factory=list)
    samples: list[tuple[str, str]] = field(default_factory=list)  # (full name, email)


# ---------------------------------------------------------------------------
# Hunter.io
# ---------------------------------------------------------------------------


class HunterClient:
    BASE = "https://api.hunter.io/v2"

    def __init__(self, browser: BrowserProvider, api_key: str):
        self.browser = browser
        self.api_key = api_key

    def domain_search(self, domain: str, limit: int = 10) -> DomainEmailIntel:
        data = self.browser.fetch_json(
            f"{self.BASE}/domain-search", params={"domain": domain, "limit": limit, "api_key": self.api_key}
        ).get("data", {})
        intel = DomainEmailIntel(pattern=data.get("pattern"), accept_all=data.get("accept_all"))
        for e in data.get("emails", []):
            name = " ".join(x for x in (e.get("first_name"), e.get("last_name")) if x)
            if not name or not e.get("value"):
                continue
            intel.samples.append((name, str(e["value"]).lower()))
            intel.people.append(
                PersonRecord(
                    name=name,
                    title=e.get("position") or "",
                    email=str(e["value"]).lower(),
                    email_confidence=float(e.get("confidence") or 0) / 100.0,
                    linkedin_url=e.get("linkedin"),
                    source="hunter",
                )
            )
        return intel

    def email_finder(self, domain: str, first_name: str, last_name: str) -> tuple[str | None, float]:
        data = self.browser.fetch_json(
            f"{self.BASE}/email-finder",
            params={"domain": domain, "first_name": first_name, "last_name": last_name, "api_key": self.api_key},
        ).get("data", {})
        email = data.get("email")
        return (str(email).lower() if email else None), float(data.get("score") or 0) / 100.0

    def verify(self, email: str) -> str:
        """Returns valid / invalid / catch_all / unknown."""
        data = self.browser.fetch_json(
            f"{self.BASE}/email-verifier", params={"email": email, "api_key": self.api_key}
        ).get("data", {})
        status = str(data.get("status") or data.get("result") or "unknown").lower()
        return {
            "valid": "valid",
            "deliverable": "valid",
            "invalid": "invalid",
            "undeliverable": "invalid",
            "accept_all": "catch_all",
            "risky": "catch_all",
        }.get(status, "unknown")


# ---------------------------------------------------------------------------
# Apollo.io
# ---------------------------------------------------------------------------

PERSONA_TITLES: dict[str, list[str]] = {
    "engineering_manager": ["engineering manager", "software engineering manager"],
    "hiring_manager": ["hiring manager"],
    "recruiter": ["technical recruiter", "talent acquisition", "recruiter", "hr manager"],
    "founder": ["founder", "co-founder"],
    "cto": ["cto", "chief technology officer"],
    "vp_engineering": ["vp engineering", "head of engineering", "director of engineering"],
    "tech_lead": ["tech lead", "engineering lead"],
}


class ApolloClient:
    BASE = "https://api.apollo.io/api/v1"

    def __init__(self, browser: BrowserProvider, api_key: str):
        self.browser = browser
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        return {"X-Api-Key": self.api_key, "Content-Type": "application/json", "Cache-Control": "no-cache"}

    def search_people(self, domain: str, personas: list[str], per_page: int = 10) -> list[PersonRecord]:
        titles = [t for p in personas for t in PERSONA_TITLES.get(p, [p.replace("_", " ")])]
        body = {"q_organization_domains_list": [domain], "person_titles": titles, "page": 1, "per_page": per_page}
        data: Any = None
        for endpoint in ("mixed_people/api_search", "mixed_people/search"):
            try:
                data = self.browser.fetch_json(
                    f"{self.BASE}/{endpoint}", method="POST", headers=self._headers(), json_body=body
                )
                break
            except Exception as e:
                logger.debug(f"Apollo {endpoint} failed: {e}")
        if not data:
            return []
        people = []
        for p in data.get("people", []) or data.get("contacts", []):
            name = p.get("name") or " ".join(x for x in (p.get("first_name"), p.get("last_name")) if x)
            if not name:
                continue
            email = p.get("email")
            if email and ("not_unlocked" in email or "domain.com" in email):
                email = None
            history = p.get("employment_history") or []
            background = "; ".join(
                f"{h.get('title')} at {h.get('organization_name')}" for h in history[:3] if h.get("organization_name")
            )
            people.append(
                PersonRecord(
                    name=name,
                    title=p.get("title") or "",
                    email=email.lower() if email else None,
                    email_confidence=0.85 if p.get("email_status") == "verified" else 0.5,
                    linkedin_url=p.get("linkedin_url"),
                    background=background or p.get("headline"),
                    source="apollo",
                )
            )
        return people

    def match_person(self, first_name: str, last_name: str, domain: str) -> tuple[str | None, float]:
        data = self.browser.fetch_json(
            f"{self.BASE}/people/match",
            method="POST",
            headers=self._headers(),
            json_body={"first_name": first_name, "last_name": last_name, "domain": domain, "reveal_personal_emails": False},
        )
        person = data.get("person") or {}
        email = person.get("email")
        if not email or "not_unlocked" in email:
            return None, 0.0
        return str(email).lower(), 0.9 if person.get("email_status") == "verified" else 0.55

    def enrich_organization(self, domain: str) -> dict[str, Any]:
        data = self.browser.fetch_json(
            f"{self.BASE}/organizations/enrich", headers=self._headers(), params={"domain": domain}
        )
        org = data.get("organization") or {}
        return {
            "employee_count": org.get("estimated_num_employees"),
            "industry": org.get("industry"),
            "funding_stage": org.get("latest_funding_stage"),
            "total_funding": org.get("total_funding_printed"),
            "technologies": org.get("technology_names") or [],
            "keywords": org.get("keywords") or [],
            "description": org.get("short_description"),
            "linkedin_url": org.get("linkedin_url"),
            "location": ", ".join(x for x in (org.get("city"), org.get("country")) if x) or None,
        }


# ---------------------------------------------------------------------------
# GitHub
# ---------------------------------------------------------------------------


class GitHubClient:
    BASE = "https://api.github.com"

    def __init__(self, browser: BrowserProvider, token: str = ""):
        self.browser = browser
        self.token = token

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return self.browser.fetch_json(f"{self.BASE}{path}", headers=headers, params=params)

    def find_org(self, company_name: str, domain: str | None) -> str | None:
        """Finds the company's GitHub organization, verified against the website domain when possible."""
        slug = re.sub(r"[^a-z0-9-]", "", company_name.lower().replace(" ", "-"))
        guesses = [g for g in {slug, slug.replace("-", ""), (domain or "").split(".")[0]} if g]
        for guess in guesses:
            try:
                org = self._get(f"/orgs/{guess}")
            except Exception:
                continue
            blog = str(org.get("blog") or "").lower()
            if not domain or domain.lower() in blog or company_name.lower() in str(org.get("name") or "").lower():
                return str(org.get("login"))
        try:
            data = self._get("/search/users", params={"q": f"{company_name} type:org", "per_page": 5})
            for item in data.get("items", []):
                org = self._get(f"/orgs/{item['login']}")
                if domain and domain.lower() in str(org.get("blog") or "").lower():
                    return str(org.get("login"))
        except Exception as e:
            logger.debug(f"GitHub org search failed for {company_name}: {e}")
        return None

    def org_people(self, org: str, limit: int = 8) -> list[PersonRecord]:
        people: list[PersonRecord] = []
        members = self._get(f"/orgs/{org}/public_members", params={"per_page": 30})
        for member in members[:limit] if isinstance(members, list) else []:
            try:
                user = self._get(f"/users/{member['login']}")
            except Exception:
                continue
            if not user.get("name"):
                continue
            bio = user.get("bio") or ""
            people.append(
                PersonRecord(
                    name=str(user["name"]),
                    title=bio[:120] or "Engineer",
                    email=str(user["email"]).lower() if user.get("email") else None,
                    email_confidence=0.8 if user.get("email") else 0.0,
                    github_url=user.get("html_url"),
                    background=f"GitHub: {user.get('public_repos', 0)} public repos. {bio}".strip(),
                    source="github",
                )
            )
        return people

    def commit_email_samples(self, org: str, domain: str, repos: int = 3) -> list[tuple[str, str]]:
        """Collects (author name, email) pairs from recent public commits that use the company domain."""
        samples: list[tuple[str, str]] = []
        repo_list = self._get(f"/orgs/{org}/repos", params={"sort": "pushed", "per_page": repos})
        for repo in repo_list if isinstance(repo_list, list) else []:
            try:
                commits = self._get(f"/repos/{org}/{repo['name']}/commits", params={"per_page": 50})
            except Exception:
                continue
            for c in commits if isinstance(commits, list) else []:
                author = (c.get("commit") or {}).get("author") or {}
                email = str(author.get("email") or "").lower()
                name = str(author.get("name") or "")
                if email.endswith("@" + domain.lower()) and name and (name, email) not in samples:
                    samples.append((name, email))
        return samples

    def org_languages(self, org: str) -> list[str]:
        repo_list = self._get(f"/orgs/{org}/repos", params={"sort": "pushed", "per_page": 10})
        counts: dict[str, int] = {}
        for repo in repo_list if isinstance(repo_list, list) else []:
            lang = repo.get("language")
            if lang:
                counts[lang] = counts.get(lang, 0) + 1
        return [lang for lang, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
