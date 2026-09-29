# Autonomous Cold-Email & Job Outreach Engine

A resumable, end-to-end cold-outreach platform for job seekers. Give it a goal like
**"Find 200 fintech companies in India"** and it discovers companies, researches and scores them, finds the right
people (engineering managers, recruiters, founders), finds and verifies their email addresses, picks the best resume
variant, writes persona-specific emails with follow-ups, creates Gmail drafts (or sends on a schedule), tracks
replies/bounces, stops follow-ups after a reply, and shows the full conversion funnel.

LLMs: **hybrid routing**. A local **Ollama** model (default `qwen3:8b`) handles discovery, research, contact and email finding, scoring and validation; **Groq** is used only for writing emails (`gpt-oss-120b`) and follow-ups. Every call's tokens are recorded per stage, and Groq has a daily budget with a reserve kept for email writing. OpenAI, Anthropic and Gemini are also supported.

---

## 🚀 Quick start

```bash
make install                      # or: uv sync
cp config.example.yaml config.yaml
# edit config.yaml: user_identity, job_preferences, target_profile, resumes, highlights
export GROQ_API_KEY=gsk_...       # Windows PowerShell: $env:GROQ_API_KEY="gsk_..."
make auth                         # Gmail OAuth (compose + read scopes)
uv run recruiting-platform doctor # checks Groq key/models, Gmail scopes, SMTP port 25, Typst, API keys

uv run recruiting-platform campaign "Find 200 fintech companies in India, reach engineering managers or recruiters"
uv run recruiting-platform campaign "Find fintech startups in India and ask if they have internship opportunities for me"
uv run recruiting-platform funnel
uv run recruiting-platform schedule install   # daily discovery + outreach cycle every 30 min (Windows/Linux)
```

---

## ✅ What it can do

### 1. Job & company discovery
| Question | Answer |
| :--- | :--- |
| Sources | **LinkedIn** (public guest jobs API, only with `allow_linkedin: true`; off by default), **Greenhouse, Lever, Ashby, Workable, SmartRecruiters** (public board APIs, auto-detected from careers pages or probed by name), **company career pages**, **Wellfound** and **Indeed** (via search results, since both block scrapers), the **Y Combinator directory**, "top X startups" list articles, and the LLM's knowledge. Toggle with `discovery.sources` / `discovery.job_sources`. |
| Companies without a job posting? | Yes, with `discovery.mode: company_outreach` or `hybrid`: the company gets a **company-level inquiry** asking about current or upcoming internships. No job title is invented. In the default `job_search` mode such companies are skipped. See [Discovery Modes](#-discovery-modes). |
| Which roles are searched? | The ones in `job_preferences.roles`. Queries are built from your roles' families (e.g. `"Razorpay" jobs product analyst data analyst`), never a hard-coded "engineer". |
| "Fintech companies in India", "Series A/B AI startups", "Trading companies" | Yes: `recruiting-platform discover "Series A/B AI startups" --count 50 --research` or `campaign "..."`. The goal is parsed into sector, stage, geography, size and personas (by the LLM with a rule-based fallback). |
| Continuous discovery every day? | Yes: `discovery.queries` + `schedule install` (Windows Task Scheduler / systemd) or `daemon`. Already-tracked companies are excluded, and a backlog continues the next day. |
| Companies stored separately from jobs? | Yes: `companies`, `jobs`, `contacts`, `applications`, `emails`, `campaigns`, `outreach_events` tables. |

### 2. Company research
Per company: description, product, **sector + sub-sectors** (fintech, trading, healthtech, edtech, ai, devtools, saas, …),
**funding stage** (normalized: seed … series_d_plus / public / bootstrapped), **headcount**, **hiring status** + open-role count,
**tech stack** (+ Apollo technologies), **recent news/launches**, location, ATS board, GitHub org.
Classification is automatic (LLM + keyword classifier). Every company gets a **fit score** from your `target_profile`
(sector preference, stage, size, hiring, stack overlap, geography) and is **rejected** if it doesn't fit
(`allowed_sectors`, `min_company_fit`, exclusions).

### 3. Contact discovery (LinkedIn-free)
LinkedIn is never searched or scraped (`discovery.allow_linkedin: false`); LinkedIn URLs found elsewhere are kept as metadata only. Sources, in order: team / leadership / about pages found from the homepage's own links, engineering and product blogs, press releases and conference speakers, GitHub, The Org, Crunchbase and Wellfound (search snippets), plus Hunter/Apollo when keys are set. **A person proposed by the LLM is kept only if their name appears in the fetched text**, and every contact stores the URL it came from and a source-based confidence. The notes below describe ranking.

Sources: **LinkedIn profile results from search engines** (titles/snippets only; profiles are never scraped), **company team/about/leadership
pages**, **press releases & interviews**, **GitHub org members**, **Hunter** and **Apollo** (optional keys).
Titles are classified into **recruiter / hiring_manager / engineering_manager / founder / cto / vp_engineering / tech_lead**.
All contacts are stored with role, persona, source, LinkedIn/GitHub URL, background and a **rank score**. The top contact is chosen by
persona preference (size-aware in `auto`: founders at small startups, EMs/recruiters at larger companies; or strict `ordered`),
source confidence, reachability and learned reply rates. For company-level inquiries the preference also follows your target
role families (product/data → product lead, PM, data lead; engineering → EM, head of engineering; QA → QA lead, EM). `max_contacts_per_company: 2` reaches e.g. an EM and a recruiter in separate threads.
When no email is public, one is inferred and verified (see 4). With no named person, it falls back to the careers inbox.

### 4. Email discovery & verification
Evidence, strongest first: address from Hunter/Apollo/GitHub → address printed on the company site or in **public commit
emails** → **company pattern inferred** from known addresses (`{first}.{last}`, `{f}{last}`, …) → Hunter email-finder /
Apollo match → LLM reading search results → ranked common patterns.
Verification: a real **SMTP RCPT TO probe** (no email is sent) with **catch-all detection** in a single session,
anti-spam policy blocks treated as "unknown" rather than "invalid", then Hunter's verifier, then evidence-weighted
acceptance. Statuses: `valid`, `catch_all`, `unverified`, `invalid`. **Catch-all domains**: the address with the strongest evidence
(an inferred pattern beats a generic guess) is used and marked `catch_all`. Bounces mark the address invalid and automatically retry with the
next-best address. `funnel` shows the exact % SMTP-verified vs deliverable-likely. SMTP needs outbound port 25 (`doctor` checks it).

### 5. Resume tailoring
Resumes can be **PDF, DOCX, Typst, Markdown or text**.
- `generate_resume: false`: the file is validated and attached unchanged.
- `generate_resume: true` with a PDF/DOCX: text is extracted, the LLM tailors it into a structured resume (reordering, rewriting bullets, applying the variant's `focus`), a **fabrication guard** rejects any new employer/date/skill, and a one-page PDF is rendered (no Typst needed). Any failure attaches your original file.

Any number of **resume variants** (e.g. fintech / backend / aiml / research) are chosen by **explicit rules** (`sectors`, `title_any`, `description_any`, `stack_any`, `personas`, `title_none`, plus `priority` and a `default`). Variants can share one PDF and differ only in `focus`. Without rules, selection falls back to tag scoring.
**Highlights** (e.g. Jio internship, GSIH finalist, Team Ignition) are **ranked per role** and passed to both the resume tailoring
(to reorder projects and rewrite whole bullet points, with no fabrication) and the email. PDFs are compiled with Typst automatically.
If a tailored resume fails to compile, the base variant's PDF is attached instead.

### 6. Personalization
The LLM gets the company description, product, stack, **recent news/launches**, the **job description** (or, for a
company-level inquiry, your target areas and the fact that no opening is known), the **contact's name, role and background**, **your resume text**, your ranked highlights, persona guidelines and tone. It opens with a specific,
sourced observation ("I saw you recently launched …") and never invents facts. A validation stage rejects placeholders and fabricated claims.

**Voice & anti-template rules** (`src/outreach/voice.py`): emails are written like a student who researched the
company for five minutes. The opening style is chosen from the research that exists (engineering post, news, job
posting, product) and rotated across consecutive emails. Asks depend on the recipient: recruiters are asked about
openings and process, founders get shorter notes, and referrals are only requested from people who can give one.
Deterministic checks trigger one rewrite for: stock phrases ("which aligns with", "I am keen to", "data-driven"…)
repeated within an email or across recent ones, tech-stack-list sentences, more than one "why this connects" sentence,
openings/asks/sentences copied from recent emails, and generic or repeated subjects. Unsupported claims about what a
company needs or focuses on, invented job openings, invented relationships and assumed recipient responsibilities
block the draft. The local model writes a plain-language version of each highlight once, and that wording is used
instead of the resume's jargon.

### 7. Outreach strategy
Initial email + **follow-up #1 and #2** (threaded replies, configurable delays). **Tone** is configurable globally or per persona.
Separate writing guidelines for **founders** (short, under 90 words), **recruiters**, **engineering managers**, hiring managers, CTOs, tech leads and generic inboxes.

### 8. Gmail
Drafts (review-first, default) **or automatic sending** (`outreach.auto_send` or `campaign --auto-send`) inside a **send window**
(e.g. weekdays 9–12 IST) with daily limits and spacing. **Reply detection** (thread reading), **bounce detection**, **auto-reply/OOO
detection**, **reply classification** (interview / positive / referral / not interested), **follow-ups stop automatically on reply**,
and drafts you send manually from Gmail are detected so their follow-up clock starts.

### 9. Scoring & learning
**Hybrid** scoring (`scoring.mode`): deterministic rules (role/experience/salary/stack/company-fit/data completeness) blended with the
LLM's judgement. Your explicit preferences (`sector_weights`, e.g. fintech 1.0 > ai 0.85 > generic 0.5) are combined with
**learned reply/interview rates** per sector, persona, funding stage, size and resume variant (Bayesian-smoothed),
producing a **reply probability** per company/application. `companies --sort probability` ranks by it, and processing order follows it.

### 10. The full goal
```bash
recruiting-platform campaign "Find 200 fintech companies in India" --personas engineering_manager,recruiter
#  → discovers & qualifies companies → finds EMs/recruiters → verifies emails → personalized drafts (+2 follow-ups)
recruiting-platform campaign "Find fintech startups in India, use actual openings when available, otherwise ask companies about internship opportunities"
#  → hybrid mode: job-specific drafts where a matching opening exists, company-level inquiries everywhere else
recruiting-platform daily      # or `schedule install` / `daemon`: continues the campaign, sends/follows up, tracks replies
recruiting-platform funnel --campaign 1
#  Companies Found → Contacts Found → Emails Found → Emails Verified → Drafts Created → Emails Sent → Replies → Interviews
```
Duplicate outreach is prevented at every level: one thread per company, company cool-down, same address never emailed twice,
Gmail Sent-folder check, do-not-contact after "not interested". Interviews are detected from replies and calendar invites,
or recorded with `recruiting-platform mark <app_id> interview`.

---

## 🧭 Discovery Modes

`discovery.mode` decides what happens with each relevant company. The default is `job_search`, the original behaviour.

| Mode | Per company | Good for |
| :--- | :--- | :--- |
| `job_search` (default) | Find real public openings and match them against `job_preferences.roles`. Only matching openings get outreach; companies without one are skipped. | Applying to posted roles only |
| `company_outreach` | Research the company and check its own careers page / ATS board. A matching listed opening is used; otherwise a **company-level inquiry** asks whether they have current or upcoming internships. A public opening is not required. | Startups that take interns without posting |
| `hybrid` | Full job search first. Matching opening → job-specific email; no matching opening → company-level inquiry instead of dropping a relevant company. | Usually the most useful |

### Job Search
Find and target actual public job openings that match your configured roles (ATS boards, careers pages, Wellfound,
Indeed and web search). If no posting matches, the company is skipped: nothing is created for it.

### Company Outreach
Find relevant companies even when there is no public opening, and ask about current or upcoming opportunities. Only
the company's own careers page / ATS board is checked (no job-board searches), so it is faster and uses fewer searches.
If that page does list a matching opening, the email refers to that real opening.

### Hybrid
Target real openings when available, and use company-level speculative outreach when no suitable public opening is
found. It runs the full job search, so it is slower than `company_outreach` but never misses a posted role.

> **Company outreach is NOT a fake job application. It is a speculative company-level inquiry.**

CareerPilot never claims an opening exists unless it actually discovered one, and it never invents a job title. (Older
versions created records such as "Product Analyst Intern (Speculative Application)". No mode does that any more, and
`job_preferences.allow_speculative_outreach` is deprecated: it only logs a warning.) A company-level inquiry is stored as
an application with `outreach_type: company_speculative`, linked to a placeholder record titled
**"Company-level internship inquiry"** (`jobs.source = company_outreach`) whose description says that no matching public
opening was found. Job-specific outreach has `outreach_type: job`.

A company-level email:
- opens with one specific, verified observation about the company;
- says who you are and describes **one** relevant experience or project;
- mentions your target areas (role families) naturally, without claiming that a specific role is open;
- asks, as a genuine question, whether there are current or upcoming internship opportunities
  (`company_outreach.ask_about_openings: true`; `false` uses the recipient's usual ask instead);
- is about 100–150 words, is written by Groq like every other email, and passes the same fact, grounding and
  anti-template checks. Validation also rejects "applying for the X role", "your open positions", "I saw your opening
  for …" and stated hiring plans ("you're growing the team"), and sends such a draft back for a rewrite.

Recipients come from the normal contact discovery (LinkedIn stays disabled). For a company-level inquiry, your target
role families decide who is preferred among the personas in `contacts.personas`:

| Target roles | Preferred recipients |
| :--- | :--- |
| Product / business analysis | product lead, product manager, recruiter |
| Data / analytics | data lead, product lead, product manager, recruiter |
| Software engineering | engineering manager, head of engineering, hiring manager, recruiter |
| QA / SDET | QA lead, engineering manager, recruiter |
| AI/ML | data lead, engineering manager, head of engineering, recruiter |
| Early-stage startup (≤ 50 people, pre-seed or seed) | founder first |

With `persona_strategy: ordered` your explicit persona order is kept as-is.

#### Configuration examples
```yaml
# Job search only (the default; same as leaving `mode` out)
discovery:
  mode: "job_search"
```
```yaml
# Company outreach: ask companies about internships, no public opening required
discovery:
  mode: "company_outreach"
company_outreach:
  ask_about_openings: true
```
```yaml
# Hybrid: real openings first, company-level inquiry otherwise
discovery:
  mode: "hybrid"
company_outreach:
  ask_about_openings: true
```

#### Role-aware job search
Job-search queries are derived from `job_preferences.roles`, not hard-coded to engineering. Roles are grouped into
families (product, data/analytics, business analysis, QA/testing, software engineering, AI/ML) and the first role of
each of your first two families becomes the search terms, so queries stay short:

| `roles` (first entries) | Wellfound query |
| :--- | :--- |
| Product Analyst Intern, Associate Product Manager Intern, Data Analyst Intern | `site:wellfound.com "Razorpay" jobs product analyst data analyst` |
| Software Engineer Intern, Backend Engineer Intern | `site:wellfound.com "Razorpay" jobs software engineer` |
| QA Engineer Intern, SDET Intern | `site:wellfound.com "Razorpay" jobs qa engineer` |

List your roles in order of preference. ATS boards and careers pages are read in full and matched against **all** your
roles: an exact role scores highest, a posting in the same family (e.g. "Business Analyst - Intern") also counts, and
generic engineering titles only count when you target engineering roles. The run log shows the terms used:
`[DISCOVERY] Job-search terms from your target roles: product analyst, data analyst`.

#### Campaign goals and precedence
```bash
recruiting-platform campaign "Find fintech startups in India and ask if they have internship opportunities for me"
#  → company_outreach
recruiting-platform campaign "Find fintech startups in India, use actual openings when available, otherwise ask companies about internship opportunities"
#  → hybrid
recruiting-platform campaign "Find 50 fintech startups in India" --mode hybrid
```
Precedence: `--mode` > `discovery.mode` when set in `config.yaml` > the mode implied by the goal > `job_search`.
A campaign keeps the mode it was created with (`campaigns` shows it). `run`, `daily` and standing `discovery.queries` use
`discovery.mode`. `targeted "<company>"` falls back to a company-level inquiry when the company you named has no matching opening.

#### Logs
```
[DISCOVERY] Mode: hybrid
[DISCOVERY] Searching for actual openings...
[DISCOVERY] Job-search terms from your target roles: product analyst, data analyst
[OUTREACH] Matching opening found -> job-specific outreach (Product Analyst Intern)
[OUTREACH] No matching public opening found for QuietCo
[OUTREACH] Hybrid mode -> falling back to company-level outreach
[OUTREACH] Creating company-level speculative outreach
```

---

## 🧠 Hybrid LLM routing & token accounting

```yaml
llm:
  provider: "groq"
  model: "openai/gpt-oss-120b"       # email writing
  fast_model: "qwen/qwen3.8-27b"     # follow-ups; fallback if the local model is down
  local_provider: "ollama"
  local_model: "qwen3:8b"            # ollama pull qwen3:8b
  routing: {email_generation: premium, email_regeneration: premium, followup_generation: premium_fast,
            resume_tailoring: premium, default: local}
  groq_daily_token_budget: 190000
  groq_reserved_for_emails: 120000   # local->Groq fallbacks can never eat this share
```

- `recruiting-platform usage` shows tokens per stage/provider and today's Groq budget (also on the dashboard's Funnel tab).
- If Groq is out of budget or rate limited, email generation is **deferred** to a later run. It never falls back to a generic template.
- `recruiting-platform explain <app_id>` shows the reply-probability breakdown (company fit, hiring signal, contact
  seniority, email confidence, resume match), the contact's source and the email evidence.
- Emails and tailored resumes pass a deterministic fact check. Awards, rankings, hackathon wins, year of study, CGPA,
  numbers and technologies must appear in your resume, profile notes or the company research.
- Duplicate outreach is blocked via an `outreach_ledger` (same email, person, company or role), plus Gmail Sent/Drafts checks.

## ⚙️ Configuration highlights (`config.yaml`)

```yaml
llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile"     # writing
  fast_model: "llama-3.1-8b-instant"   # extraction / classification
  fallback_models: ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
  api_key: ""                          # or GROQ_API_KEY

discovery:
  mode: "hybrid"                       # job_search (default) | company_outreach | hybrid

target_profile:
  sector_weights: {fintech: 1.0, trading: 0.95, ai: 0.85, saas: 0.65, generic: 0.5}
  funding_stages: ["seed", "series_a", "series_b", "series_c"]
  skills: ["python", "fastapi", "react", "postgresql"]

outreach:
  auto_send: false
  send_window: {timezone: "Asia/Kolkata", start_hour: 9, end_hour: 12, weekdays_only: true}
  followups: [{after_days: 4}, {after_days: 7}]
```
See `config.example.yaml` for every option (discovery sources, contact personas, email verification, resume variants,
highlights, API keys, learning). Existing databases are migrated automatically (new columns and tables are added in place).

Optional keys (env vars work too): `HUNTER_API_KEY`, `APOLLO_API_KEY`, `GITHUB_TOKEN`, `SERPER_API_KEY` / `BRAVE_API_KEY`.
Keyless search (DuckDuckGo → Yahoo) is throttled by those engines. For 100+ companies per day a Serper or Brave key is strongly recommended.

---

## 🛠️ Commands

| Command | Description |
| :--- | :--- |
| `campaign "<goal>" [--target N] [--personas a,b] [--mode job_search\|company_outreach\|hybrid] [--auto-send]` | Start a campaign from a natural-language goal (the mode can also come from the goal's wording) |
| `continue-campaign <id>` / `campaigns` | Next batch of a campaign / list campaigns with progress |
| `discover "<query>" [--count N] [--research]` | Build a classified, fit-scored target list without outreach |
| `companies [--sector fintech] [--sort probability]` | Rank companies by fit and reply probability |
| `run` / `targeted "<company>"` | Classic pipeline run / one specific company (optionally `--jd`) |
| `outreach` (alias `send-scheduled`) | Replies & bounces → send due emails → follow-ups |
| `funnel [--campaign id]` / `insights` | Conversion funnel & verification stats / what's working (learned rates) |
| `mark <app_id> interview` | Record outcomes manually (stops follow-ups) |
| `daily` / `daemon` / `schedule install` | One automation cycle / foreground loop / OS scheduler (Windows + Linux) |
| `doctor` | Verify Groq key & models, Ollama model, routing table, Playwright, Gmail scopes, SMTP port 25, Typst, providers |
| `usage [--days N]` | Token usage per stage/task/provider and the Groq daily budget |
| `explain <app_id>` | Why a contact/email was chosen and the reply-probability breakdown |
| `ui` | Dashboard on http://localhost:18492 (funnel, campaigns, companies, contacts, applications) |
| `search` / `research` / `tailor` / `draft` / `resume` / `retry` | Stage-range runs & recovery |
| `export` / `clean-invalid` / `status` / `config-summary` | Data export & hygiene |

Every command is also available via `make` (see `make help`).

---

## 🛡️ Security & Privacy

- `credentials.json`, `token.json`, `config.yaml`, `data/`, `logs/`, `exports/` and personal resumes are git-ignored.
- API keys can live in environment variables. Nothing is hardcoded.
- SMTP verification only issues `RCPT TO` and quits; it never sends a message. Default mode is **drafts only**.
- Respect recipients: keep daily limits modest, honour "not interested" replies (they are marked do-not-contact automatically).

---

## 📜 License

Source-Available under the [PolyForm Noncommercial License 1.0.0](LICENSE). Free for personal, educational, and non-commercial use.
