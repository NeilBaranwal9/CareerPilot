# Autonomous Cold-Email & Job Outreach Engine

A resumable, end-to-end cold-outreach platform for job seekers. Give it a goal like
**"Find 200 fintech companies in India"** and it discovers companies, researches and scores them, finds the right
people (engineering managers, recruiters, founders), finds and verifies their email addresses, picks the best resume
variant, writes persona-specific emails with follow-ups, creates Gmail drafts (or sends on a schedule), tracks
replies/bounces, stops follow-ups after a reply, and shows the full conversion funnel.

LLM: **Groq** by default (Llama 3.3 70B for writing, Llama 3.1 8B for extraction), with OpenAI, Anthropic, Gemini
and local AGY also supported.

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
uv run recruiting-platform funnel
uv run recruiting-platform schedule install   # daily discovery + outreach cycle every 30 min (Windows/Linux)
```

---

## ✅ What it can do

### 1. Job & company discovery
| Question | Answer |
| :--- | :--- |
| Sources | **LinkedIn** (public guest jobs API), **Greenhouse, Lever, Ashby, Workable, SmartRecruiters** (public board APIs, auto-detected from careers pages or probed by name), **company career pages**, **Wellfound** and **Indeed** (via search results, since both block scrapers), the **Y Combinator directory**, "top X startups" list articles, and the LLM's knowledge. Toggle with `discovery.sources` / `discovery.job_sources`. |
| Companies without a job posting? | Yes. `allow_speculative_outreach` creates a speculative application so hiring managers can still be contacted. |
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

### 3. Contact discovery
Sources: **LinkedIn profile results from search engines** (titles/snippets only; profiles are never scraped), **company team/about/leadership
pages**, **press releases & interviews**, **GitHub org members**, **Hunter** and **Apollo** (optional keys).
Titles are classified into **recruiter / hiring_manager / engineering_manager / founder / cto / vp_engineering / tech_lead**.
All contacts are stored with role, persona, source, LinkedIn/GitHub URL, background and a **rank score**. The top contact is chosen by
persona preference (size-aware in `auto`: founders at small startups, EMs/recruiters at larger companies; or strict `ordered`),
source confidence, reachability and learned reply rates. `max_contacts_per_company: 2` reaches e.g. an EM and a recruiter in separate threads.
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
The LLM gets the company description, product, stack, **recent news/launches**, the **job description**, the **contact's
name, role and background**, **your resume text**, your ranked highlights, persona guidelines and tone. It opens with a specific,
sourced observation ("I saw you recently launched …") and never invents facts. A validation stage rejects placeholders and fabricated claims.

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
recruiting-platform daily      # or `schedule install` / `daemon`: continues the campaign, sends/follows up, tracks replies
recruiting-platform funnel --campaign 1
#  Companies Found → Contacts Found → Emails Found → Emails Verified → Drafts Created → Emails Sent → Replies → Interviews
```
Duplicate outreach is prevented at every level: one thread per company, company cool-down, same address never emailed twice,
Gmail Sent-folder check, do-not-contact after "not interested". Interviews are detected from replies and calendar invites,
or recorded with `recruiting-platform mark <app_id> interview`.

---

## ⚙️ Configuration highlights (`config.yaml`)

```yaml
llm:
  provider: "groq"
  model: "llama-3.3-70b-versatile"     # writing
  fast_model: "llama-3.1-8b-instant"   # extraction / classification
  fallback_models: ["openai/gpt-oss-120b", "llama-3.1-8b-instant"]
  api_key: ""                          # or GROQ_API_KEY

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
| `campaign "<goal>" [--target N] [--personas a,b] [--auto-send]` | Start a campaign from a natural-language goal |
| `continue-campaign <id>` / `campaigns` | Next batch of a campaign / list campaigns with progress |
| `discover "<query>" [--count N] [--research]` | Build a classified, fit-scored target list without outreach |
| `companies [--sector fintech] [--sort probability]` | Rank companies by fit and reply probability |
| `run` / `targeted "<company>"` | Classic pipeline run / one specific company (optionally `--jd`) |
| `outreach` (alias `send-scheduled`) | Replies & bounces → send due emails → follow-ups |
| `funnel [--campaign id]` / `insights` | Conversion funnel & verification stats / what's working (learned rates) |
| `mark <app_id> interview` | Record outcomes manually (stops follow-ups) |
| `daily` / `daemon` / `schedule install` | One automation cycle / foreground loop / OS scheduler (Windows + Linux) |
| `doctor` | Verify Groq key & models, Gmail scopes, SMTP port 25, Typst, keys, resumes |
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
