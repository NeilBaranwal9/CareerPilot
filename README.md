# Autonomous Cold-Email & Job Outreach Engine

A resumable, end-to-end cold-outreach platform for job seekers. Give it a goal like
**"Find 200 fintech companies in India"** and it discovers companies, researches and scores them, finds the right
people (recruiters, hiring managers, engineering/product leads, founders), finds and verifies their email addresses,
checks whether a relevant opening exists (and never pretends one does), picks the best resume
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

# config.yaml → discovery.mode: "company_outreach"
uv run recruiting-platform campaign "Find 200 fintech companies in India" --batch 20 --max-stage 5 --drafts-only
uv run recruiting-platform continue-campaign 1   # writes the Gmail drafts
uv run recruiting-platform funnel --campaign 1
uv run recruiting-platform schedule install      # daily run at 09:00 + outreach cycle every 30 min (Windows/Linux)
```
New here? Follow the [step-by-step guide](#-step-by-step-guide-find-200-fintech-companies-in-india) below.

---

## 📘 Step-by-step guide: "Find 200 fintech companies in India"

This guide runs a complete **company-outreach campaign**. Here "fintech" describes *which companies to contact*, not a
job keyword. CareerPilot finds fintech startups **and** established financial firms, qualifies them against your
profile, finds the right people and their verified email addresses, and only then checks whether a relevant opening is
listed. If one is, the email mentions that real opening. If not, the email asks whether they have current or upcoming
internships, and never pretends a job exists. Every email lands in Gmail as a draft for you to review.

### How a campaign flows

```
"Find 200 fintech companies in India"          fintech = company preference, not a job keyword
        │
        ▼
 1. Company discovery ── YC directory, Wellfound, "top / largest fintech" lists, the LLM's knowledge
        │                startups (payments, lending, neobanks, wealth, insurtech, broking) AND
        │                established banks / financial-services firms with technology teams
        ▼
 2. Qualification ────── research: sector, sub-sectors, funding stage, headcount, stack, news
        │                fit score from target_profile; off-target companies are rejected
        ▼
 3. Contact discovery ── all relevant people are found, classified and ranked
        │                (campus/tech recruiters, hiring managers, EMs, product/data/QA leads, founders)
        ▼
 4. Email discovery ──── published address → company pattern → Hunter/Apollo → SMTP verification
        ▼
 5. Opening check ────── (optional) the company's careers page / ATS board: is a matching role listed?
        │          │
       yes         no
        ▼          ▼
  job-specific    company-level internship inquiry
  email           (never claims an opening exists)
        └────┬─────┘
             ▼
 6. Resume + email ───── your resume attached; Groq writes; fact, grounding and style checks
             ▼
 7. Gmail draft ──────── you review and send (or auto-send inside a send window)
             ▼
 8. Follow-ups, reply/bounce tracking, funnel
```

Job discovery is an **enrichment step (5), not the entry point**. A company is contacted because it fits your target,
whether or not it advertises a role, and no job search is spent on a company until a reachable person has been found there.

### Step 1: Install and connect the services (one time)

| What | How |
| :--- | :--- |
| Python dependencies | `make install` (or `uv sync`), then `cp config.example.yaml config.yaml` |
| Local model (discovery, research, contacts, scoring) | Install [Ollama](https://ollama.com), run `ollama pull qwen3:8b` (or the model set in `llm.local_model`; `qwen2.5:7b` needs less memory) and keep `ollama serve` running |
| Email writing (Groq) | Set `GROQ_API_KEY` (or `llm.api_key`) |
| Gmail | Google Cloud Console → enable the **Gmail API** → create an OAuth client ID of type **Desktop app** → download it as `credentials.json` into the project folder → `recruiting-platform auth`. This creates `token.json` (scopes: compose + read-only) |
| Web search (recommended for 200 companies) | `api_keys.serper` or `api_keys.brave`. Without a key, DuckDuckGo/Yahoo are used and they throttle bursts |
| Optional enrichment | `api_keys.hunter` (emails and verification), `api_keys.apollo` (people, funding, headcount), `api_keys.github` |

Then run `recruiting-platform doctor`. It checks the Groq key and models, the Ollama model and routing table,
Playwright, the Gmail scopes, outbound SMTP port 25 (used to verify addresses) and Typst.

### Step 2: Configure `config.yaml`

The settings that matter for this campaign (every option is documented in `config.example.yaml`):

```yaml
user_identity:                 # used in the signature
  name: "Your Name"
  email: "you@example.com"
  linkedin_url: "https://www.linkedin.com/in/you/"
  github_url: "https://github.com/you"

job_preferences:
  roles:                       # ORDER = preference: decides the areas the email mentions, who is contacted
    - "Product Analyst Intern" # first, and which listed openings count as a match
    - "Data Analyst Intern"
    - "QA Engineer Intern"
    - "Software Engineer Intern"
  geographies: ["India", "Remote"]
  company_size: {min_employees: 20, max_employees: 100000}   # soft: outside the range only lowers the fit score
  experience_years_max: 1.0

target_profile:                # which companies qualify
  sector_weights: {fintech: 1.0, trading: 0.9, insurtech: 0.8, generic: 0.4}
  allowed_sectors: ["fintech", "insurtech", "trading"]       # hard filter after research
  funding_stages: []           # [] = startups AND established / listed companies
  skills: ["python", "sql", "fastapi"]                       # stack overlap raises the fit score
  min_company_fit: 0.45

discovery:
  mode: "company_outreach"
  sources: ["yc", "web_search", "wellfound", "llm", "ats_boards"]            # where companies come from
  job_sources: ["ats", "career_page", "wellfound", "indeed", "web_search"]   # the opening check uses ats + career_page
  allow_linkedin: false
  companies_per_run: 20        # companies discovered per campaign run

company_outreach:
  ask_about_openings: true     # the email asks about current/upcoming internships
  check_openings: true         # step 5; false = always send the company-level inquiry

contacts:
  personas: ["recruiter", "talent_acquisition", "hiring_manager", "engineering_manager", "head_of_engineering",
             "product_lead", "product_manager", "data_lead", "qa_lead", "founder"]   # who you are willing to email
  persona_strategy: "auto"     # size- and role-aware ranking (see step 5)
  max_contacts_per_company: 1  # 2 = a second thread to a different persona (e.g. recruiter + product lead)
  allow_generic_inbox: true    # fall back to the careers/hiring inbox when no person is found

email_verification:
  smtp_check: true
  allow_unverified: true       # keep the best-evidence address when port 25 is blocked

pipeline:
  resume_mode: "attach_base"   # attach your resume unchanged
  base_resume_path: "resumes/Your_Resume.pdf"
  daily_draft_limit: 15        # emails written per day

highlights:                    # experiences the email may use (one is picked per email, most relevant first)
  - name: "Payments analytics project"
    summary: "Built a dashboard that tracks refund and payment-failure rates"
    tags: ["data", "product", "fintech"]

outreach:
  auto_send: false             # drafts only; you send from Gmail
  daily_send_limit: 15
  send_window: {timezone: "Asia/Kolkata", start_hour: 9, end_hour: 12, weekdays_only: true}
  followups: [{after_days: 4}, {after_days: 7}]
```

Tips:
- **Put your strongest role family first.** Product first means product leads and PMs are preferred and the email
  talks about product/data internships; engineering first means engineering managers and engineering internships.
- **"fintech" covers the whole family.** A campaign goal that names fintech automatically accepts fintech,
  insurtech and trading/brokerage companies, and discovery asks for payments, lending, neobanks, wealth, insurance,
  broking and established banks.
- **Large banks are welcome.** Headcount above `max_employees` only lowers the fit score slightly. Raise the cap if
  you want them ranked higher. `exclusions.companies` removes companies you never want to contact.

### Step 3: Start with a small, review-only batch

```bash
recruiting-platform campaign "Find 200 fintech companies in India" --batch 20 --max-stage 5 --drafts-only
```

- `200` becomes the campaign target (qualified companies). `--batch 20` discovers 20 companies in this run.
- `--max-stage 5` stops after email discovery: companies are found, researched and qualified, their contacts are
  ranked and email addresses verified, but no email is written yet.
- `--drafts-only` keeps this campaign in draft mode even if `outreach.auto_send` is true.
- The mode comes from `discovery.mode`. Without it, the goal's wording decides (see [Discovery Modes](#-discovery-modes)),
  and `--mode company_outreach` forces it.

Check what it found before any email is written:

```bash
recruiting-platform campaigns                                # progress, mode and target of each campaign
recruiting-platform companies --sector fintech --sort fit    # sector, funding, size, fit, reply probability, status
recruiting-platform companies --status rejected              # companies that did not qualify
recruiting-platform ui                                       # dashboard: companies, all ranked contacts, applications
recruiting-platform explain <app_id>                         # why this contact and email address were chosen
```

If the list looks wrong, adjust `target_profile`, `job_preferences.roles` or `contacts.personas` now. Rejection
reasons are in the run log and in each application's history on the dashboard.

### Step 4: Write the drafts

```bash
recruiting-platform continue-campaign 1
```

This continues every application of campaign #1: the opening check, scoring, resume, email writing, validation and
the Gmail draft. It also discovers the next batch of companies. The log shows which path each company took:

```
[DISCOVERY] Mode: company_outreach
[DISCOVERY] Company-level outreach enabled; public job opening not required.
[OUTREACH] Company-level inquiry: preferring product_lead, product_manager, recruiter, ...
[OUTREACH] Checking the careers page / ATS board for a matching opening...
[OUTREACH] Matching opening found -> job-specific outreach (Product Analyst Intern)
[OUTREACH] No matching public opening found for QuietLend
[OUTREACH] Keeping the company-level internship inquiry
```

Each draft is saved in Gmail with your resume attached, and follow-ups #1 and #2 are prepared for the same thread.
At most `pipeline.daily_draft_limit` emails are written per day. Beyond that, companies are still researched and
their contacts found, and their emails are written on the next run.

### Step 5: Who receives the email

For every company, CareerPilot finds **all relevant contacts**, not every employee. Up to the ten best are stored,
classified by persona and ranked. The best one (or two, with `max_contacts_per_company: 2`) are emailed:

| Company | Contacts typically found | Usually chosen first |
| :--- | :--- | :--- |
| Early-stage startup (≤ 50 people, pre-seed/seed) | founder, CTO, engineering or product lead, recruiter | founder |
| Mid-size fintech | recruiter, engineering manager, product lead/PM, data lead, QA lead | the lead of your first role family, then a recruiter |
| Large firm or bank (> 1000 people) | campus / university recruiting, technology recruiter, engineering manager, data or product hiring manager | recruiter (campus / early careers), then the hiring manager for your first role family |

Ranking = persona fit (company size + your target role families) × source confidence × reachability (a known email)
× learned reply rates. Only personas listed in `contacts.personas` are preferred, and `persona_strategy: ordered`
follows your list strictly. Sources are team/leadership pages, engineering and product blogs, press and conference
talks, GitHub, The Org, Crunchbase and Wellfound snippets, plus Hunter/Apollo with keys. A person is kept only if
their name appears in the fetched text. LinkedIn is never searched or scraped. If no person is found, the company's
careers/hiring inbox is used (`allow_generic_inbox`).

Every address gets a confidence level: `verified` (the mail server accepted it), `high_confidence`, `pattern_match`
(company pattern inferred from real addresses), `catch_all` or `guessed`. A bounce marks the address invalid and the
next-best address is tried automatically.

### Step 6: The two kinds of email

| | Job-specific | Company-level inquiry |
| :--- | :--- | :--- |
| When | The opening check found a listed role that matches your roles | No matching public opening |
| Opens with | One verified observation (product, news, engineering post) or the posting itself | One verified observation about the company |
| Middle | Who you are + the one experience most relevant to that role | Who you are + one experience; your target areas mentioned naturally |
| Ask | About that role and how to be considered | Whether there are current or upcoming internship opportunities |
| Length | `outreach.min_words`–`max_words` | About 100–150 words |

For example, a job-specific email may say *"I saw the software engineering internship listed on your careers
page…"*, while a company-level inquiry says something like *"I'm writing to ask whether your technology teams have
any current or upcoming internship opportunities…"*. These show the meaning only: every email is written fresh by
Groq from the research and your resume.

Before a draft is created, validation checks that:
- no opening, job title or hiring plan is invented;
- every claim about you appears in your resume or highlights;
- every claim about the company appears in the research;
- stock phrases, jargon lists and copies of recent emails are rewritten.

A failed check sends the email back for one rewrite. A draft that still fails the fact or grounding checks is not
created. Style issues left after the rewrite are logged but don't block the draft.

### Step 7: Review, send and follow up

- Open **Gmail → Drafts**, edit if you like, and send. Drafts you send yourself are detected on the next `outreach`
  run and start the follow-up clock. Deleting a draft cancels its follow-ups.
- Follow-ups #1 (after 4 days) and #2 (after 7) are created as threaded replies when due. With auto-send they are
  sent; otherwise they wait in Drafts. They stop as soon as the person replies.
- `recruiting-platform outreach` checks replies and bounces, sends due emails (auto-send only) and prepares follow-ups.
- Replies are classified (interview / positive / referral / not interested / out of office). "Not interested" marks
  the contact do-not-contact. `recruiting-platform mark <app_id> interview` records outcomes you learn elsewhere.
- Auto-send (`outreach.auto_send: true` or `campaign --auto-send`) sends inside `send_window`, at most
  `daily_send_limit` per day and `min_minutes_between_sends` apart. Start with drafts until you trust the output.

### Step 8: Run it daily until 200

```bash
recruiting-platform daily              # outreach cycle → continue active campaigns → outreach cycle
recruiting-platform schedule install   # the daily run at 09:00 + an outreach cycle every 30 min (Windows/Linux)
recruiting-platform daemon             # the same in a foreground loop
```

Every daily run continues each active campaign with `discovery.companies_per_run` more companies until 200 qualified
companies are reached. The campaign completes when its applications are finished. The practical limits are:
- **Drafts per day:** `pipeline.daily_draft_limit`.
- **Groq tokens:** an email costs a few thousand tokens (more when a rewrite is needed), and
  `llm.groq_reserved_for_emails` is the daily budget for writing. When it runs out, emails are deferred to the next run
  and are never replaced by a template. `recruiting-platform usage` shows the real cost.
- **Search engines:** keyless search is throttled. A Serper or Brave key is the biggest speed-up for discovery.

At about 15 emails a day, 200 companies take roughly two weeks of daily runs.

### Step 9: Track results

```bash
recruiting-platform funnel --campaign 1
#  Discovered → Qualified → Researching → Contact Found → Email Found → Draft Created → Sent → Replied → Interview
recruiting-platform insights          # which sectors, personas and company sizes get replies
recruiting-platform explain <app_id>  # reply-probability breakdown, contact source, email evidence
```

The learned reply rates feed back into company and contact ranking automatically.

### Adding specific companies

```bash
recruiting-platform targeted "Goldman Sachs"
recruiting-platform targeted "razorpay.com" --jd "<pasted job description>"
```

A company you name is never rejected for fit. If it has no matching opening, it gets a company-level inquiry.

### Troubleshooting

| Symptom | What to do |
| :--- | :--- |
| Few companies found | Add a Serper/Brave key. The log prints `Discovery source '<name>' returned N candidates` for each source |
| Many companies rejected | `companies --status rejected`; loosen `allowed_sectors`, `min_company_fit` or `company_size` |
| "No deliverable professional email" | Port 25 may be blocked (`doctor`). Keep `allow_unverified: true`; a Hunter key helps |
| Applications stuck in "Email Generation" | Groq budget or rate limit (see `usage`). They resume automatically on the next run |
| Local-model errors | Make sure `ollama serve` is running and `llm.local_model` is pulled; run `doctor` |
| Wrong kind of recipient | Reorder `job_preferences.roles` (the first family wins), narrow `contacts.personas`, or start the campaign with `--personas recruiter,product_lead` |
| Worried about double emails | Not possible: one thread per company, `company_cooldown_days`, the outreach ledger and a Gmail Sent-folder check |

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

Titles are classified into **recruiter** (including campus / university / early-careers recruiting) **/ hiring_manager /
engineering_manager / vp_engineering / product_lead / product_manager / data_lead / qa_lead / founder / cto / tech_lead**.
At companies above 1000 people, campus and early-careers recruiters are searched for as well.
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
recruiting-platform campaign "Find 200 fintech companies in India" --mode company_outreach
#  → discovers & qualifies companies → finds and ranks contacts → verifies emails → checks for a matching opening
#  → job-specific email or company-level inquiry → Gmail drafts (+2 follow-ups)
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
| `company_outreach` | Company first: discover → qualify → contacts → verified email, **then** check the company's careers page / ATS board. A matching listed opening is mentioned; otherwise a **company-level inquiry** asks whether they have current or upcoming internships. A public opening is never required. | Campaigns like "Find 200 fintech companies in India" |
| `hybrid` | Full job search first. Matching opening → job-specific email; no matching opening → company-level inquiry instead of dropping a relevant company. | Usually the most useful |

### Job Search
Find and target actual public job openings that match your configured roles (ATS boards, careers pages, Wellfound,
Indeed and web search). If no posting matches, the company is skipped: nothing is created for it.

### Company Outreach
Find relevant companies even when there is no public opening, and ask about current or upcoming opportunities.
Company discovery is independent of jobs: every qualified company is researched and its contacts and email addresses
are found first. Only then does the **opening check** read the company's own careers page / ATS board (no job-board
searches). If a matching opening is listed, the thread becomes job-specific and the email refers to that real opening.
Otherwise it stays a company-level inquiry. Set `company_outreach.check_openings: false` to skip the check and always
send the inquiry. See the [step-by-step guide](#-step-by-step-guide-find-200-fintech-companies-in-india).

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
| Large company (> 1000 people) | recruiter first (campus / early-careers recruiting) |

With `persona_strategy: ordered` your explicit persona order is kept as-is.

#### Configuration examples
```yaml
# Job search only (the default; same as leaving `mode` out)
discovery:
  mode: "job_search"
```
```yaml
# Company outreach: company first, openings checked after contacts are found, no public opening required
discovery:
  mode: "company_outreach"
company_outreach:
  ask_about_openings: true
  check_openings: true       # false = always send the company-level inquiry
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
[DISCOVERY] Mode: company_outreach
[DISCOVERY] Company-level outreach enabled; public job opening not required.
[DISCOVERY] Openings are checked after contacts and emails are found
[OUTREACH] Creating company-level speculative outreach
[OUTREACH] Checking the careers page / ATS board for a matching opening...
[OUTREACH] Keeping the company-level internship inquiry
```
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
