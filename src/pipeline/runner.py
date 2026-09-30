import contextlib
import sys
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Session

from src.config import AppConfig, load_config
from src.db.models import Application, Campaign, Company, Contact, Email, History, Job, Run
from src.db.session import get_session_factory, init_db
from src.intel.classify import expand_sectors
from src.outreach.dedupe import backfill_ledger
from src.outreach.engine import OutreachEngine
from src.pipeline.campaign import (
    campaign_progress,
    campaign_spec,
    create_campaign,
    parse_goal,
    qualified_company_count,
    resolve_mode,
)

# Import stage functions
from src.pipeline.stages import (
    TERMINAL_STATES,
    company_outreach_target,
    research_company,
    run_opening_check,
    run_stage_0_company_discovery,
    run_stage_1_job_discovery,
    run_stage_2_filtering,
    run_stage_3_company_research,
    run_stage_4_contact_research,
    run_stage_5_email_discovery,
    run_stage_6_opportunity_scoring,
    run_stage_7_resume_tailoring,
    run_stage_8_email_generation,
    run_stage_9_validation,
    run_stage_10_gmail_draft_creation,
    run_stage_11_database_finalization,
    spec_from_config,
)
from src.providers.browser import BrowserProvider
from src.providers.gmail import GmailProvider
from src.providers.llm import BaseLLMProvider
from src.providers.llm.router import LLMRouter, UsageTracker
from src.scheduler import manage_automation
from src.sources.companies import DiscoverySpec
from src.utils.caching import DBCache
from src.utils.logging import PipelineLogger, get_logger

logger = get_logger("recruiting-platform.pipeline.runner")


def generate_run_id(session: Session) -> str:
    """
    Generates a unique Run ID: RUN-YYYYMMDD-NNN.
    """
    today_str = datetime.now(UTC).strftime("%Y%m%d")
    prefix = f"RUN-{today_str}-"

    existing_runs = session.query(Run).filter(Run.id.like(f"{prefix}%")).order_by(Run.id.desc()).all()

    if not existing_runs:
        return f"{prefix}001"

    last_run_id = existing_runs[0].id
    try:
        suffix = int(last_run_id.split("-")[-1])
        new_suffix = f"{suffix + 1:03d}"
    except ValueError:
        new_suffix = "001"

    return f"{prefix}{new_suffix}"


class PipelineRunner:
    """
    Orchestrates the entire recruiting pipeline.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: AppConfig = load_config(config_path)
        init_db(self.config.pipeline.db_path)
        self.SessionLocal = get_session_factory(self.config.pipeline.db_path)

        # Initialize providers
        self.usage = UsageTracker(lambda: self.SessionLocal())
        self.router = LLMRouter(self.config.llm, self.usage)
        self.llm: BaseLLMProvider = self.router.premium
        self._initial_llm = self.llm
        self._llm_fast: BaseLLMProvider | None = None
        self.browser = BrowserProvider()
        self.browser.configure_search(
            provider=self.config.search.provider,
            serper_key=self.config.api_keys.get("serper"),
            brave_key=self.config.api_keys.get("brave"),
            min_interval_seconds=self.config.search.min_interval_seconds,
        )
        self.gmail = GmailProvider(
            credentials_path=self.config.gmail.credentials_file,
            token_path=self.config.gmail.token_file,
            scopes=self.config.gmail.scopes,
        )

        # Duplicate-prevention ledger must cover drafts created before it existed.
        with contextlib.suppress(Exception):
            session = self.SessionLocal()
            try:
                backfill_ledger(session)
            finally:
                session.close()

        # Failsafe: Self-heals background automation if automation=true, or disables it if automation=false
        manage_automation(self.config.pipeline.automation)

    def llm_for(self, task: str) -> BaseLLMProvider:
        """
        LLM for a pipeline task, routed by `llm.routing` (local Ollama vs premium Groq) with token accounting.
        If `llm` was replaced (tests, custom providers), that provider is used for everything.
        """
        if self.llm is not self._initial_llm:
            return self.llm
        return self.router.get(task)

    @property
    def llm_fast(self) -> BaseLLMProvider:
        """Backwards-compatible alias: the provider used for cheap extraction work."""
        if self._llm_fast is not None:
            return self._llm_fast
        return self.llm_for("default")

    @llm_fast.setter
    def llm_fast(self, provider: BaseLLMProvider) -> None:
        self._llm_fast = provider

    def get_todays_draft_count(self, session: Session) -> int:
        """
        Count initial draft emails successfully created in Gmail today.
        """
        today_start = datetime.combine(date.today(), datetime.min.time())
        return (
            session.query(Email)
            .filter(
                Email.gmail_draft_id.isnot(None),
                Email.status.in_(["draft_created", "scheduled", "sent"]),
                Email.created_at >= today_start,
                or_(Email.sequence_step == 0, Email.sequence_step.is_(None)),
            )
            .count()
        )

    def _new_run(self, session: Session, label: str) -> tuple[Run, PipelineLogger]:
        run_id = generate_run_id(session)
        p_log = PipelineLogger(logger, run_id, label)
        new_run = Run(id=run_id, status="running")
        session.add(new_run)
        session.commit()
        self.usage.run_id = run_id
        self._startup_checks(p_log)
        return new_run, p_log

    def _startup_checks(self, p_log: PipelineLogger) -> None:
        """Surface actionable problems once per process instead of failing silently later."""
        if getattr(self, "_startup_checked", False):
            return
        self._startup_checked = True
        playwright_status = getattr(self.browser, "playwright_status", None)
        if callable(playwright_status):
            ready, message = playwright_status()
            if not ready:
                p_log.warning(message)
        if self.llm is self._initial_llm and self.router.local is not None:
            if self.router.local_available():
                p_log.info(f"Local LLM: {self.router.local_status}")
            else:
                p_log.warning(
                    f"Local LLM unavailable: {self.router.local_status}. Local tasks will "
                    + ("use the Groq fast model within the non-email budget." if self.config.llm.local_fallback == "premium_fast" else "fail.")
                )

    def _check_gmail(self, p_log: PipelineLogger) -> None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
        p_log.info("Checking Gmail API credentials...")
        if not self.gmail.authenticate(interactive=interactive):
            p_log.warning(
                "Gmail API authentication failed. Draft creation stages will be skipped/paused. "
                "Run 'python -m src.cli auth' in terminal to authorize Gmail."
            )

    @staticmethod
    def _priority(app: Application) -> float:
        company = app.job.company
        fit = company.fit_score if company.fit_score is not None else 0.5
        probability = company.response_probability if company.response_probability is not None else 0.08
        return fit * (0.5 + probability * 5)

    def _process_apps(
        self, session: Session, apps: list[Application], run_id: str, max_stage: int, limit_drafts: int | None
    ) -> None:
        p_log = PipelineLogger(logger, run_id, "Pipeline Processing")
        max_drafts = limit_drafts or self.config.pipeline.daily_draft_limit
        for app in sorted(apps, key=self._priority, reverse=True):
            limit_reached = self.get_todays_draft_count(session) >= max_drafts
            if limit_reached and app.current_stage >= 8:
                p_log.warning(
                    f"Daily draft limit of {max_drafts} reached. "
                    f"Skipping further draft creation today. Application #{app.id} paused.",
                    status="PAUSED",
                )
                continue
            # Once the limit is hit, keep researching/finding contacts but stop before writing emails.
            self._process_application(session, app, run_id, min(max_stage, 7) if limit_reached else max_stage)
            self.usage.flush()

    def _discover(
        self,
        session: Session,
        run_id: str,
        spec: DiscoverySpec | None = None,
        campaign_id: int | None = None,
        limit: int | None = None,
    ) -> list[Company]:
        """Stage 0 using the explicit spec, or each standing `discovery.queries` entry, or your target profile."""
        if spec is not None:
            return run_stage_0_company_discovery(
                session, self.config, self.llm_for("discovery"), self.browser, run_id, spec=spec, campaign_id=campaign_id, limit=limit
            )
        want = limit or self.config.discovery.companies_per_run
        if not self.config.discovery.queries:
            return run_stage_0_company_discovery(
                session, self.config, self.llm_for("discovery"), self.browser, run_id, spec=spec_from_config(self.config), limit=want
            )
        companies: list[Company] = []
        cache = DBCache(session)
        for query in self.config.discovery.queries:
            if len(companies) >= want:
                break
            cache_key = f"parsed_spec_{query.lower()[:150]}"
            cached = cache.get(cache_key)
            if isinstance(cached, dict):
                query_spec = DiscoverySpec.from_dict(cached)
            else:
                query_spec = parse_goal(self.llm_for("campaign_parsing"), query, default_count=want)
                cache.set(cache_key, query_spec.to_dict(), 30 * 86400)
            companies.extend(
                run_stage_0_company_discovery(
                    session, self.config, self.llm_for("discovery"), self.browser, run_id, spec=query_spec, limit=want - len(companies)
                )
            )
        return companies

    def run(
        self,
        resume_only: bool = False,
        max_stage: int = 12,
        limit_drafts: int | None = None,
    ) -> str:
        """
        Executes the pipeline runner.
        - Checks for interrupted/paused applications and resumes them.
        - If none (and resume_only is False), discovers new companies/jobs and runs them.
        - Enforces daily draft limits.
        """
        session = self.SessionLocal()
        new_run, p_log = self._new_run(session, "Pipeline Run Start")
        run_id = new_run.id
        p_log.info(f"Initializing pipeline run: {run_id} (Up to Stage {max_stage})")

        if max_stage >= 10:
            self._check_gmail(p_log)

        try:
            active_apps = session.query(Application).filter(Application.state.notin_(TERMINAL_STATES)).all()

            if active_apps:
                p_log.info(f"Resuming {len(active_apps)} interrupted applications...")
            elif resume_only:
                p_log.info("No active applications to resume. (resume_only = True)")
                new_run.status = "completed"
                new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
                session.commit()
                return run_id
            else:
                p_log.info("No interrupted applications found. Running fresh discovery...")
                companies = self._discover(session, run_id)
                jobs = run_stage_1_job_discovery(session, self.config, self.llm_for("job_extraction"), self.browser, companies, run_id)
                active_apps = run_stage_2_filtering(session, self.config, jobs, run_id)

            self._process_apps(session, active_apps, run_id, max_stage, limit_drafts)

            new_run.status = "completed"
            new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            p_log.info("Pipeline run execution finished.", status="COMPLETED")

        except Exception as e:
            new_run.status = "failed"
            new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            p_log.error(f"Pipeline execution crashed: {e}", status="CRASHED")
            raise e
        finally:
            session.close()
            self.usage.flush()

        return run_id

    def retry_failed(self) -> str:
        """
        Resets failed applications back to their preceding active stage,
        then executes the pipeline runner to retry them.
        """
        session = self.SessionLocal()
        failed_apps = (
            session.query(Application)
            .filter(Application.state.in_(["Failed", "Research Failed", "Draft Failed", "Validation Failed"]))
            .all()
        )

        if not failed_apps:
            logger.info("No failed applications found to retry.")
            session.close()
            return self.run(resume_only=True)

        run_id = generate_run_id(session)
        new_run = Run(id=run_id, status="running")
        session.add(new_run)
        session.flush()

        p_log = PipelineLogger(logger, run_id, "Pipeline Retry")
        p_log.info("Resetting failed applications for retry...")

        for app in failed_apps:
            old_state = app.state
            if old_state == "Research Failed":
                app.current_stage = 3
                app.state = "Company Research"
            elif old_state == "Draft Failed":
                app.current_stage = 10
                app.state = "Gmail Draft Creation"
            elif old_state == "Validation Failed":
                app.current_stage = 7
                app.state = "Resume Tailoring"
            else:
                # Default fallback: resume tailoring
                app.current_stage = 7
                app.state = "Resume Tailoring"

            session.add(
                History(
                    application_id=app.id,
                    stage=app.current_stage,
                    state=app.state,
                    run_id=run_id,
                    notes=f"Reset state from '{old_state}' to retry processing.",
                )
            )
            p_log.info(f"Reset Application #{app.id} state to '{app.state}' for retry.")

        session.commit()
        session.close()

        # Run normal pipeline to process the reset applications
        return self.run(resume_only=True)

    def _process_application(self, session: Session, app: Application, run_id: str, max_stage: int = 12) -> None:
        """
        Drives a single Application forward through the stages, up to max_stage.
        """
        company = app.job.company
        p_log = PipelineLogger(logger, run_id, f"App #{app.id} processing", company.name)

        try:
            # Stage 3: Company Research
            if app.current_stage == 3 and max_stage >= 3:
                if not run_stage_3_company_research(session, self.config, self.llm_for("research"), self.browser, app, run_id):
                    return

            # Stage 4: Contact Research
            if app.current_stage == 4 and max_stage >= 4:
                if not run_stage_4_contact_research(session, self.config, self.llm_for("contact_finding"), self.browser, app, run_id):
                    return

            # Stage 5: Professional Email Discovery
            if app.current_stage == 5 and max_stage >= 5:
                if not run_stage_5_email_discovery(session, self.config, self.llm_for("email_finding"), self.browser, app, run_id):
                    return

            # Stage 6: Opportunity Scoring. A company-level inquiry first checks for a matching opening, now that the
            # company is qualified and a contact with an email exists (job discovery is enrichment, not the entry point).
            if app.current_stage == 6 and max_stage >= 6:
                run_opening_check(session, self.config, self.llm_for("job_extraction"), self.browser, app, run_id)
                if not run_stage_6_opportunity_scoring(session, self.config, self.llm_for("scoring"), app, run_id):
                    return

            # Stage 7: Resume Tailoring
            if app.current_stage == 7 and max_stage >= 7:
                if not run_stage_7_resume_tailoring(session, self.config, self.llm_for("resume_tailoring"), app, run_id):
                    return

            # Stage 8: Email Generation
            # Stages 8-9: generate & validate; a failed validation may send the email back for one rewrite.
            for attempt in range(2):
                if app.current_stage == 8 and max_stage >= 8:
                    writer = self.llm_for("email_regeneration" if attempt else "email_generation")
                    if not run_stage_8_email_generation(
                        session,
                        self.config,
                        writer,
                        app,
                        run_id,
                        followup_llm=self.llm_for("followup_generation"),
                        helper_llm=self.llm_for("summarization"),
                    ):
                        return
                if app.current_stage == 9 and max_stage >= 9:
                    if not run_stage_9_validation(session, self.config, self.llm_for("validation"), app, run_id):
                        if app.current_stage == 8:
                            continue  # rewrite requested with the validator's findings
                        return
                break

            # Stage 10: Gmail Draft Creation
            if app.current_stage == 10 and max_stage >= 10:
                if not run_stage_10_gmail_draft_creation(session, self.gmail, app, run_id, self.config):
                    return

            # Stage 11: Database Finalization
            if app.current_stage == 11 and max_stage >= 11:
                if not run_stage_11_database_finalization(session, app, run_id):
                    return

        except Exception as e:
            p_log.error(f"Error processing application {app.id}: {e}")
            with contextlib.suppress(Exception):
                session.rollback()
            try:
                app.state = "Failed"
                session.add(
                    History(
                        application_id=app.id,
                        stage=app.current_stage,
                        state="Failed",
                        run_id=run_id,
                        notes=f"Exception encountered during processing: {e}",
                    )
                )
                session.commit()
            except Exception as rollback_err:
                with contextlib.suppress(Exception):
                    session.rollback()
                logger.error(f"Failed to save failure state for application {app.id}: {rollback_err}")

    def run_targeted(self, target_input: str, jd: str | None = None, max_stage: int = 12) -> str:
        """
        Executes a targeted search, research, and outreach for a specific company or contact.
        """
        session = self.SessionLocal()
        new_run, p_log = self._new_run(session, "Targeted Outreach Start")
        run_id = new_run.id
        p_log.info(f"Initializing targeted outreach run: {run_id} for input: '{target_input}'")

        if max_stage >= 10:
            self._check_gmail(p_log)

        try:
            # 1. Parse target input using LLM
            prompt = (
                f"Analyze the following targeted outreach input string:\n"
                f"'{target_input}'\n\n"
                f"This input may contain a company name, a domain name, an email address, or a combination. "
                f"Extract and return the cleaned canonical company name, the domain, "
                f"the contact email (if explicitly provided), and the contact name (if deducible)."
            )

            from pydantic import BaseModel, Field

            class TargetedParseResponse(BaseModel):
                company_name: str = Field(description="Canonical cleaned company name")
                domain: str | None = Field(description="Company website domain name or null")
                contact_email: str | None = Field(description="Direct contact email address or null")
                contact_name: str | None = Field(description="Deducible contact name or null")

            parsed_target: TargetedParseResponse = self.llm_for("targeted_parsing").generate_json(prompt, TargetedParseResponse)  # type: ignore

            p_log.info(
                f"Parsed target: Company='{parsed_target.company_name}', Domain='{parsed_target.domain}', "
                f"Email='{parsed_target.contact_email}', Name='{parsed_target.contact_name}'"
            )

            # 2. Get or create Company
            company = session.query(Company).filter(Company.name.ilike(parsed_target.company_name)).first()
            if not company and parsed_target.domain:
                company = session.query(Company).filter(Company.domain == parsed_target.domain.lower()).first()

            if not company:
                p_log.info(f"Company '{parsed_target.company_name}' not found in DB. Creating new company entry.")
                company = Company(
                    name=parsed_target.company_name,
                    domain=parsed_target.domain.lower() if parsed_target.domain else None,
                    source="targeted",
                )
                session.add(company)
                session.flush()

            # Explicit targets bypass fit-based rejection.
            extra = dict(company.extra_data) if isinstance(company.extra_data, dict) else {}
            extra["targeted"] = True
            company.extra_data = extra
            company.status = "target"

            # Temporarily clear exclusions for this targeted company name so it isn't filtered out
            original_exclusions = list(self.config.exclusions.companies)
            self.config.exclusions.companies = [c for c in original_exclusions if c.lower() not in company.name.lower()]

            # 3. Discover jobs specifically for this company (Stage 1 logic) or use pasted JD
            if jd:
                p_log.info("Pasted Job Description provided. Parsing Job details using LLM...")
                import hashlib

                class JDParseResponse(BaseModel):
                    title: str = Field(description="Title of the job role, e.g. Software Engineer")
                    location: str | None = Field(description="Job location details or Remote")
                    salary: str | None = Field(description="Salary range or details")
                    experience_years: float | None = Field(description="Required experience years if mentioned")
                    description: str = Field(description="Summarized description of key requirements and role responsibilities")

                jd_prompt = (
                    f"Analyze the following pasted job description for a role at {company.name}:\n\n"
                    f"{jd}\n\n"
                    f"Extract the Job Title, Location, Salary details, Required Experience years (float, e.g. 2.5), "
                    f"and a summary of key requirements and responsibilities."
                )
                parsed_jd: JDParseResponse = self.llm_for("targeted_parsing").generate_json(jd_prompt, JDParseResponse)  # type: ignore

                jd_hash = hashlib.md5(jd.encode("utf-8")).hexdigest()[:8]
                pasted_url = f"pasted://{company.name.lower().replace(' ', '_')}_{jd_hash}"

                existing_job = session.query(Job).filter(Job.url == pasted_url).first()
                if not existing_job:
                    p_log.info(f"Creating new Job entry from pasted JD: '{parsed_jd.title}'")
                    new_job = Job(
                        company_id=company.id,
                        title=parsed_jd.title,
                        url=pasted_url,
                        location=parsed_jd.location or (self.config.job_preferences.geographies[0] if self.config.job_preferences.geographies else "Remote"),
                        salary=parsed_jd.salary,
                        experience_years_required=parsed_jd.experience_years,
                        description=parsed_jd.description,
                        source="pasted",
                    )
                    session.add(new_job)
                    session.flush()
                    jobs = [new_job]
                else:
                    p_log.info("Job from pasted JD already exists in database.")
                    jobs = [existing_job]
            else:
                jobs = run_stage_1_job_discovery(session, self.config, self.llm_for("job_extraction"), self.browser, [company], run_id)

                # 4. No matching opening: you named this company, so ask it about internships at company level
                #    (in any mode). No job title is invented.
                if not jobs:
                    p_log.info(f"[OUTREACH] No matching public opening found for {company.name}")
                    p_log.info("[OUTREACH] Targeted company -> creating company-level speculative outreach")
                    target = company_outreach_target(self.config, company)
                    existing_job = session.query(Job).filter(Job.url == target["url"]).first()
                    if not existing_job:
                        existing_job = Job(
                            company_id=company.id,
                            title=target["title"],
                            url=target["url"],
                            description=target["description"],
                            source=target["source"],
                        )
                        session.add(existing_job)
                        session.flush()
                    jobs = [existing_job]

            # Restore exclusions
            self.config.exclusions.companies = original_exclusions

            # 5. Initialize application & filtering (Stage 2 logic)
            active_apps = run_stage_2_filtering(session, self.config, jobs, run_id)

            # 6. If contact email was provided in the input, associate it with the active applications
            if parsed_target.contact_email:
                contact_record = session.query(Contact).filter(Contact.email == parsed_target.contact_email.lower()).first()
                if not contact_record:
                    c_name = parsed_target.contact_name or parsed_target.contact_email.split("@")[0].title()
                    p_log.info(f"Creating new Contact for email: {parsed_target.contact_email}")
                    contact_record = Contact(
                        company_id=company.id,
                        name=c_name,
                        role="Decision Maker",
                        email=parsed_target.contact_email.lower(),
                        email_status="provided",
                        email_source="provided",
                        source="provided",
                    )
                    session.add(contact_record)
                    session.flush()

                for app in active_apps:
                    app.contact_id = contact_record.id
                    p_log.info(f"Linking contact {contact_record.name} ({contact_record.email}) to Application #{app.id}")
                    session.commit()

            # 7. Process these specific applications
            self._process_apps(session, active_apps, run_id, max_stage, None)

            new_run.status = "completed"
            new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            p_log.info("Targeted pipeline run execution finished.", status="COMPLETED")

        except Exception as e:
            new_run.status = "failed"
            new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            p_log.error(f"Targeted pipeline execution crashed: {e}", status="CRASHED")
            raise e
        finally:
            session.close()
            self.usage.flush()

        return run_id

    # ------------------------------------------------------------------
    # Campaigns, discovery-only runs, outreach & daily automation
    # ------------------------------------------------------------------

    def start_campaign(
        self,
        goal: str,
        target: int | None = None,
        personas: list[str] | None = None,
        auto_send: bool | None = None,
        batch_size: int | None = None,
        max_stage: int = 12,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Creates a campaign from a natural-language goal and runs its first batch. The discovery mode is fixed when the
        campaign is created: `mode` (CLI --mode) > discovery.mode set in config.yaml > mode implied by the goal > default.
        """
        session = self.SessionLocal()
        try:
            spec = parse_goal(self.llm_for("campaign_parsing"), goal, default_count=target or 25)
            if target:
                spec.count = target
            if personas:
                spec.personas = personas
            spec.mode, reason = resolve_mode(self.config, spec.mode, mode)
            campaign = create_campaign(session, goal, spec, target=target, personas=personas or spec.personas, auto_send=auto_send)
            campaign_id = campaign.id
            logger.info(f"Created campaign #{campaign_id}: {goal} -> {spec.describe()} (target {campaign.target_companies})")
            logger.info(f"[DISCOVERY] Campaign mode: {spec.mode} ({reason})")
        finally:
            session.close()
            self.usage.flush()
        return self.continue_campaign(campaign_id, batch_size=batch_size, max_stage=max_stage)

    def continue_campaign(self, campaign_id: int, batch_size: int | None = None, max_stage: int = 12) -> dict[str, Any]:
        """Discovers more companies until the campaign target is reached and advances its applications."""
        session = self.SessionLocal()
        new_run, p_log = self._new_run(session, f"Campaign #{campaign_id}")
        run_id = new_run.id
        original_personas = list(self.config.contacts.personas)
        original_allowed = list(self.config.target_profile.allowed_sectors)
        try:
            campaign = session.get(Campaign, campaign_id)
            if campaign is None:
                raise ValueError(f"Campaign #{campaign_id} not found")
            if campaign.personas:
                self.config.contacts.personas = [str(p) for p in campaign.personas]
            if max_stage >= 10:
                self._check_gmail(p_log)

            spec = campaign_spec(campaign)
            if spec.sectors:
                # A campaign for e.g. fintech rejects companies outside that sector family
                # (fintech also covers insurtech and trading/brokerage).
                self.config.target_profile.allowed_sectors = expand_sectors(spec.sectors)
            qualified = qualified_company_count(session, campaign_id)
            remaining = campaign.target_companies - qualified
            batch = batch_size or self.config.discovery.companies_per_run
            if remaining > 0 and campaign.status == "active":
                p_log.info(f"Campaign progress {qualified}/{campaign.target_companies}; discovering up to {min(batch, remaining)} more.")
                companies = self._discover(session, run_id, spec=spec, campaign_id=campaign_id, limit=min(batch, remaining))
                if not companies:
                    p_log.warning("No new companies found this run (sources may be exhausted or rate limited).")
                jobs = run_stage_1_job_discovery(
                    session, self.config, self.llm_for("job_extraction"), self.browser, companies, run_id,
                    mode=spec.mode or None,
                )
                run_stage_2_filtering(session, self.config, jobs, run_id, campaign_id=campaign_id)

            active = (
                session.query(Application)
                .filter(Application.campaign_id == campaign_id, Application.state.notin_(TERMINAL_STATES))
                .all()
            )
            self._process_apps(session, active, run_id, max_stage, None)

            still_active = (
                session.query(Application)
                .filter(Application.campaign_id == campaign_id, Application.state.notin_(TERMINAL_STATES))
                .count()
            )
            if qualified_company_count(session, campaign_id) >= campaign.target_companies and still_active == 0:
                campaign.status = "completed"
            new_run.status = "completed"
            new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            return campaign_progress(session, campaign)
        except Exception as e:
            new_run.status = "failed"
            new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            p_log.error(f"Campaign run crashed: {e}", status="CRASHED")
            raise
        finally:
            self.usage.flush()
            self.config.contacts.personas = original_personas
            self.config.target_profile.allowed_sectors = original_allowed
            session.close()

    def discover_companies(self, query: str, count: int = 25, research: bool = False) -> list[dict[str, Any]]:
        """Builds (and optionally researches, classifies and fit-scores) a target company list without outreach."""
        session = self.SessionLocal()
        new_run, p_log = self._new_run(session, "Company Discovery")
        try:
            spec = parse_goal(self.llm_for("campaign_parsing"), query, default_count=count)
            spec.count = count
            companies = self._discover(session, new_run.id, spec=spec, limit=count)
            rows = []
            for company in companies:
                if research:
                    try:
                        fit = research_company(session, self.config, self.llm_for("research"), self.browser, company, p_log)
                        company.status = "rejected" if fit.rejected else "target"
                        session.commit()
                    except Exception as e:
                        p_log.warning(f"Research failed for {company.name}: {e}")
                rows.append(
                    {
                        "id": company.id, "name": company.name, "domain": company.domain, "sector": company.sector,
                        "funding_stage": company.funding_stage, "employees": company.employee_count,
                        "hiring": company.hiring_status, "fit": company.fit_score, "status": company.status,
                        "source": company.source,
                    }
                )
            new_run.status = "completed"
            new_run.completed_at = datetime.now(UTC).replace(tzinfo=None)
            session.commit()
            return rows
        finally:
            session.close()
            self.usage.flush()

    def run_outreach_cycle(self) -> dict[str, int]:
        """Sync manual sends, detect replies/bounces, send due emails and follow-ups."""
        session = self.SessionLocal()
        try:
            if not self.gmail.authenticate(interactive=False):
                logger.warning("Gmail not authorized; outreach cycle skipped. Run `recruiting-platform auth`.")
                return {}
            engine = OutreachEngine(session, self.config, self.gmail, self.llm_for("reply_classification"))
            summary = engine.run_cycle()
            logger.info(f"Outreach cycle: {summary}")
            return summary
        finally:
            session.close()
            self.usage.flush()

    def run_daily(self, max_stage: int = 12) -> dict[str, Any]:
        """One full day of automation: outreach cycle, campaigns, standing discovery/resume, outreach cycle again."""
        results: dict[str, Any] = {"outreach_before": self.run_outreach_cycle()}
        session = self.SessionLocal()
        try:
            campaign_ids = [c.id for c in session.query(Campaign).filter(Campaign.status == "active").all()]
        finally:
            session.close()
            self.usage.flush()
        results["campaigns"] = []
        for campaign_id in campaign_ids:
            try:
                results["campaigns"].append(self.continue_campaign(campaign_id, max_stage=max_stage))
            except Exception as e:
                logger.error(f"Campaign #{campaign_id} failed during daily run: {e}")
        if not campaign_ids or self.config.discovery.queries:
            try:
                results["run_id"] = self.run(max_stage=max_stage)
            except Exception as e:
                logger.error(f"Daily pipeline run failed: {e}")
        results["outreach_after"] = self.run_outreach_cycle()
        return results
