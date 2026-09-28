import os

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.db.models import Application, Run
from src.db.session import get_session_factory
from src.db.session import init_db as db_init
from src.pipeline.runner import PipelineRunner

app = typer.Typer(help="AI-assisted Recruiting Platform CLI")
console = Console()


def get_runner(config_path: str = "config.yaml") -> PipelineRunner:
    """Helper to initialize PipelineRunner with error handling."""
    if not os.path.exists(config_path):
        console.print(f"[bold red]Error:[/bold red] Config file '{config_path}' not found. Please create one.")
        raise typer.Exit(code=1)
    try:
        return PipelineRunner(config_path)
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        raise typer.Exit(code=1) from e


def auto_start_widget(config_path: str = "config.yaml") -> None:
    """Auto-launches dark mode status widget server on port 18492 if not already running."""
    try:
        from src.config import load_config
        from src.web_server import DEFAULT_WIDGET_PORT, ensure_widget_server_running

        cfg = load_config(config_path)
        ensure_widget_server_running(db_path=cfg.pipeline.db_path, port=DEFAULT_WIDGET_PORT, auto_open=False)
    except Exception as e:
        console.print(f"[yellow]Widget auto-start warning: {e}[/yellow]")


@app.command("run")
def run_pipeline(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    limit: int | None = typer.Option(None, help="Override daily draft limit"),
    max_stage: int = typer.Option(12, "--max-stage", "--stage", help="Maximum stage to execute (0 to 12)"),
) -> None:
    """
    Run the entire recruiting pipeline end-to-end (discover, filter, research, tailor, draft).
    """
    auto_start_widget(config)
    console.print("[bold green]Starting Recruiting Platform End-to-End Pipeline...[/bold green]")
    runner = get_runner(config)
    try:
        run_id = runner.run(max_stage=max_stage, limit_drafts=limit)
        console.print(f"[bold green]Pipeline completed successfully for {run_id}![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Pipeline failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("targeted")
def run_targeted_outreach(
    target: str = typer.Argument(..., help="Company name, company email, domain, or related identifier"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    jd: str | None = typer.Option(None, help="Pasted Job Description text"),
) -> None:
    """
    Run targeted search, research, and outreach for a specific company or contact.
    """
    if not jd:
        import os
        jd = os.environ.get("JD")
        # Clean up empty strings or whitespace-only inputs
        if jd and not jd.strip():
            jd = None

    console.print(f"[bold green]Starting Targeted Outreach for: '{target}'...[/bold green]")
    runner = get_runner(config)
    try:
        run_id = runner.run_targeted(target, jd=jd, max_stage=12)
        console.print(f"[bold green]Targeted outreach completed successfully for {run_id}![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Targeted outreach failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("search")
def search_jobs(config: str = typer.Option("config.yaml", help="Path to config.yaml")) -> None:
    """
    Stage 0 - 2: Search for jobs, discover companies, and apply filtering criteria.
    """
    console.print("[bold cyan]Executing Job & Company Discovery (Stages 0-2)...[/bold cyan]")
    runner = get_runner(config)
    try:
        run_id = runner.run(max_stage=2)
        console.print(
            f"[bold green]Discovery completed for {run_id}. Applications initialized and filtered.[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Discovery failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("research")
def research_companies(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Stage 3 - 5: Gather research on filtered companies and discover contacts & email patterns.
    """
    console.print("[bold magenta]Executing Research and Contact Discovery (Stages 3-5)...[/bold magenta]")
    runner = get_runner(config)
    try:
        run_id = runner.run(resume_only=True, max_stage=5)
        console.print(f"[bold green]Research completed for {run_id}. Contacts and email profiles added.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Research failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("tailor")
def tailor_resumes(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Stage 6 - 7: Score opportunities and generate tailored Typst resumes.
    """
    console.print("[bold yellow]Executing Opportunity Scoring & Resume Tailoring (Stages 6-7)...[/bold yellow]")
    runner = get_runner(config)
    try:
        run_id = runner.run(resume_only=True, max_stage=7)
        console.print(
            f"[bold green]Resume tailoring completed for {run_id}. Output saved in resumes/generated.[/bold green]"
        )
    except Exception as e:
        console.print(f"[bold red]Resume tailoring failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("draft")
def create_drafts(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    limit: int | None = typer.Option(None, help="Override daily draft limit"),
) -> None:
    """
    Stage 8 - 10: Generate personalized emails, validate content, and save Gmail drafts.
    """
    console.print("[bold blue]Executing Email Generation & Gmail Draft Creation (Stages 8-10)...[/bold blue]")
    runner = get_runner(config)
    try:
        run_id = runner.run(resume_only=True, max_stage=10, limit_drafts=limit)
        console.print(f"[bold green]Draft generation completed for {run_id}. Gmail drafts populated.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Drafting failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("resume")
def resume_pipeline(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Resume processing all paused/interrupted applications from their saved stage.
    """
    console.print("[bold green]Resuming Pipeline for Pause-State Applications...[/bold green]")
    runner = get_runner(config)
    try:
        run_id = runner.run(resume_only=True, max_stage=12)
        console.print(f"[bold green]Resumed pipeline runs finished for {run_id}.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Resume execution failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("retry")
def retry_failed_applications(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Reset applications in failed states (Failed, Draft Failed, etc.) and retry them.
    """
    console.print("[bold orange3]Retrying failed pipeline stages...[/bold orange3]")
    runner = get_runner(config)
    try:
        run_id = runner.retry_failed()
        console.print(f"[bold green]Retry run finished for {run_id}.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Retry process failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e



@app.command("auth")
def authenticate_gmail(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Authenticate Gmail API connection interactively to generate token.json.
    """
    console.print("[bold cyan]Starting interactive Gmail authentication...[/bold cyan]")
    runner = get_runner(config)
    success = runner.gmail.authenticate(interactive=True)
    if success:
        console.print("[bold green]Gmail authentication completed successfully! Credentials saved.[/bold green]")
    else:
        console.print("[bold red]Gmail authentication failed.[/bold red]")
        raise typer.Exit(code=1)


@app.command("init-db")
def initialize_database(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Initialize SQLite database tables.
    """
    runner = get_runner(config)
    db_path = runner.config.pipeline.db_path
    console.print(f"[bold green]Initializing SQLite database at: {db_path}[/bold green]")
    try:
        db_init(db_path)
        console.print("[bold green]Database tables created successfully.[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Database initialization failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("config-summary")
def config_summary(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Print a summary of active configurations and preferences.
    """
    runner = get_runner(config)
    cfg = runner.config

    table = Table(
        title="Recruiting Platform Configuration Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="yellow")

    table.add_row("Database Path", cfg.pipeline.db_path)
    table.add_row("Base Resume Path", cfg.pipeline.base_resume_path)
    table.add_row("Roles Pref", ", ".join(cfg.job_preferences.roles))
    table.add_row("Geographies", ", ".join(cfg.job_preferences.geographies))
    table.add_row("Salary Min LPA", f"{cfg.job_preferences.salary_range.min_lpa} LPA")
    table.add_row("Salary Max LPA", f"{cfg.job_preferences.salary_range.max_lpa} LPA")
    table.add_row(
        "Company Size (Employees)",
        f"{cfg.job_preferences.company_size.min_employees} - {cfg.job_preferences.company_size.max_employees}",
    )
    table.add_row("Daily Draft Limit", str(cfg.pipeline.daily_draft_limit))
    table.add_row("LLM Provider", cfg.llm.provider)
    table.add_row("LLM Model", cfg.llm.model)
    table.add_row("LLM Fast Model", cfg.llm.fast_model or "-")
    table.add_row("Cache Lifetime (seconds)", str(cfg.pipeline.cache_lifetime_seconds))
    table.add_row("Sector Weights", ", ".join(f"{k}={v}" for k, v in cfg.target_profile.sector_weights.items()))
    table.add_row("Discovery Sources", ", ".join(cfg.discovery.sources))
    table.add_row("Job Sources", ", ".join(cfg.discovery.job_sources))
    table.add_row("Contact Sources", ", ".join(cfg.contacts.sources))
    table.add_row("Target Personas", ", ".join(cfg.contacts.personas))
    table.add_row("Resume Variants", ", ".join(v.name for v in cfg.resumes) or "legacy AI/Base")
    table.add_row("Auto Send", str(cfg.outreach.auto_send))
    table.add_row("Follow-ups (days)", ", ".join(str(f.after_days) for f in cfg.outreach.followups) or "none")

    console.print(table)


@app.command("status")
def view_status(config: str = typer.Option("config.yaml", help="Path to config.yaml")) -> None:
    """
    View current status and statistics of job applications.
    """
    runner = get_runner(config)
    session_factory = get_session_factory(runner.config.pipeline.db_path)
    session = session_factory()

    try:
        total_apps = session.query(Application).count()
        completed_apps = session.query(Application).filter(Application.state == "Completed").count()
        failed_apps = (
            session.query(Application)
            .filter(Application.state.in_(["Failed", "Research Failed", "Draft Failed", "Validation Failed"]))
            .count()
        )
        filtered_apps = (
            session.query(Application)
            .filter(Application.state.in_(["Excluded Company", "Salary Too Low", "Low Score", "Poor Fit", "Ghost Job"]))
            .count()
        )

        panel_content = (
            f"[bold cyan]Total Applications Tracked:[/bold cyan] {total_apps}\n"
            f"[bold green]Drafts Completed & Finalized:[/bold green] {completed_apps}\n"
            f"[bold red]Failed Processing Stages:[/bold red] {failed_apps}\n"
            f"[bold yellow]Filtered Out / Excluded:[/bold yellow] {filtered_apps}\n\n"
            f"[bold]Active runs in progress:[/bold] {session.query(Run).filter(Run.status == 'running').count()}"
        )
        console.print(Panel(panel_content, title="Recruiting Pipeline Dashboard", expand=False))

        # Display recent 10 applications
        if total_apps > 0:
            app_table = Table(
                title="Recent Job Applications",
                show_header=True,
                header_style="bold blue",
            )
            app_table.add_column("ID", style="dim")
            app_table.add_column("Company", style="bold")
            app_table.add_column("Job Title", style="cyan")
            app_table.add_column("Stage", style="magenta")
            app_table.add_column("State/Terminal Status", style="green")
            app_table.add_column("Score", style="yellow")

            recent = session.query(Application).order_by(Application.updated_at.desc()).limit(10).all()
            for app in recent:
                score_str = f"{app.score:.2f}" if app.score else "N/A"
                app_table.add_row(
                    str(app.id),
                    app.job.company.name,
                    app.job.title,
                    f"Stage {app.current_stage}",
                    app.state,
                    score_str,
                )
            console.print(app_table)

    finally:
        session.close()



@app.command("ui")
@app.command("widget")
def start_widget(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    port: int = typer.Option(18492, help="Port to serve dark mode widget on"),
    host: str = typer.Option("127.0.0.1", help="Host address"),
) -> None:
    """
    Launch minimal dark mode job status widget & dashboard web server on port 18492.
    """
    from src.config import load_config
    from src.web_server import is_widget_server_running, run_widget_server

    cfg = load_config(config)
    db_path = cfg.pipeline.db_path
    if is_widget_server_running(port):
        console.print(f"[bold green]UI / Job Status Widget is already running at http://{host}:{port}[/bold green]")
        return
    console.print(f"[bold cyan]Starting Minimal Dark Mode Status Widget at http://{host}:{port}...[/bold cyan]")
    run_widget_server(db_path=db_path, port=port, host=host)


@app.command("export")
def export_data(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    all_records: bool = typer.Option(False, "--all", help="Export all records, bypassing incremental manifest"),
    dir: str = typer.Option("exports", help="Target export directory"),
) -> None:
    """
    Incrementally export outreach data (company & email details) to JSON & CSV.
    """
    from src.config import load_config
    from src.utils.exporter import export_outreach_data

    cfg = load_config(config)
    console.print("[bold cyan]Executing Incremental Outreach Data Export...[/bold cyan]")
    res = export_outreach_data(db_path=cfg.pipeline.db_path, export_dir=dir, export_all=all_records)
    if res.get("exported_count", 0) > 0:
        console.print(f"[bold green]{res['message']}[/bold green]")
        console.print(f"[dim]Latest CSV file: {res['latest_csv_path']}[/dim]")
    else:
        console.print(f"[bold yellow]{res['message']}[/bold yellow]")


@app.command("clean-invalid")
def clean_invalid(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Clean invalid email IDs and non-working application states (NEVER deletes company/job/contact/email records).
    """
    from src.config import load_config
    from src.utils.cleaner import clean_invalid_emails_and_states

    cfg = load_config(config)
    console.print("[bold magenta]Cleaning invalid contact email IDs and non-working states...[/bold magenta]")
    res = clean_invalid_emails_and_states(db_path=cfg.pipeline.db_path)
    console.print(f"[bold green]{res['message']}[/bold green]")


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _print_progress(progress: dict[str, object]) -> None:
    table = Table(title=f"Campaign #{progress['id']}: {progress['goal']}", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    for key in (
        "status", "target", "companies_found", "companies_qualified", "companies_rejected",
        "applications", "drafts", "sent", "replies", "interviews",
    ):
        table.add_row(key.replace("_", " ").title(), str(progress.get(key)))
    console.print(table)


@app.command("campaign")
def start_campaign(
    goal: str = typer.Argument(..., help='Natural-language goal, e.g. "Find 200 fintech companies in India"'),
    target: int | None = typer.Option(None, help="Number of qualified companies to reach (defaults to the number in the goal)"),
    personas: str | None = typer.Option(None, help="Comma-separated personas, e.g. engineering_manager,recruiter"),
    auto_send: bool | None = typer.Option(None, "--auto-send/--drafts-only", help="Override outreach.auto_send"),
    batch: int | None = typer.Option(None, help="Companies to discover per run (default discovery.companies_per_run)"),
    max_stage: int = typer.Option(12, "--max-stage", help="Stop at this stage (e.g. 5 = research & emails only)"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """
    Start a campaign from a goal: discover companies -> contacts -> verified emails -> personalized drafts.
    Continue it on later days with `continue-campaign` (or automatically via `daily`/`daemon`).
    """
    runner = get_runner(config)
    try:
        progress = runner.start_campaign(
            goal, target=target, personas=_parse_list(personas), auto_send=auto_send, batch_size=batch, max_stage=max_stage
        )
        _print_progress(progress)
    except Exception as e:
        console.print(f"[bold red]Campaign failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("continue-campaign")
def continue_campaign(
    campaign_id: int = typer.Argument(..., help="Campaign ID (see `campaigns`)"),
    batch: int | None = typer.Option(None, help="Companies to discover this run"),
    max_stage: int = typer.Option(12, "--max-stage"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """Run the next batch of an existing campaign."""
    runner = get_runner(config)
    try:
        _print_progress(runner.continue_campaign(campaign_id, batch_size=batch, max_stage=max_stage))
    except Exception as e:
        console.print(f"[bold red]Campaign failed:[/bold red] {e}")
        raise typer.Exit(code=1) from e


@app.command("campaigns")
def list_campaigns(config: str = typer.Option("config.yaml", help="Path to config.yaml")) -> None:
    """List campaigns and their progress."""
    from src.db.models import Campaign
    from src.pipeline.campaign import campaign_progress

    runner = get_runner(config)
    session = runner.SessionLocal()
    try:
        table = Table(title="Campaigns", header_style="bold magenta")
        for col in ("ID", "Goal", "Status", "Target", "Qualified", "Drafts", "Sent", "Replies", "Interviews"):
            table.add_column(col)
        for campaign in session.query(Campaign).order_by(Campaign.id.desc()).all():
            p = campaign_progress(session, campaign)
            table.add_row(
                str(p["id"]), str(p["goal"])[:50], str(p["status"]), str(p["target"]), str(p["companies_qualified"]),
                str(p["drafts"]), str(p["sent"]), str(p["replies"]), str(p["interviews"]),
            )
        console.print(table)
    finally:
        session.close()


@app.command("discover")
def discover(
    query: str = typer.Argument(..., help='e.g. "Series A/B AI startups" or "Trading companies in India"'),
    count: int = typer.Option(25, help="How many companies to add"),
    research: bool = typer.Option(False, "--research", help="Also research, classify and fit-score each company"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """Build a target company list (no outreach). Companies are stored separately from jobs."""
    runner = get_runner(config)
    rows = runner.discover_companies(query, count=count, research=research)
    table = Table(title=f"Discovered companies for '{query}'", header_style="bold cyan")
    for col in ("ID", "Company", "Domain", "Sector", "Funding", "Employees", "Hiring", "Fit", "Status", "Source"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            str(r["id"]), str(r["name"]), str(r["domain"] or "-"), str(r["sector"] or "-"), str(r["funding_stage"] or "-"),
            str(r["employees"] or "-"), str(r["hiring"] or "-"), f"{r['fit']:.2f}" if r["fit"] is not None else "-",
            str(r["status"] or "-"), str(r["source"] or "-"),
        )
    console.print(table)


@app.command("companies")
def list_companies(
    sector: str | None = typer.Option(None, help="Filter by sector, e.g. fintech"),
    status: str | None = typer.Option(None, help="candidate | target | rejected | contacted"),
    min_fit: float = typer.Option(0.0, help="Minimum fit score"),
    sort: str = typer.Option("priority", help="priority | fit | probability | name"),
    limit: int = typer.Option(50),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """Rank tracked companies by fit and estimated probability of a reply."""
    from src.db.models import Company

    runner = get_runner(config)
    session = runner.SessionLocal()
    try:
        query = session.query(Company)
        if sector:
            query = query.filter(Company.sector == sector)
        if status:
            query = query.filter(Company.status == status)
        companies = [c for c in query.all() if (c.fit_score or 0.0) >= min_fit]
        keys = {
            "fit": lambda c: c.fit_score or 0.0,
            "probability": lambda c: c.response_probability or 0.0,
            "name": lambda c: c.name.lower(),
            "priority": lambda c: (c.fit_score or 0.5) * (0.5 + (c.response_probability or 0.08) * 5),
        }
        companies.sort(key=keys.get(sort, keys["priority"]), reverse=sort != "name")
        table = Table(title="Companies", header_style="bold cyan")
        for col in ("ID", "Company", "Sector", "Funding", "Size", "Hiring", "Fit", "Reply prob.", "Status"):
            table.add_column(col)
        for c in companies[:limit]:
            table.add_row(
                str(c.id), c.name, c.sector or "-", c.funding_stage or "-", str(c.employee_count or "-"),
                c.hiring_status or "-", f"{c.fit_score:.2f}" if c.fit_score is not None else "-",
                f"{c.response_probability:.1%}" if c.response_probability is not None else "-", c.status or "-",
            )
        console.print(table)
    finally:
        session.close()


@app.command("outreach")
@app.command("send-scheduled")
def outreach_cycle(config: str = typer.Option("config.yaml", help="Path to config.yaml")) -> None:
    """Run the outreach cycle: detect replies/bounces, send due emails, create/send follow-ups."""
    runner = get_runner(config)
    summary = runner.run_outreach_cycle()
    console.print(Panel("\n".join(f"{k}: {v}" for k, v in summary.items()) or "Gmail not authorized.", title="Outreach cycle"))


@app.command("funnel")
def show_funnel(
    campaign: int | None = typer.Option(None, help="Restrict to a campaign ID"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """Conversion funnel: Companies -> Contacts -> Emails -> Drafts -> Sent -> Replies -> Interviews."""
    from src.analytics.funnel import compute_funnel, email_verification_stats

    runner = get_runner(config)
    session = runner.SessionLocal()
    try:
        data = compute_funnel(session, campaign)
        table = Table(title="Outreach Funnel" + (f" (campaign #{campaign})" if campaign else ""), header_style="bold green")
        table.add_column("Stage", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("% of previous", justify="right")
        table.add_column("", style="green")
        top = max(data["funnel"][0]["count"], 1)
        for row in data["funnel"]:
            bar = "█" * max(1 if row["count"] else 0, int(30 * row["count"] / top))
            prev = f"{row['pct_of_previous']}%" if row["pct_of_previous"] is not None else "-"
            table.add_row(row["stage"], str(row["count"]), prev, bar)
        console.print(table)
        extra = data["extra"]
        stats = email_verification_stats(session)
        console.print(
            f"Qualified companies: {extra['qualified_companies']}  |  Rejected (poor fit): {extra['rejected_companies']}  |  "
            f"Bounced: {extra['bounced']}  |  Reply rate: {extra['reply_rate']}%  |  Interview rate: {extra['interview_rate']}%\n"
            f"Email verification: {stats['smtp_verified_pct']}% SMTP-verified, {stats['deliverable_likely_pct']}% "
            f"deliverable-likely (valid + catch-all) of {stats['checked']} checked; status counts {stats['counts']}; "
            f"catch-all domains: {stats['catch_all_domains']}"
        )
    finally:
        session.close()


@app.command("insights")
def insights(config: str = typer.Option("config.yaml", help="Path to config.yaml")) -> None:
    """What is working: reply rates by sector, persona, resume variant and funding stage (learned from outcomes)."""
    from src.analytics.funnel import outcome_breakdown
    from src.intel.learning import compute_outcome_stats

    runner = get_runner(config)
    session = runner.SessionLocal()
    try:
        stats = compute_outcome_stats(session, runner.config)
        console.print(
            f"[bold]Sent:[/bold] {stats.total_sent}  [bold]Replies:[/bold] {stats.total_replies}  "
            f"[bold]Interviews:[/bold] {stats.total_interviews}  [bold]Smoothed reply rate:[/bold] {stats.global_rate:.1%}"
        )
        for feature in ("sector", "persona", "resume_variant", "funding_stage", "size_bucket"):
            rows = outcome_breakdown(session, feature)
            if not rows:
                continue
            table = Table(title=f"By {feature}", header_style="bold magenta")
            for col in ("Value", "Sent", "Replies", "Interviews", "Reply rate", "Learned multiplier"):
                table.add_column(col)
            for r in rows:
                table.add_row(
                    str(r["value"]), str(r["sent"]), str(r["replies"]), str(r["interviews"]), f"{r['reply_rate']}%",
                    f"x{stats.multiplier(feature, str(r['value'])):.2f}",
                )
            console.print(table)
        if stats.total_sent == 0:
            console.print("[yellow]No sent outreach yet — insights appear once emails are sent and tracked.[/yellow]")
    finally:
        session.close()


@app.command("mark")
def mark_application(
    app_id: int = typer.Argument(..., help="Application ID"),
    status: str = typer.Argument(..., help="interview | offer | replied | not_interested | rejected | no_response"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """Record an outcome manually (e.g. an interview scheduled over the phone). Stops pending follow-ups."""
    from src.db.models import Application as App
    from src.outreach.engine import OutreachEngine, log_event

    allowed = {"interview", "offer", "replied", "not_interested", "rejected", "no_response"}
    if status not in allowed:
        console.print(f"[bold red]Status must be one of {sorted(allowed)}[/bold red]")
        raise typer.Exit(code=1)
    runner = get_runner(config)
    session = runner.SessionLocal()
    try:
        application = session.get(App, app_id)
        if application is None:
            console.print(f"[bold red]Application #{app_id} not found.[/bold red]")
            raise typer.Exit(code=1)
        from datetime import UTC, datetime

        now = datetime.now(UTC).replace(tzinfo=None)
        application.outreach_status = status
        if status in ("replied", "interview", "offer", "not_interested", "rejected"):
            application.replied_at = application.replied_at or now
        if status in ("interview", "offer"):
            application.interview_at = application.interview_at or now
        OutreachEngine(session, runner.config, runner.gmail).cancel_followups(application, f"marked {status}")
        log_event(session, app_id, f"marked_{status}", details="manual update")
        session.commit()
        console.print(f"[bold green]Application #{app_id} marked as {status}.[/bold green]")
    finally:
        session.close()


@app.command("daily")
def daily(config: str = typer.Option("config.yaml", help="Path to config.yaml")) -> None:
    """One day of automation: replies & follow-ups, continue campaigns, discover new companies, draft/send."""
    runner = get_runner(config)
    results = runner.run_daily()
    console.print(Panel(str(results), title="Daily run"))


@app.command("daemon")
def daemon(
    daily_at: str = typer.Option("09:00", help="Local time for the daily discovery run (HH:MM)"),
    every: int = typer.Option(15, help="Minutes between outreach cycles"),
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
) -> None:
    """Keep running in the foreground: continuous discovery every day + outreach/reply tracking every N minutes."""
    from src.scheduler import run_daemon

    console.print(f"[bold green]CareerPilot daemon running (daily at {daily_at}, outreach every {every} min). Ctrl+C to stop.[/bold green]")
    try:
        run_daemon(lambda: get_runner(config), daily_at=daily_at, outreach_every_minutes=every)
    except KeyboardInterrupt:
        console.print("Daemon stopped.")


@app.command("schedule")
def schedule(
    action: str = typer.Argument("status", help="install | remove | status"),
    daily_at: str = typer.Option("09:00", help="Daily run time (HH:MM)"),
    every: int = typer.Option(30, help="Minutes between outreach cycles"),
) -> None:
    """Install OS-level automation (Windows Task Scheduler or systemd user timers)."""
    from pathlib import Path

    from src.scheduler import TASK_DAILY, TASK_OUTREACH, install_schedule, remove_schedule, windows_task_status

    if action == "install":
        for line in install_schedule(Path.cwd(), daily_at, every):
            console.print(line)
        console.print("[bold green]Automation installed.[/bold green]")
    elif action == "remove":
        for line in remove_schedule():
            console.print(line)
    else:
        import sys

        if sys.platform == "win32":
            console.print(f"{TASK_DAILY}: {windows_task_status(TASK_DAILY) or 'not installed'}")
            console.print(f"{TASK_OUTREACH}: {windows_task_status(TASK_OUTREACH) or 'not installed'}")
        else:
            import subprocess

            subprocess.run(["systemctl", "--user", "list-timers", "careerpilot*"], check=False)


@app.command("doctor")
def doctor(
    config: str = typer.Option("config.yaml", help="Path to config.yaml"),
    skip_llm: bool = typer.Option(False, "--skip-llm", help="Do not make a test LLM call"),
) -> None:
    """Check setup: LLM (Groq) key & model, Gmail scopes, SMTP port 25, Typst, API keys, resumes, timezone."""
    import shutil
    import socket

    from src.config import load_config
    from src.outreach.scheduling import resolve_timezone
    from src.providers.llm import GroqProvider, get_llm_provider

    cfg = load_config(config)
    table = Table(title="CareerPilot doctor", header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Result")

    def row(name: str, ok: bool, detail: str) -> None:
        table.add_row(name, f"[green]OK[/green] {detail}" if ok else f"[red]!![/red] {detail}")

    key = cfg.llm.resolved_api_key()
    row("LLM provider", True, f"{cfg.llm.provider} / {cfg.llm.model} (fast: {cfg.llm.fast_model or '-'})")
    if cfg.llm.provider.lower() in ("groq", "openai", "anthropic", "gemini"):
        row("LLM API key", bool(key), "found" if key else f"missing: set llm.api_key or {cfg.llm.provider.upper()}_API_KEY")
    llm = get_llm_provider(cfg.llm)
    if isinstance(llm, GroqProvider) and key:
        try:
            models = llm.list_models()
            wanted = [m for m in [llm.model, cfg.llm.fast_model, *cfg.llm.fallback_models] if m]
            missing = [m for m in wanted if m not in models]
            row("Groq models", not missing, f"missing {missing}; available: {', '.join(models[:12])}" if missing else f"{', '.join(wanted)} available")
        except Exception as e:
            row("Groq models", False, str(e)[:200])
    if not skip_llm and (key or cfg.llm.provider.lower() in ("agy_cli", "local_agy")):
        try:
            reply = llm.generate_text("Reply with the single word OK.")
            row("LLM test call", "ok" in reply.lower(), reply.strip()[:60])
        except Exception as e:
            row("LLM test call", False, str(e)[:200])

    row("Gmail credentials.json", os.path.exists(cfg.gmail.credentials_file), cfg.gmail.credentials_file)
    if os.path.exists(cfg.gmail.token_file):
        import json as _json

        with open(cfg.gmail.token_file, encoding="utf-8") as f:
            token = _json.load(f)
        granted = set(token.get("scopes") or [])
        can_read = bool(granted & {"https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.modify"})
        row("Gmail token scopes", can_read, "compose + read (reply tracking on)" if can_read else "no read scope: run `recruiting-platform auth` to enable reply tracking")
    else:
        row("Gmail token", False, "not authorized yet: run `recruiting-platform auth`")

    try:
        with socket.create_connection(("gmail-smtp-in.l.google.com", 25), timeout=6):
            row("SMTP port 25 (mailbox verification)", True, "reachable")
    except Exception as e:
        row("SMTP port 25 (mailbox verification)", False, f"blocked/unreachable ({e}); emails will be pattern/Hunter-verified instead")

    row("Typst (PDF resumes)", bool(shutil.which("typst")), shutil.which("typst") or "not installed: .typ files will be attached")
    for name in ("hunter", "apollo", "github", "serper", "brave"):
        value = cfg.api_keys.get(name)
        table.add_row(f"API key: {name}", "[green]set[/green]" if value else "[yellow]not set (optional)[/yellow]")
    variants = cfg.resumes or []
    existing = [v.name for v in variants if os.path.exists(v.path)]
    row("Resume variants", bool(existing) or os.path.exists(cfg.pipeline.base_resume_path), f"{existing or 'legacy base resume'}")
    tz = resolve_timezone(cfg.outreach.send_window.timezone)
    row("Send-window timezone", "UTC" not in str(tz) or cfg.outreach.send_window.timezone == "UTC", str(tz))
    import sys

    if sys.platform == "win32":
        from src.scheduler import TASK_DAILY, windows_task_status

        status = windows_task_status(TASK_DAILY)
        row("Daily automation", status is not None, status or "not installed: run `recruiting-platform schedule install` (or `daemon`)")
    console.print(table)


if __name__ == "__main__":
    app()
