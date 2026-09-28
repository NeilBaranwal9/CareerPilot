export TARGET := $(target)
export JD := $(jd)
export GOAL := $(goal)
export QUERY := $(query)

.DEFAULT_GOAL := default

.PHONY: default install lint test run search research tailor draft retry resume send-scheduled auth ui widget export clean-invalid clean format migrate help target targeted status campaign campaigns discover companies outreach funnel insights daily daemon schedule-install doctor

default:
ifneq ($(target),)
	uv run recruiting-platform targeted "$(TARGET)"
else
	@$(MAKE) help
endif

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make lint          - Lint code using ruff and typecheck using mypy"
	@echo "  make format        - Format code using ruff"
	@echo "  make test          - Run tests using pytest"
	@echo "  make migrate       - Initialize or migrate SQLite database"
	@echo "  make run           - Run the entire recruiting pipeline end-to-end (auto-launches web widget)"
	@echo "  make ui            - Launch recruiting web dashboard & dark mode status widget on port 18492"
	@echo "  make widget        - Launch recruiting web dashboard & dark mode status widget on port 18492"
	@echo "  make export        - Incrementally export outreach data (Company + Contact + Email info)"
	@echo "  make clean-invalid - Clean non-existent emails & bad states without deleting company/job/contact/email"
	@echo "  make search        - Run Stage 0-2 (Job and Company Discovery)"
	@echo "  make research      - Run Stage 3-5 (Company and Contact Research)"
	@echo "  make tailor        - Run Stage 6-7 (Opportunity Scoring and Resume Tailoring)"
	@echo "  make draft         - Run Stage 8-10 (Email Gen and Gmail Draft Creation)"
	@echo "  make resume        - Resume any active/paused job applications"
	@echo "  make retry         - Reset failed applications and retry them"
	@echo "  make send-scheduled - Send drafts whose scheduled time has passed"
	@echo "  make auth          - Authenticate Gmail API connection interactively"
	@echo "  make target        - Run targeted outreach (e.g. make target target=\"ElevenLabs\")"
	@echo "  make campaign      - Start a campaign (e.g. make campaign goal=\"Find 200 fintech companies in India\")"
	@echo "  make campaigns     - List campaigns and progress"
	@echo "  make discover      - Build a target list (e.g. make discover query=\"Series A/B AI startups\")"
	@echo "  make companies     - Rank companies by fit & reply probability"
	@echo "  make outreach      - Replies/bounces, send due emails, follow-ups"
	@echo "  make funnel        - Show the conversion funnel"
	@echo "  make insights      - Reply rates by sector/persona/resume variant"
	@echo "  make daily         - One full automation cycle"
	@echo "  make daemon        - Run continuously (daily discovery + outreach every 15 min)"
	@echo "  make schedule-install - Install OS scheduler (Windows Task Scheduler / systemd)"
	@echo "  make doctor        - Check Groq, Gmail scopes, SMTP port 25, Typst, API keys"
	@echo "  make status        - Check status of systemd service, timer, and recent logs"
	@echo "  make clean         - Clean temporary Python files and logs"

install:
	uv sync

lint:
	uv run ruff check .
	uv run mypy src/ --explicit-package-bases

format:
	uv run ruff format .

test:
	uv run pytest

migrate:
	uv run recruiting-platform init-db

run:
	uv run recruiting-platform run

ui widget:
	uv run recruiting-platform ui --port 18492

export:
	uv run recruiting-platform export

clean-invalid:
	uv run recruiting-platform clean-invalid

search:
	uv run recruiting-platform search

research:
	uv run recruiting-platform research

tailor:
	uv run recruiting-platform tailor

draft:
	uv run recruiting-platform draft

resume:
	uv run recruiting-platform resume

retry:
	uv run recruiting-platform retry

send-scheduled:
	uv run recruiting-platform send-scheduled

auth:
	uv run recruiting-platform auth

target targeted:
	uv run recruiting-platform targeted "$(TARGET)"

campaign:
	uv run recruiting-platform campaign "$(GOAL)"

campaigns:
	uv run recruiting-platform campaigns

discover:
	uv run recruiting-platform discover "$(QUERY)" --research

companies:
	uv run recruiting-platform companies

outreach:
	uv run recruiting-platform outreach

funnel:
	uv run recruiting-platform funnel

insights:
	uv run recruiting-platform insights

daily:
	uv run recruiting-platform daily

daemon:
	uv run recruiting-platform daemon

schedule-install:
	uv run recruiting-platform schedule install

doctor:
	uv run recruiting-platform doctor

status:
	@echo "=== systemd UI Service Status ==="
	@systemctl --user status careerpilot-ui.service --no-pager || true
	@echo ""
	@echo "=== systemd Agent Service Status ==="
	@systemctl --user status careerpilot.service --no-pager || true
	@echo ""
	@echo "=== systemd Timer Status ==="
	@systemctl --user status careerpilot.timer --no-pager || true
	@echo ""
	@echo "=== Recent Application Logs ==="
	@if [ -f logs/platform.log ]; then tail -n 25 logs/platform.log; else echo "No local log file found."; fi

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf src/__pycache__ src/**/*.__pycache__ tests/__pycache__
	rm -f logs/platform.log
	@echo "Cleaned cache files and logs."
