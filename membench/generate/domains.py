"""Fact templates for the synthetic generator (`generate/synthetic.py`).

Each `Domain` is one pack. An `Attribute` is a fact about the user, stated
once (`state`), possibly revised (`update`), and asked about now (`question`)
or as of a date (`past_question`); questions avoid the statement's wording
where they can. A `Link` is a two-fact chain for multi-hop questions: the
first fact names an entity, the second gives the entity's value, and the
question names the entity only through the first fact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Attribute:
    key: str
    label: str
    state: str
    update: str
    question: str
    past_question: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class Link:
    key: str
    first: str
    second: str
    question: str
    entities: tuple[str, ...]
    values: tuple[str, ...]


@dataclass(frozen=True)
class Domain:
    description: str
    fact_role: str
    forget_request: str
    attributes: tuple[Attribute, ...]
    links: tuple[Link, ...]
    distractors: tuple[tuple[str, str], ...]


_PEOPLE = ("Priya Raman", "Tomás Ortega", "Aiko Tanaka", "Lena Fischer", "Marcus Bell",
           "Nadia Haddad", "Owen Gallagher", "Sofia Rossi")
_CITIES = ("Lisbon", "Denver", "Osaka", "Nairobi", "Krakow", "Montreal", "Austin", "Dublin")

CUSTOMER_SUPPORT = Domain(
    description="Customer support tickets across months (scaled customer_support).",
    fact_role="user",
    forget_request="Please erase my {what} from your records.",
    attributes=(
        Attribute("plan", "plan", "I'm currently subscribed to the {v} tier.",
                  "I've just moved my subscription over to the {v} tier.",
                  "Which subscription level does the customer have right now?",
                  "Which subscription level did the customer have on {date}?",
                  ("Basic", "Plus", "Pro", "Enterprise", "Starter")),
        Attribute("address", "shipping address", "Please ship everything to {v}.",
                  "We relocated, deliveries should now go to {v}.",
                  "Where should orders for this customer be delivered?",
                  "Where were this customer's orders being delivered on {date}?",
                  ("12 Elm Street, Leeds", "48 Harbor Road, Cork", "7 Rue Cler, Lyon",
                   "301 Pine Ave, Tacoma", "22 Kingsway, Perth")),
        Attribute("device", "device", "The app keeps crashing on my {v}.",
                  "I replaced my phone, I'm on a {v} now.",
                  "What hardware is the customer using the app on?",
                  "What hardware was the customer using the app on as of {date}?",
                  ("Pixel 8", "iPhone 15", "Galaxy S23", "OnePlus 12", "Moto G84")),
        Attribute("billing", "billing cycle", "I pay {v} for my subscription.",
                  "Can you switch me to paying {v} from now on?",
                  "How often is this customer billed?",
                  "How often was this customer billed on {date}?",
                  ("monthly", "quarterly", "annually", "every two years")),
        Attribute("contact", "preferred contact channel", "The best way to reach me is {v}.",
                  "Actually, please use {v} to contact me going forward.",
                  "How does the customer prefer to be contacted?",
                  "How did the customer prefer to be contacted as of {date}?",
                  ("SMS", "a phone call", "WhatsApp", "in-app chat", "Telegram")),
        Attribute("email", "email address", "My account email is {v}.",
                  "I changed my login email to {v}.",
                  "What is the customer's login email?",
                  "What was the customer's login email on {date}?",
                  ("dana.w@example.com", "r.okafor@example.net", "mei.lin@example.org",
                   "jb.cole@example.com", "ana.p@example.net")),
        Attribute("language", "language", "I'd like support replies in {v}, please.",
                  "Could you start replying in {v} instead?",
                  "In which language should support write to this customer?",
                  "In which language was support writing to this customer on {date}?",
                  ("Spanish", "German", "Portuguese", "Japanese", "French")),
    ),
    links=(
        Link("manager", "My account manager is {e}.", "{e} is based in our {v} office.",
             "Which office is the customer's account manager based in?", _PEOPLE, _CITIES),
        Link("refund", "The refund for my broken order was approved by {e}.",
             "{e} said the money will be back in {v}.",
             "How long will the approved refund take to arrive, per whoever approved it?",
             _PEOPLE, ("3 business days", "5 business days", "two weeks", "48 hours")),
    ),
    distractors=(
        ("user", "Thanks, that worked for now."),
        ("assistant", "Happy to help! Is there anything else I can do today?"),
        ("user", "Is your status page showing any outage right now?"),
        ("assistant", "All systems are operational at the moment."),
        ("user", "My colleague said the new dashboard looks great."),
        ("assistant", "Glad to hear it. We shipped that redesign last month."),
        ("user", "Do you have a holiday schedule for support hours?"),
        ("assistant", "Support runs 24/7, with reduced staffing on public holidays."),
        ("user", "The weather here is terrible today, sorry for the slow replies."),
        ("assistant", "No problem at all, take your time."),
    ),
)

CODING_ASSISTANT = Domain(
    description="A developer's coding-assistant sessions over months (scaled coding_assistant).",
    fact_role="user",
    forget_request="Please forget the {what} I gave you earlier.",
    attributes=(
        Attribute("python", "Python version", "This service runs on Python {v}.",
                  "We finished the upgrade, the service is on Python {v} now.",
                  "Which interpreter release does the service target?",
                  "Which interpreter release did the service target on {date}?",
                  ("3.9", "3.10", "3.11", "3.12", "3.13")),
        Attribute("db", "database", "Our persistence layer is {v}.",
                  "We migrated storage to {v} last sprint.",
                  "What datastore backs the project?",
                  "What datastore backed the project as of {date}?",
                  ("PostgreSQL", "MySQL", "SQLite", "MongoDB", "DynamoDB")),
        Attribute("tests", "test runner", "We run our suite with {v}.",
                  "We switched the suite over to {v}.",
                  "Which tool executes the project's tests?",
                  "Which tool executed the project's tests on {date}?",
                  ("pytest", "unittest", "nose2", "ward", "hypothesis-only runs")),
        Attribute("style", "formatter", "Code style is enforced by {v}.",
                  "We dropped the old setup, {v} formats everything now.",
                  "What enforces formatting in this repo?",
                  "What enforced formatting in this repo as of {date}?",
                  ("black", "ruff format", "yapf", "autopep8")),
        Attribute("deploy", "deploy target", "We deploy to {v}.",
                  "Deployments moved to {v}.",
                  "Where does the service get deployed?",
                  "Where was the service deployed on {date}?",
                  ("AWS Lambda", "Kubernetes on GKE", "Fly.io", "Heroku", "a bare-metal VM")),
        Attribute("token", "API token", "The staging API token is {v}.",
                  "I rotated the staging API token, it's {v} now.",
                  "What credential does staging use?",
                  "What credential did staging use on {date}?",
                  ("tok_live_4f9a", "tok_live_88c2", "tok_live_19be", "tok_live_c07d")),
        Attribute("branch", "release branch", "Releases are cut from {v}.",
                  "We now cut releases from {v}.",
                  "Which branch do releases come from?",
                  "Which branch did releases come from on {date}?",
                  ("release/main", "release/stable", "trunk-lts", "prod-cut", "release/next")),
    ),
    links=(
        Link("owner", "The billing module is owned by {e}.", "{e} prefers reviews in {v}.",
             "Where does the billing module's owner want code reviews done?", _PEOPLE,
             ("GitHub PR comments", "a Zoom pairing session", "Gerrit", "a shared doc")),
        Link("ci", "Our CI pipeline lives in the {e} repo.",
             "The {e} repo pins {v}.",
             "Which Node version does the repo holding our CI pipeline pin?",
             ("infra-tools", "platform-ci", "build-kit", "ops-shared"), ("Node 18", "Node 20", "Node 22")),
    ),
    distractors=(
        ("user", "Can you explain list comprehensions again?"),
        ("assistant", "Sure: [f(x) for x in xs if cond(x)] builds a list in one expression."),
        ("user", "What's the difference between a tuple and a list?"),
        ("assistant", "Tuples are immutable; lists can be changed in place."),
        ("user", "I'm getting a flaky test on CI but not locally."),
        ("assistant", "Check for shared state, test ordering, and time-dependent code."),
        ("user", "Remind me how to squash commits."),
        ("assistant", "Use an interactive rebase and mark later commits as squash."),
        ("user", "Is it worth adding type hints to old code?"),
        ("assistant", "Usually yes, starting with public interfaces."),
    ),
)

TOOL_CALL_MEMORY = Domain(
    description="Facts that only appear in tool results an agent received.",
    fact_role="tool",
    forget_request='forget_user_data(field="{what}") -> {{"status": "queued"}}',
    attributes=(
        Attribute("carrier", "carrier", 'get_order(id="A-1042") -> {{"carrier": "{v}"}}',
                  'get_order(id="A-1042") -> {{"carrier": "{v}", "note": "reassigned"}}',
                  "Which company is delivering the user's order?",
                  "Which company was delivering the user's order on {date}?",
                  ("UPS", "DHL", "FedEx", "Royal Mail", "PostNL")),
        Attribute("balance", "account balance", 'get_balance() -> {{"available": "{v}"}}',
                  'get_balance() -> {{"available": "{v}", "refreshed": true}}',
                  "How much money does the user have available?",
                  "How much money did the user have available on {date}?",
                  ("$1,204.17", "$88.40", "$3,950.00", "$412.66", "$27.05")),
        Attribute("meeting", "meeting room", 'calendar.lookup("design sync") -> {{"room": "{v}"}}',
                  'calendar.lookup("design sync") -> {{"room": "{v}", "moved": true}}',
                  "Where is the design sync being held?",
                  "Where was the design sync being held as of {date}?",
                  ("Room 4B", "Atlas", "Juniper", "Room 12", "the Loft")),
        Attribute("flight", "flight gate", 'flight_status("LX318") -> {{"gate": "{v}"}}',
                  'flight_status("LX318") -> {{"gate": "{v}", "changed": true}}',
                  "From which gate does the user's flight leave?",
                  "From which gate was the user's flight leaving on {date}?",
                  ("A12", "B7", "C33", "D4", "E19")),
        Attribute("ticket", "support ticket status", 'jira.get("OPS-77") -> {{"status": "{v}"}}',
                  'jira.get("OPS-77") -> {{"status": "{v}"}}',
                  "What state is the ops ticket the user filed in?",
                  "What state was the user's ops ticket in on {date}?",
                  ("Triage", "In Progress", "Blocked on vendor", "In Review", "Resolved")),
        Attribute("card", "saved card", 'wallet.list() -> {{"default_card": "{v}"}}',
                  'wallet.list() -> {{"default_card": "{v}"}}',
                  "Which payment card is the user's default?",
                  "Which payment card was the user's default on {date}?",
                  ("Visa ending 4417", "Amex ending 1009", "Mastercard ending 7731",
                   "Visa ending 2280")),
        Attribute("weather", "forecast", 'weather("home") -> {{"tomorrow": "{v}"}}',
                  'weather("home") -> {{"tomorrow": "{v}", "updated": true}}',
                  "What will tomorrow be like where the user lives?",
                  "What did the forecast for the user's home say on {date}?",
                  ("light rain", "sunny, 24C", "snow showers", "overcast, 11C", "thunderstorms")),
    ),
    links=(
        Link("assignee", 'jira.get("OPS-91") -> {{"assignee": "{e}"}}',
             'directory.lookup("{e}") -> {{"timezone": "{v}"}}',
             "What timezone is the person handling OPS-91 in?", _PEOPLE,
             ("UTC+1", "UTC-5", "UTC+9", "UTC+5:30", "UTC-8")),
        Link("vendor", 'po.get("PO-5521") -> {{"vendor": "{e}"}}',
             'vendors.get("{e}") -> {{"payment_terms": "{v}"}}',
             "What are the payment terms of the supplier on PO-5521?",
             ("Acme Corp", "Globex", "Initech", "Umbrella Ltd"), ("net 30", "net 60", "prepaid")),
    ),
    distractors=(
        ("user", "Can you check that for me?"),
        ("assistant", "Done, I looked it up."),
        ("tool", 'get_time() -> {"utc": "14:02"}'),
        ("tool", 'search_docs("vacation policy") -> {"hits": 3}'),
        ("user", "Thanks. Also, how's the traffic downtown?"),
        ("tool", 'traffic("downtown") -> {"delay_min": 12}'),
        ("assistant", "There's about a 12 minute delay downtown."),
        ("tool", 'news.top() -> {"headline": "Local team wins final"}'),
        ("user", "Never mind, I'll sort it out later."),
        ("assistant", "Sounds good, just ask whenever."),
    ),
)

DOCUMENT_CHAIN = Domain(
    description="Facts spread across a chain of shared documents that point to each other.",
    fact_role="document",
    forget_request="Retraction notice: remove the {what} from the shared project docs.",
    attributes=(
        Attribute("budget", "budget figure", "Project brief (v1): approved budget {v}.",
                  "Finance addendum, supersedes the brief: approved budget is now {v}.",
                  "How much funding does the project have?",
                  "How much funding did the project have on {date}?",
                  ("$120k", "$95k", "$150k", "$80k", "$210k")),
        Attribute("launch", "launch date", "Roadmap doc: launch planned for {v}.",
                  "Roadmap doc, revision note: launch moved to {v}.",
                  "When is the project going live?",
                  "When was the project scheduled to go live as of {date}?",
                  ("March 3", "April 14", "May 29", "June 30", "July 21")),
        Attribute("vendor", "selected vendor", "Procurement memo: we selected {v}.",
                  "Procurement memo (amended): selection changed to {v}.",
                  "Which supplier was chosen?",
                  "Which supplier had been chosen as of {date}?",
                  ("Northwind", "Contoso", "Fabrikam", "Tailspin", "Litware")),
        Attribute("region", "hosting region", "Architecture note: hosting in {v}.",
                  "Architecture note v2: hosting relocated to {v}.",
                  "Where will the system be hosted?",
                  "Where was the system going to be hosted on {date}?",
                  ("eu-west-1", "us-east-2", "ap-southeast-1", "eu-central-1", "ca-central-1")),
        Attribute("sla", "uptime target", "Ops runbook: uptime target {v}.",
                  "Ops runbook update: uptime target revised to {v}.",
                  "What availability commitment does the service have?",
                  "What availability commitment did the service have on {date}?",
                  ("99.5%", "99.9%", "99.95%", "99.99%")),
        Attribute("codename", "codename", "Kickoff deck: internal codename {v}.",
                  "Comms note: codename changed to {v}.",
                  "What do people call the project internally?",
                  "What did people call the project internally as of {date}?",
                  ("Bluebird", "Granite", "Halcyon", "Ironwood", "Juniper")),
        Attribute("headcount", "team size", "Staffing plan: {v} assigned.",
                  "Staffing plan revision: now {v} assigned.",
                  "How many engineers are on the project?",
                  "How many engineers were on the project on {date}?",
                  ("4 engineers", "6 engineers", "9 engineers", "12 engineers",
                   "15 engineers")),
    ),
    links=(
        Link("review", "Project brief: the security review is owned by {e}.",
             "Notes by {e}: we are going with {v} for single sign-on.",
             "Which SSO provider is the security review owner going with?", _PEOPLE,
             ("Okta", "Auth0", "Azure AD", "Keycloak", "Google Workspace")),
        Link("appendix", "Main spec: data retention rules are in appendix {e}.",
             "Appendix {e}: logs are retained for {v}.",
             "How long are logs kept, per the appendix the main spec points to?",
             ("C", "D", "F", "H"), ("30 days", "90 days", "1 year", "7 days")),
    ),
    distractors=(
        ("document", "Meeting notes: agreed to meet again next week."),
        ("document", "Glossary: SLO means service level objective."),
        ("user", "I uploaded the latest docs to the shared folder."),
        ("assistant", "Got them, I'll keep them in mind."),
        ("document", "Style guide: use sentence case for headings."),
        ("document", "FAQ: the office closes at 6pm on Fridays."),
        ("user", "Can you summarize the meeting notes later?"),
        ("assistant", "Sure, whenever you need it."),
        ("document", "Changelog: fixed typos in the onboarding guide."),
        ("document", "Template: weekly status report, sections A to D."),
    ),
)

MULTI_STEP_DECISIONS = Domain(
    description="Constraints set early in a planning conversation and needed later.",
    fact_role="user",
    forget_request="Scratch what I said about the {what}; please don't keep it.",
    attributes=(
        Attribute("budget", "spending cap", "For the offsite we can't spend more than {v}.",
                  "Finance came back: the offsite cap is now {v}.",
                  "What's the most we're allowed to spend on the offsite?",
                  "What was the most we were allowed to spend on the offsite as of {date}?",
                  ("$5,000", "$8,000", "$12,000", "$3,500", "$15,000")),
        Attribute("diet", "dietary requirement", "Catering must be {v} for the whole group.",
                  "Change of plan: catering has to be {v} instead.",
                  "What food restriction applies to the event catering?",
                  "What food restriction applied to the event catering on {date}?",
                  ("vegetarian", "vegan", "gluten-free", "halal", "kosher")),
        Attribute("dates", "event window", "The event has to happen in {v}.",
                  "We had to shift the event window to {v}.",
                  "When does the event need to take place?",
                  "When did the event need to take place as of {date}?",
                  ("the first week of May", "mid June", "late September", "early March")),
        Attribute("size", "headcount", "Plan for {v}.",
                  "RSVPs changed things, plan for {v} now.",
                  "How many people should the venue hold?",
                  "How many people was the venue supposed to hold on {date}?",
                  ("25 attendees", "40 attendees", "60 attendees", "80 attendees",
                   "120 attendees")),
        Attribute("travel", "travel limit", "Nobody should travel more than {v} to get there.",
                  "We relaxed it: the travel limit is {v} now.",
                  "How far can attendees be asked to travel?",
                  "How far could attendees be asked to travel as of {date}?",
                  ("2 hours", "90 minutes", "3 hours", "one short flight")),
        Attribute("access", "accessibility need", "The venue must have {v}.",
                  "Update: the venue now needs {v}.",
                  "What accessibility feature must the venue provide?",
                  "What accessibility feature did the venue have to provide on {date}?",
                  ("step-free access", "a hearing loop", "an elevator", "quiet rooms")),
        Attribute("approver", "sign-off owner", "Final sign-off on the venue sits with {v}.",
                  "Sign-off on the venue moved to {v}.",
                  "Who has to approve the venue choice?",
                  "Who had to approve the venue choice as of {date}?",
                  ("the CFO", "our VP of People", "the office manager", "the CEO")),
    ),
    links=(
        Link("speaker", "Our keynote speaker will be {e}.",
             "{e} can only present on {v}.",
             "Which day is the keynote speaker available?", _PEOPLE,
             ("Tuesday", "Wednesday", "Thursday", "Friday")),
        Link("venue", "We shortlisted the {e} as the venue.",
             "The {e} requires a deposit of {v}.",
             "How big a deposit does the shortlisted venue need?",
             ("Harbor Hall", "Grand Atrium", "Mill House", "Skyline Loft"),
             ("10%", "25%", "$1,000", "$2,500")),
    ),
    distractors=(
        ("user", "Let's think about the agenda later."),
        ("assistant", "Sure, we can draft it once the venue is settled."),
        ("user", "Did you see the game last night?"),
        ("assistant", "I don't follow live games, but happy to chat about it."),
        ("user", "Remind me to book a photographer at some point."),
        ("assistant", "Noted as a to-do for later."),
        ("user", "What's a good icebreaker for a group this size?"),
        ("assistant", "Two truths and a lie works well for most groups."),
        ("user", "My manager wants a short update on Friday."),
        ("assistant", "I can draft a summary for you then."),
    ),
)

DOMAINS = {
    "tool_call_memory": TOOL_CALL_MEMORY,
    "document_chain": DOCUMENT_CHAIN,
    "multi_step_decisions": MULTI_STEP_DECISIONS,
    "customer_support_xl": CUSTOMER_SUPPORT,
    "coding_assistant_xl": CODING_ASSISTANT,
}
