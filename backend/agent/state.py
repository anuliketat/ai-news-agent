from typing import TypedDict, List, Dict


class AgentState(TypedDict):
    raw_articles: List[Dict]      # From scrapers
    deduplicated: List[Dict]      # After cache check
    validated: List[Dict]         # After LLM validation
    actionable: List[Dict]        # Filtered for user
    digest: str                   # Final formatted Telegram message
    approval_status: str          # pending/approved/rejected/sent/skipped
    run_id: str
    stats: Dict
    errors: List[str]
