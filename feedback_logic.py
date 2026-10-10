"""Prompt building, parsing, and aggregation for the Customer Feedback Analyzer.

Pure functions in this file take a `call_llm`-shaped function as a
parameter so they can be unit tested with a fake instead of the real API.
"""
import csv
import io
import json
from collections import Counter
from dataclasses import dataclass, field

from llm_client import call_llm as default_call_llm

SENTIMENTS = ("positive", "neutral", "negative")

# Real model output occasionally uses a synonym instead of one of the three
# allowed categories (e.g. "mixed" for feedback that's part praise, part
# complaint). Normalize known synonyms instead of failing the whole batch.
SENTIMENT_ALIASES = {
    "mixed": "neutral",
    "mixed/neutral": "neutral",
    "positive/negative": "neutral",
}


class FeedbackAnalysisError(RuntimeError):
    """Raised when the feedback CSV or the model's response can't be parsed."""


@dataclass
class Feedback:
    id: str
    text: str
    rating: str = ""


@dataclass
class AnalyzedFeedback:
    id: str
    text: str
    rating: str
    sentiment: str
    themes: list[str] = field(default_factory=list)


def parse_feedback_csv(csv_text: str) -> list[Feedback]:
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        raise FeedbackAnalysisError("CSV appears to be empty")

    lower_fields = {f.lower().strip(): f for f in reader.fieldnames if f and isinstance(f, str)}
    text_col = next(
        (lower_fields[c] for c in ("review_text", "feedback", "comment", "text", "review") if c in lower_fields),
        None,
    )
    if text_col is None:
        raise FeedbackAnalysisError(
            f"CSV must have a 'review_text' (or 'feedback'/'comment'/'text') column. "
            f"Found columns: {reader.fieldnames}"
        )
    id_col = next((lower_fields[c] for c in ("review_id", "id") if c in lower_fields), None)
    rating_col = next((lower_fields[c] for c in ("rating", "score", "stars") if c in lower_fields), None)

    feedback = []
    for i, row in enumerate(reader):
        text = (row.get(text_col) or "").strip()
        extra_fields = row.get(None)
        if extra_fields:
            text = (text + ", " + ", ".join(extra_fields)).strip()
        if not text:
            continue
        row_id = (row.get(id_col) or "").strip() if id_col else ""
        feedback.append(
            Feedback(
                id=row_id or str(i + 1),
                text=text,
                rating=(row.get(rating_col) or "").strip() if rating_col else "",
            )
        )

    if not feedback:
        raise FeedbackAnalysisError("No feedback rows found in CSV")
    return feedback


def chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


ANALYSIS_SCHEMA_HINT = """[
  {
    "id": "must exactly match the id given",
    "sentiment": "one of: positive, neutral, negative",
    "themes": ["1-3 short lowercase tags, e.g. 'pricing', 'customer support', 'bugs', 'onboarding'"]
  }
]"""


def build_analysis_prompt(batch: list[Feedback]) -> str:
    listing = "\n".join(f'{f.id}: "{f.text}"' for f in batch)
    return f"""You are analyzing customer feedback for a product team.

FEEDBACK (id: text):
{listing}

For EVERY item above, classify sentiment and tag it with 1-3 short themes
describing what the feedback is actually about. Use consistent, reusable
theme tags across items (e.g. reuse "pricing" rather than inventing a new
phrase each time) so themes can be aggregated across the whole dataset.

Respond with ONLY a JSON array (no markdown fences, no commentary), one
object per feedback item, in this schema:
{ANALYSIS_SCHEMA_HINT}
"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_json_text(text: str) -> str:
    cleaned = _strip_code_fence(text)
    if not (cleaned.startswith("[") and cleaned.endswith("]")):
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1].strip()
    return cleaned


def parse_analysis_json(text: str, batch: list[Feedback]) -> list[AnalyzedFeedback]:
    cleaned = _extract_json_text(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise FeedbackAnalysisError(f"The model did not return valid JSON: {exc}\n\nRaw response:\n{text}") from exc

    if not isinstance(data, list):
        raise FeedbackAnalysisError("The model's response was valid JSON but not a list")

    by_id = {str(f.id): f for f in batch}
    results = []
    for item in data:
        fid = str(item.get("id", "")).strip()
        source = by_id.get(fid)
        if source is None:
            continue
        sentiment = str(item.get("sentiment", "")).lower().strip()
        sentiment = SENTIMENT_ALIASES.get(sentiment, sentiment)
        if sentiment not in SENTIMENTS:
            raise FeedbackAnalysisError(f"Unexpected sentiment '{sentiment}' for id {fid}")

        raw_themes = item.get("themes", [])
        if isinstance(raw_themes, str):
            themes = [t.lower().strip() for t in raw_themes.split(",") if t.strip()]
        elif isinstance(raw_themes, list):
            themes = [str(t).lower().strip() for t in raw_themes if str(t).strip()]
        else:
            themes = []

        results.append(
            AnalyzedFeedback(
                id=source.id,
                text=source.text,
                rating=source.rating,
                sentiment=sentiment,
                themes=themes,
            )
        )

    found_ids = {r.id for r in results}
    missing = [f.id for f in batch if f.id not in found_ids]
    if missing:
        raise FeedbackAnalysisError(f"The model's response didn't include analysis for ids: {missing}")

    return results


def analyze_feedback(
    csv_text: str,
    call_llm=default_call_llm,
    batch_size: int = 20,
    progress_callback=None,
) -> list[AnalyzedFeedback]:
    feedback = parse_feedback_csv(csv_text)
    results = []
    batches = chunk(feedback, batch_size)
    total_batches = len(batches)
    for i, batch in enumerate(batches):
        if progress_callback:
            progress_callback(i, total_batches)
        prompt = build_analysis_prompt(batch)
        try:
            response_text = call_llm(prompt)
        except Exception as exc:
            if isinstance(exc, FeedbackAnalysisError):
                raise
            raise FeedbackAnalysisError(f"LLM call failed: {exc}") from exc
        results.extend(parse_analysis_json(response_text, batch))
    if progress_callback:
        progress_callback(total_batches, total_batches)
    return results


def mock_call_llm(prompt: str, system: str = "", model: str | None = None, max_tokens: int = 4000) -> str:
    """Offline simulated LLM that inspects prompt reviews and returns realistic JSON tags."""
    import re

    results = []
    # Match patterns like: 1: "The app..."
    for match in re.finditer(r'^(\S+?):\s*"(.*)"\s*$', prompt, re.MULTILINE):
        item_id = match.group(1).strip()
        text = match.group(2).strip()
        lower = text.lower()

        # Determine sentiment
        pos_words = (
            "love", "best", "great", "smooth", "saved", "saves", "fair",
            "impressed", "helpful", "knowledgeable", "easy", "good", "fast"
        )
        neg_words = (
            "crash", "crashed", "crashes", "freezes", "froze", "freeze",
            "slower", "slow", "bug", "bugs", "billed", "duplicate", "lost",
            "unreliable", "frustrating", "embarrassing", "cancelled"
        )

        pos_count = sum(1 for w in pos_words if w in lower)
        neg_count = sum(1 for w in neg_words if w in lower)

        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        # Determine themes
        themes = []
        if any(w in lower for w in ("bug", "crash", "freeze", "froze", "error", "reset")):
            themes.append("bugs & reliability")
        if any(w in lower for w in ("support", "ticket", "chat", "representative", "agent", "service")):
            themes.append("customer support")
        if any(w in lower for w in ("pricing", "bill", "billed", "tier", "charge", "invoice", "cost", "renewal")):
            themes.append("pricing & billing")
        if any(w in lower for w in ("slow", "slower", "performance", "speed", "load", "loading")):
            themes.append("performance")
        if any(w in lower for w in ("onboarding", "setup", "guidance", "checklist")):
            themes.append("onboarding")
        if any(w in lower for w in ("mobile", "app", "notification", "alerts")):
            themes.append("mobile app")
        if any(w in lower for w in ("export", "csv", "report", "sso", "dark mode", "shortcut", "widget", "layout")):
            themes.append("features & ui")

        if not themes:
            themes = ["general feedback"]

        results.append({
            "id": item_id,
            "sentiment": sentiment,
            "themes": themes[:3],
        })

    return json.dumps(results)


def sentiment_counts(analyzed: list[AnalyzedFeedback]) -> dict:
    counts = Counter(a.sentiment for a in analyzed)
    return {s: counts.get(s, 0) for s in SENTIMENTS}


def top_themes(analyzed: list[AnalyzedFeedback], n: int = 10) -> list[tuple[str, int]]:
    counts = Counter(theme for a in analyzed for theme in a.themes)
    return counts.most_common(n)


def example_quote(analyzed: list[AnalyzedFeedback], theme: str) -> str:
    for a in analyzed:
        if theme in a.themes:
            return a.text
    return ""
