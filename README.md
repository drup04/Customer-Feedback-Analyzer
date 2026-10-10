# Customer Feedback Analyzer

An AI-powered product intelligence application built with **Streamlit** and **OpenAI**. Transform raw customer feedback (support tickets, app reviews, customer survey exports) into structured, actionable roadmap insights with sentiment analysis, normalized theme clusters, direct evidence quotes, and interactive exploration.

---

## Key Features

- **Automated Sentiment Classification**: Identifies positive, neutral, and negative sentiment distribution with direct metric KPIs and visual charts.
- **Normalized Theme Extraction**: Groups feedback into consistent, actionable product themes (e.g., *bugs & reliability*, *pricing & billing*, *customer support*, *onboarding*, *performance*).
- **Direct Evidence & Quotes**: Generates quote cards linking every extracted theme directly back to real customer quotes.
- **Interactive Review Explorer**: Search and filter reviews by sentiment, theme, or keywords.
- **Enriched Dataset Export**: Export the complete analyzed dataset with model-generated sentiment and theme tags as a CSV file.
- **Dual AI Engine**:
  - **Live OpenAI Mode**: Connect with an API key using fast, cost-effective models (`gpt-4o-mini`, `gpt-4o`, `gpt-3.5-turbo`).
  - **Demo Simulation Mode**: Fast offline processing for instant testing and evaluation without API costs.

---

## Architecture & Pipeline

```text
app.py (Streamlit UI: Upload, Metrics, Charts, Explorer, Export)
   └── feedback_logic.py (CSV Parsing, Batching, Schema Validation, Aggregation)
         └── llm_client.py (OpenAI Client Wrapper)
               └── OpenAI API / Offline Simulation
```

1. **Robust CSV Ingestion**: Tolerant of column name variations (`review_text`, `feedback`, `comment`, `text`, `review`) and safely recovers unquoted overflow fields.
2. **Batch Prompting**: Batches feedback rows to maintain tag consistency and optimize token usage.
3. **Structured Schema Validation**: Validates JSON output strictly against sentiment vocabulary and feedback IDs.
4. **Deterministic Aggregation**: Pure Python aggregation using `Counter` to calculate sentiment counts and rank top themes.

---

## Quickstart

### 1. Clone the Repository

```bash
git clone https://github.com/drup04/Customer-Feedback-Analyzer.git
cd Customer-Feedback-Analyzer
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment (Optional for Live Mode)

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

> You can also enter your OpenAI API key directly in the application's sidebar, or select **Demo Simulation** to run completely offline without an API key.

### 5. Run the Application

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) (or the displayed port) in your browser.

---

## Running Unit Tests

All pure logic, prompt building, and parsing functions are covered by unit tests with mocked responses (no external API calls required):

```bash
pytest
```

---

## License

MIT License.
