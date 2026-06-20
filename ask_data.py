# ask_data.py
from dotenv import load_dotenv
load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────
LLM_PROVIDER = "claude"
DB_PATH       = "rosterEdge.db"
# ────────────────────────────────────────────────────────────

import sqlite3
import requests
import streamlit as st
import pandas as pd

if LLM_PROVIDER == "claude":
    import anthropic

SCHEMA_CONTEXT = """
You are a SQL assistant for a college football roster database called RosterEdge.
The SQLite database has these tables:

teams table:
  team_id (PK), school (e.g. 'Florida State', 'Georgia'), conference, collective_name

players table:
  player_id (PK), team_id (FK -> teams), first_name, last_name, full_name,
  position, depth_role (starter/backup/depth),
  jersey, year (1=FR 2=SO 3=JR 4=SR 5=GR), height (inches), weight,
  home_city, home_state

nil_valuations table:
  valuation_id (PK), player_id (FK -> players),
  recruiting_rating (247Sports composite, 0-1 scale),
  predicted_nil (estimated NIL value in dollars),
  floor, ceiling

recruiting_ranks table:
  rank_id (PK), player_id (FK -> players, nullable),
  full_name, rating, source, year

manual_overrides table:
  override_id (PK), player_id (FK -> players, nullable),
  full_name, rating, reason

transfers table:
  transfer_id (PK), player_id (FK -> players, nullable),
  first_name, last_name, position,
  from_team_id (FK -> teams), to_team_id (FK -> teams),
  origin (raw school name they left), destination (raw school name they joined),
  season, rating, stars, eligibility,
  direction ('Incoming' or 'Outgoing' relative to the home school),
  transfer_date

Common join patterns:
  -- player NIL values with names:
  SELECT p.full_name, p.position, n.predicted_nil
  FROM nil_valuations n JOIN players p ON n.player_id = p.player_id
  JOIN teams t ON p.team_id = t.team_id WHERE t.school = 'Florida State'

  -- FSU roster:
  SELECT p.first_name, p.last_name, p.position, p.year
  FROM players p JOIN teams t ON p.team_id = t.team_id
  WHERE t.school = 'Florida State'

  -- incoming transfers to FSU:
  SELECT t.first_name, t.last_name, t.position, t.origin, t.rating
  FROM transfers t JOIN teams tm ON t.to_team_id = tm.team_id
  WHERE tm.school = 'Florida State' AND t.direction = 'Incoming'

Rules:
- Only write SELECT statements
- Always LIMIT to 20 rows unless user asks for more
- Use LIKE for name searches e.g. first_name LIKE '%Ja%'
- For full name searches use p.full_name LIKE '%name%'
- Keep SQL simple and readable
- Always JOIN through teams to filter by school — never assume team_id values

Respond with ONLY a JSON object like this:
{
  "sql": "SELECT ...",
  "explanation": "plain English explanation of what this query does"
}
"""


# def generate_sql_ollama(question: str) -> dict:
#     prompt = f"{SCHEMA_CONTEXT}\n\nQuestion: {question}"
#     response = requests.post(
#         OLLAMA_URL,
#         json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
#         timeout=30
#     )
#     response.raise_for_status()
#     raw = response.json().get("response", "")
#     # strip markdown fences if present
#     if "```" in raw:
#         raw = raw.split("```")[1]
#         if raw.startswith("json"):
#             raw = raw[4:]
#     import json
#     return json.loads(raw.strip())


def generate_sql_claude(question: str) -> dict:
    import json
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"{SCHEMA_CONTEXT}\n\nQuestion: {question}"
        }]
    )
    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def generate_sql(question: str) -> dict:
    # if LLM_PROVIDER == "ollama":
    #     return generate_sql_ollama(question)
    # else:
        return generate_sql_claude(question)


def is_safe_sql(sql: str) -> bool:
    cleaned = sql.strip().upper()
    if not cleaned.startswith("SELECT"):
        return False
    for word in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE"]:
        if word in cleaned:
            return False
    return True


def run_query(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


# ── STREAMLIT UI ─────────────────────────────────────────────

def render_ask_tab():
    st.header("Ask the Data")
    st.caption(f"Powered by 'Claude Haiku'")

    # example questions
    st.markdown("**Try asking:**")
    examples = [
        "Show me all quarterbacks on the 2025 FSU roster",
        "Which players have the highest estimated NIL value?",
        "Who transferred out of FSU this season?",
        "List 4-star or higher recruits by position",
        "Which positions have the most transfer portal activity?",
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"ex_{i}"):
            st.session_state["ask_query"] = example

    st.divider()

    question = st.text_input(
        "Ask a question about the roster:",
        value=st.session_state.get("ask_query", ""),
        placeholder="e.g. Show me seniors with high NIL valuations",
        key="ask_input"
    )

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Thinking..."):
            try:
                result = generate_sql(question)
                sql = result.get("sql", "")
                explanation = result.get("explanation", "")

                if not is_safe_sql(sql):
                    st.error("Generated an unsafe query — only SELECT statements are allowed.")
                    return

                df = run_query(sql)

                # answer
                st.success(explanation)

                # results
                if df.empty:
                    st.info("No results found for that query.")
                else:
                    st.dataframe(df, use_container_width=True)
                    st.caption(f"{len(df)} row(s) returned")

                # show sql in expander — good for academic transparency
                with st.expander("See generated SQL"):
                    st.code(sql, language="sql")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to Ollama. Make sure it's running: `ollama serve`")
            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                with st.expander("Error details"):
                    st.exception(e)


# ── if running standalone ────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(page_title="RosterEdge — Ask the Data", layout="wide")
    render_ask_tab()