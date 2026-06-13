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

roster table:
  id, firstName, lastName, team, weight, height (in inches), jersey,
  year (1=FR, 2=SO, 3=JR, 4=SR, 5=GR), position,
  homeCity, homeState, homeCountry

transfers table:
  season, firstName, lastName, position,
  origin (school they left), destination (school they joined),
  transferDate, rating, stars, eligibility,
  direction ('incoming' or 'outgoing' relative to FSU)

nil_valuations table:
  Full_Name, position, year, total_social (Instagram followers),
  depth_role (Starter/Backup/Depth), role_mult,
  base_nil, predicted_nil (estimated NIL value in dollars),
  predicted_nil_fmt (formatted string e.g. '$120,000')

Rules:
- Only write SELECT statements
- Always LIMIT to 20 rows unless user asks for more
- Use LIKE for name searches e.g. firstName LIKE '%Ja%'
- For full name searches use firstName || ' ' || lastName
- Keep the SQL simple and readable
- The team name in the roster table is 'Florida State' not 'FSU'

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