import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sqlite3
from ask_data import render_ask_tab
st.set_page_config(page_title="RosterEdge – FSU", page_icon="🏈", layout="wide")


import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rosterEdge.db")


GARNET = "#782F40"
GOLD   = "#CEB888"

YEAR_MAP = {1: "Freshman", 2: "Sophomore", 3: "Junior", 4: "Senior", 5: "Grad"}

def inches_to_feet(n):
    try:
        n = int(n)
        return f"{n // 12}'{n % 12}\""
    except:
        return "N/A"

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_PATH)

    roster = pd.read_sql("SELECT * FROM roster", conn)
    roster["name"]           = roster["firstName"] + " " + roster["lastName"]
    roster["class"]          = roster["year"].map(YEAR_MAP).fillna("Other")
    roster["height_display"] = roster["height"].apply(inches_to_feet)
    roster["hometown"]       = roster["homeCity"] + ", " + roster["homeState"]

    # alias columns to match what the rest of the app expects
    nil = pd.read_sql("""
        SELECT Full_Name as name, position, predicted_nil as nil_value,
               depth_role as nil_source
        FROM nil_valuations
    """, conn)

    transfers = pd.read_sql("SELECT * FROM transfers", conn)
    transfers["name"] = transfers["firstName"] + " " + transfers["lastName"]
    transfers["transferDate"] = pd.to_datetime(transfers["transferDate"]).dt.strftime("%Y-%m-%d")

    conn.close()
    return roster, nil, transfers

roster, nil, transfers = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#782F40;margin-bottom:0'>🏈 RosterEdge</h1>"
    "<p style='color:#666;margin-top:0'>Florida State Seminoles · 2025 Season</p>",
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📋 Roster", "💰 NIL Valuations", "🔄 Transfer Portal", "🔮 Ask the Data"])

##### ROSTER TAB #####
with tab1:
    st.subheader(f"2025 FSU Roster  ({len(roster)} players)")

    c1, c2, c3 = st.columns(3)
    with c1:
        positions = ["All"] + sorted(roster["position"].dropna().unique().tolist())
        sel_pos = st.selectbox("Position", positions)
    with c2:
        classes = ["All", "Freshman", "Sophomore", "Junior", "Senior", "Grad"]
        sel_class = st.selectbox("Class Year", classes)
    with c3:
        search = st.text_input("Search player name")

    f = roster.copy()
    if sel_pos   != "All": f = f[f["position"] == sel_pos]
    if sel_class != "All": f = f[f["class"]    == sel_class]
    if search:             f = f[f["name"].str.contains(search, case=False, na=False)]

    m1, m2, m3 = st.columns(3)
    m1.metric("Players", len(f))
    m2.metric("Positions", f["position"].nunique())
    m3.metric("Home States", f["homeState"].nunique())

    left, right = st.columns([2, 1])
    with left:
        st.dataframe(
            f[["jersey","name","position","class","hometown","height_display","weight"]]
             .rename(columns={"jersey":"#","height_display":"Height","class":"Year"}),
            use_container_width=True, hide_index=True, height=600,
        )
    with right:
        pos_counts = f["position"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 5))
        ax.barh(pos_counts.index, pos_counts.values, color=GARNET)
        ax.set_xlabel("Players")
        ax.set_title("By Position")
        for i, v in enumerate(pos_counts.values):
            ax.text(v + 0.05, i, str(v), va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

#### NIL VALUATIONS TAB ####
with tab2:
    st.subheader("NIL Valuations")
    st.caption("Model-predicted NIL valuations based on position, social following, and roster role")

    nil_sorted = nil.sort_values("nil_value", ascending=False).reset_index(drop=True)

    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Total Roster NIL Value", f"${nil_sorted['nil_value'].sum():,.0f}")
    n2.metric("Avg per Player",         f"${nil_sorted['nil_value'].mean():,.0f}")
    n3.metric("Highest Value",          f"${nil_sorted['nil_value'].max():,.0f}")
    n4.metric("Players Tracked",        len(nil_sorted))

    l2, r2 = st.columns(2)
    with l2:
        st.markdown("**Top 10 by NIL Value**")
        top10 = nil_sorted.head(10)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.barh(top10["name"][::-1], top10["nil_value"][::-1], color=GARNET)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax2.set_title("Top 10 NIL Values")
        ax2.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    with r2:
        st.markdown("**NIL Value by Position**")
        pos_nil = nil.groupby("position")["nil_value"].sum().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.bar(pos_nil.index, pos_nil.values, color=GOLD, edgecolor=GARNET)
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax3.set_title("Total NIL Value by Position")
        ax3.tick_params(axis="x", rotation=45, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("**Full NIL Table**")
    st.dataframe(
        nil_sorted[["name","position","nil_value","nil_source"]]
            .rename(columns={"nil_value":"NIL Value ($)","nil_source":"Source"}),
        use_container_width=True, hide_index=True, height=500,
    )

#### TRANSFER PORTAL TAB ###
with tab3:
    st.subheader("2025 Transfer Portal Activity")

    incoming = transfers[transfers["direction"] == "Incoming"]
    outgoing = transfers[transfers["direction"] == "Outgoing"]

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Incoming",    len(incoming))
    t2.metric("Outgoing",    len(outgoing))
    t3.metric("Net Change",  f"{len(incoming) - len(outgoing):+d}")
    t4.metric("Avg Incoming Rating", f"{incoming['rating'].mean():.2f}" if incoming['rating'].notna().any() else "N/A")

    st.markdown("#### ✅ Incoming Transfers")
    st.dataframe(
        incoming[["name","position","origin","transferDate","eligibility","rating","stars"]]
            .rename(columns={"origin":"From","transferDate":"Date","eligibility":"Eligibility"}),
        use_container_width=True, hide_index=True, height=500,
    )

    st.markdown("#### 🚪 Outgoing Transfers")
    st.dataframe(
        outgoing[["name","position","destination","transferDate","eligibility","rating","stars"]]
            .rename(columns={"destination":"To","transferDate":"Date","eligibility":"Eligibility"}),
        use_container_width=True, hide_index=True, height=600,
    )

    st.markdown("#### Outgoing by Position")
    out_pos = outgoing["position"].value_counts()
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.bar(out_pos.index, out_pos.values, color=GARNET)
    ax4.set_ylabel("Players")
    ax4.set_title("Outgoing Transfer Losses by Position")
    st.pyplot(fig4)

    st.markdown("#### Outgoing by Rating")
    out_pos = outgoing["stars"].value_counts()
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.bar(out_pos.index, out_pos.values, color=GOLD)
    ax4.set_ylabel("Players")
    ax4.set_title("Outgoing Transfer Losses by Stars")
    st.pyplot(fig4)

with tab4:
    render_ask_tab()

st.divider()
