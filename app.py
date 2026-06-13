import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sqlite3
from ask_data import render_ask_tab

st.set_page_config(page_title="RosterEdge", page_icon="🏈", layout="wide")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "rosterEdge.db")

GARNET = "#782F40"
GOLD   = "#CEB888"
RED    = "#BA0C2F"   # georgia red
BLACK  = "#000000"   # georgia black

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

    nil = pd.read_sql("""
        SELECT Full_Name as name, position, predicted_nil as nil_value,
               depth_role as nil_source
        FROM nil_valuations
    """, conn)

    transfers = pd.read_sql("SELECT * FROM transfers", conn)
    transfers["name"] = transfers["firstName"] + " " + transfers["lastName"]
    transfers["transferDate"] = pd.to_datetime(transfers["transferDate"]).dt.strftime("%Y-%m-%d")

    uga_roster = pd.read_sql("SELECT * FROM uga_roster", conn)
    uga_roster["name"]           = uga_roster["firstName"] + " " + uga_roster["lastName"]
    uga_roster["class"]          = uga_roster["year"].map(YEAR_MAP).fillna("Other")
    uga_roster["height_display"] = uga_roster["height"].apply(inches_to_feet)
    uga_roster["hometown"]       = uga_roster["homeCity"] + ", " + uga_roster["homeState"]

    uga_nil = pd.read_sql("""
        SELECT Full_Name as name, position, predicted_nil as nil_value,
               depth_role as nil_source
        FROM uga_nil_valuations
    """, conn)

    uga_transfers = pd.read_sql("SELECT * FROM uga_transfers", conn)
    uga_transfers["name"] = uga_transfers["firstName"] + " " + uga_transfers["lastName"]
    uga_transfers["transferDate"] = pd.to_datetime(uga_transfers["transferDate"]).dt.strftime("%Y-%m-%d")

    comparison = pd.read_sql("SELECT * FROM school_comparison", conn)

    conn.close()
    return roster, nil, transfers, uga_roster, uga_nil, uga_transfers, comparison

roster, nil, transfers, uga_roster, uga_nil, uga_transfers, comparison = load_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='color:#782F40;margin-bottom:0'>🏈 RosterEdge</h1>"
    "<p style='color:#666;margin-top:0'>NCAA College Football Roster Intelligence · 2025</p>",
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Roster",
    "💰 NIL Valuations",
    "🔄 Transfer Portal",
    "🏫 School Comparison",
    "🔮 Ask the Data"
])

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
    nil_school = st.selectbox("Select School", ["Florida State", "Georgia"], key="nil_school")
    nil_data_sel = nil if nil_school == "Florida State" else uga_nil
    nil_color = GARNET if nil_school == "Florida State" else RED
    school_label = "Florida State Seminoles" if nil_school == "Florida State" else "Georgia Bulldogs"

    st.subheader(f"NIL Valuations — {school_label}")
    st.caption("Model-predicted NIL valuations based on position, recruiting rating, and roster role")

    nil_sorted = nil_data_sel.sort_values("nil_value", ascending=False).reset_index(drop=True)

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
        ax2.barh(top10["name"][::-1], top10["nil_value"][::-1], color=nil_color)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax2.set_title("Top 10 NIL Values")
        ax2.tick_params(labelsize=8)
        plt.tight_layout()
        st.pyplot(fig2)

    with r2:
        st.markdown("**NIL Value by Position**")
        pos_nil = nil.groupby("position")["nil_value"].sum().sort_values(ascending=False)
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.bar(pos_nil.index, pos_nil.values, color=GOLD, edgecolor=nil_color)
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        ax3.set_title("Total NIL Value by Position")
        ax3.tick_params(axis="x", rotation=45, labelsize=8)
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("**Full NIL Table**")
    st.dataframe(
        nil_sorted[["name","position","nil_value","nil_source"]]
            .rename(columns={"nil_value":"NIL Value ($)","nil_source":"Depth Role"}),
        use_container_width=True, hide_index=True, height=500,
    )

#### TRANSFER PORTAL TAB ###
with tab3:
    transfer_school = st.selectbox("Select School", ["Florida State", "Georgia"], key="transfer_school")
    transfers_sel = transfers if transfer_school == "Florida State" else uga_transfers
    transfer_color = GARNET if transfer_school == "Florida State" else RED
    transfer_label = "Florida State Seminoles" if transfer_school == "Florida State" else "Georgia Bulldogs"

    st.subheader(f"2025 Transfer Portal Activity — {transfer_label}")

    incoming = transfers_sel[transfers_sel["direction"] == "Incoming"]
    outgoing = transfers_sel[transfers_sel["direction"] == "Outgoing"]

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
    ax4.bar(out_pos.index, out_pos.values, color=transfer_color)
    ax4.set_ylabel("Players")
    ax4.set_title("Outgoing Transfer Losses by Position")
    st.pyplot(fig4)

    st.markdown("#### Outgoing by Rating")
    out_stars = outgoing["stars"].value_counts()
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    ax5.bar(out_stars.index, out_stars.values, color=GOLD)
    ax5.set_ylabel("Players")
    ax5.set_title("Outgoing Transfer Losses by Stars")
    st.pyplot(fig5)

#### SCHOOL COMPARISON TAB ####
with tab4:
    st.subheader("🏫 School Comparison — Florida State vs Georgia")
    st.caption("Predicted NIL market value compared across both programs using the same model")

    fsu_data = comparison[comparison["school"] == "Florida State"]
    uga_data = comparison[comparison["school"] == "Georgia"]

    # top level metrics side by side
    col_fsu, col_uga = st.columns(2)

    with col_fsu:
        st.markdown("### 🟥 Florida State")
        st.metric("Total Roster NIL",   f"${fsu_data['nil_value'].sum():,.0f}")
        st.metric("Avg per Player",     f"${fsu_data['nil_value'].mean():,.0f}")
        st.metric("Median per Player",  f"${fsu_data['nil_value'].median():,.0f}")
        st.metric("Roster Size",        len(fsu_data))

    with col_uga:
        st.markdown("### ⬛ Georgia")
        st.metric("Total Roster NIL",   f"${uga_data['nil_value'].sum():,.0f}")
        st.metric("Avg per Player",     f"${uga_data['nil_value'].mean():,.0f}")
        st.metric("Median per Player",  f"${uga_data['nil_value'].median():,.0f}")
        st.metric("Roster Size",        len(uga_data))

    st.divider()

    # position median comparison
    st.markdown("#### NIL Market Value by Position")
    st.caption("Median predicted NIL value per position — same model, different roster compositions")

    fsu_pos = fsu_data.groupby("position")["nil_value"].median().rename("Florida State")
    uga_pos = uga_data.groupby("position")["nil_value"].median().rename("Georgia")
    pos_compare = pd.concat([fsu_pos, uga_pos], axis=1).fillna(0)
    pos_compare = pos_compare.sort_values("Florida State", ascending=False)

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    x = range(len(pos_compare))
    width = 0.35
    ax6.bar([i - width/2 for i in x], pos_compare["Florida State"], width, label="Florida State", color=GARNET)
    ax6.bar([i + width/2 for i in x], pos_compare["Georgia"],        width, label="Georgia",        color=RED)
    ax6.set_xticks(list(x))
    ax6.set_xticklabels(pos_compare.index, rotation=45, ha="right", fontsize=9)
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax6.set_title("Median NIL Value by Position: FSU vs Georgia")
    ax6.legend()
    plt.tight_layout()
    st.pyplot(fig6)

    st.divider()

    # recruiting rating comparison
    st.markdown("#### Recruiting Talent Comparison")
    st.caption("Average recruiting rating by position — higher = more highly recruited players at that position")

    fsu_rec = fsu_data.groupby("position")["recruiting_rating"].mean().rename("Florida State")
    uga_rec = uga_data.groupby("position")["recruiting_rating"].mean().rename("Georgia")
    rec_compare = pd.concat([fsu_rec, uga_rec], axis=1).fillna(0)
    rec_compare = rec_compare.sort_values("Georgia", ascending=False)

    fig7, ax7 = plt.subplots(figsize=(10, 5))
    x2 = range(len(rec_compare))
    ax7.bar([i - width/2 for i in x2], rec_compare["Florida State"], width, label="Florida State", color=GARNET)
    ax7.bar([i + width/2 for i in x2], rec_compare["Georgia"],        width, label="Georgia",        color=RED)
    ax7.set_xticks(list(x2))
    ax7.set_xticklabels(rec_compare.index, rotation=45, ha="right", fontsize=9)
    ax7.set_ylim(0.7, 1.0)
    ax7.set_title("Avg Recruiting Rating by Position: FSU vs Georgia")
    ax7.legend()
    plt.tight_layout()
    st.pyplot(fig7)

    st.divider()

    # starter comparison - top predicted value starters side by side
    st.markdown("#### Top Starters by Predicted NIL Value")
    left3, right3 = st.columns(2)

    with left3:
        st.markdown("**Florida State**")
        fsu_starters = fsu_data[fsu_data["depth_role"] == "starter"] \
            .sort_values("nil_value", ascending=False).head(10)
        st.dataframe(
            fsu_starters[["name","position","recruiting_rating","nil_value"]]
                .rename(columns={"nil_value":"NIL Est.", "recruiting_rating":"Recruit Rating"}),
            use_container_width=True, hide_index=True
        )

    with right3:
        st.markdown("**Georgia**")
        uga_starters = uga_data[uga_data["depth_role"] == "starter"] \
            .sort_values("nil_value", ascending=False).head(10)
        st.dataframe(
            uga_starters[["name","position","recruiting_rating","nil_value"]]
                .rename(columns={"nil_value":"NIL Est.", "recruiting_rating":"Recruit Rating"}),
            use_container_width=True, hide_index=True
        )

with tab5:
    render_ask_tab()

st.divider()