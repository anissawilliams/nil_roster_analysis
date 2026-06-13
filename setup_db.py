# sets up the SQLite database and loads CSVs into tables
# final milestone - added recruiting ranks, UGA data, and cross-school comparison

import sqlite3
import pandas as pd

conn = sqlite3.connect("rosterEdge.db")

# ── FSU data (same as intermediate) ──────────────────────────────────────────
roster = pd.read_csv("data/fsu/fsu_roster_2025.csv")
roster.to_sql("roster", conn, if_exists="replace", index=False)
print(f"roster: {len(roster)} rows")

nil = pd.read_csv("data/fsu/fsu_nil_valuations_final.csv")
nil.to_sql("nil_valuations", conn, if_exists="replace", index=False)
print(f"nil_valuations: {len(nil)} rows")

transfers = pd.read_csv("data/fsu/fsu_transfers_2025.csv")
transfers.to_sql("transfers", conn, if_exists="replace", index=False)
print(f"transfers: {len(transfers)} rows")

# ── NEW: recruiting ranks from nil_data pull ──────────────────────────────────
recruiting = pd.read_csv("data/training/nil_recruiting_ranks_raw.csv")
recruiting.to_sql("recruiting_ranks", conn, if_exists="replace", index=False)
print(f"recruiting_ranks: {len(recruiting)} rows")

# ── NEW: manual overrides - good to have in db for transparency ───────────────
overrides = pd.read_csv("data/training/manual_recruiting_overrides.csv")
overrides.to_sql("manual_overrides", conn, if_exists="replace", index=False)
print(f"manual_overrides: {len(overrides)} rows")

# ── NEW: UGA roster and predictions ──────────────────────────────────────────
uga_roster = pd.read_csv("data/uga/uga_roster_2025.csv")
uga_roster.to_sql("uga_roster", conn, if_exists="replace", index=False)
print(f"uga_roster: {len(uga_roster)} rows")

uga_nil = pd.read_csv("data/uga/uga_nil_valuations_final.csv")
uga_nil.to_sql("uga_nil_valuations", conn, if_exists="replace", index=False)
print(f"uga_nil_valuations: {len(uga_nil)} rows")

uga_transfers = pd.read_csv("data/uga/uga_transfers_2025.csv")
uga_transfers.to_sql("uga_transfers", conn, if_exists="replace", index=False)
print(f"uga_transfers: {len(uga_transfers)} rows")

# ── NEW: pre-built cross-school comparison table ──────────────────────────────
# tag each school and combine nil valuations for comparison queries
# this is what powers the School Comparison tab in app.py

fsu_compare = nil[['Full_Name', 'position', 'depth_role',
                    'recruiting_rating', 'predicted_nil']].copy()
fsu_compare['school'] = 'Florida State'
fsu_compare = fsu_compare.rename(columns={
    'Full_Name': 'name',
    'predicted_nil': 'nil_value'
})

uga_compare = uga_nil[['Full_Name', 'position', 'depth_role',
                        'recruiting_rating', 'predicted_nil']].copy()
uga_compare['school'] = 'Georgia'
uga_compare = uga_compare.rename(columns={
    'Full_Name': 'name',
    'predicted_nil': 'nil_value'
})

comparison = pd.concat([fsu_compare, uga_compare], ignore_index=True)
comparison.to_sql("school_comparison", conn, if_exists="replace", index=False)
print(f"school_comparison: {len(comparison)} rows ({len(fsu_compare)} FSU + {len(uga_compare)} UGA)")

conn.close()
print("\ndone - rosterEdge.db created with all tables")
print("tables: roster, nil_valuations, transfers, recruiting_ranks,")
print("        manual_overrides, uga_roster, uga_nil_valuations,")
print("        uga_transfers, school_comparison")
