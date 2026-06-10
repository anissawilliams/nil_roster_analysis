# sets up the SQLite database and loads CSVs into tables

import sqlite3
import pandas as pd

conn = sqlite3.connect("rosterEdge.db")

# load CSVs into tables
roster = pd.read_csv("data/fsu/fsu_roster_2025.csv")
roster.to_sql("roster", conn, if_exists="replace", index=False)
print(f"roster: {len(roster)} rows")

nil = pd.read_csv("data/fsu/fsu_nil_valuations_final.csv")
nil.to_sql("nil_valuations", conn, if_exists="replace", index=False)
print(f"nil_valuations: {len(nil)} rows")

transfers = pd.read_csv("data/fsu/fsu_transfers_2025.csv")
transfers.to_sql("transfers", conn, if_exists="replace", index=False)
print(f"transfers: {len(transfers)} rows")

conn.close()
print("done - rosterEdge.db created")