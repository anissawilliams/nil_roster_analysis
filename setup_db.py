# setup_db.py — normalized schema
# tables: teams, players, nil_valuations, recruiting_ranks, manual_overrides, transfers

import sqlite3
import pandas as pd

conn = sqlite3.connect("rosterEdge.db")
cur = conn.cursor()

# ── schema ────────────────────────────────────────────────────────────────────

cur.executescript("""
    DROP TABLE IF EXISTS transfers;
    DROP TABLE IF EXISTS manual_overrides;
    DROP TABLE IF EXISTS recruiting_ranks;
    DROP TABLE IF EXISTS nil_valuations;
    DROP TABLE IF EXISTS players;
    DROP TABLE IF EXISTS teams;

    CREATE TABLE teams (
        team_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        school          TEXT NOT NULL UNIQUE,
        conference      TEXT,
        collective_name TEXT
    );

    CREATE TABLE players (
        player_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        team_id      INTEGER NOT NULL REFERENCES teams(team_id),
        first_name   TEXT,
        last_name    TEXT,
        full_name    TEXT,
        position     TEXT,
        depth_role   TEXT,
        jersey       INTEGER,
        year         INTEGER,
        height       INTEGER,
        weight       INTEGER,
        home_city    TEXT,
        home_state   TEXT
    );

    CREATE TABLE nil_valuations (
        valuation_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id         INTEGER NOT NULL REFERENCES players(player_id),
        recruiting_rating REAL,
        predicted_nil     REAL,
        floor             REAL,
        ceiling           REAL
    );

    CREATE TABLE recruiting_ranks (
        rank_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id  INTEGER REFERENCES players(player_id),
        full_name  TEXT,
        rating     REAL,
        source     TEXT,
        year       INTEGER
    );

    CREATE TABLE manual_overrides (
        override_id INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id   INTEGER REFERENCES players(player_id),
        full_name   TEXT,
        rating      REAL,
        reason      TEXT
    );

    CREATE TABLE transfers (
        transfer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        player_id     INTEGER REFERENCES players(player_id),
        first_name    TEXT,
        last_name     TEXT,
        position      TEXT,
        from_team_id  INTEGER REFERENCES teams(team_id),
        to_team_id    INTEGER REFERENCES teams(team_id),
        origin        TEXT,
        destination   TEXT,
        season        INTEGER,
        rating        REAL,
        stars         INTEGER,
        eligibility   TEXT,
        direction     TEXT,
        transfer_date TEXT
    );
""")
print("schema created")

# ── seed known teams ──────────────────────────────────────────────────────────

cur.executemany(
    "INSERT OR IGNORE INTO teams (school, conference) VALUES (?, ?)",
    [("Florida State", "ACC"), ("Georgia", "SEC")]
)
conn.commit()

def get_or_create_team(school_name):
    if not school_name or str(school_name).strip() in ("", "nan"):
        return None
    name = str(school_name).strip()
    row = cur.execute("SELECT team_id FROM teams WHERE school = ?", (name,)).fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO teams (school) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid

fsu_id = get_or_create_team("Florida State")
uga_id = get_or_create_team("Georgia")
print(f"teams seeded: FSU={fsu_id}, UGA={uga_id}")

# ── load players ──────────────────────────────────────────────────────────────

def load_players(csv_path, tid, label):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        fn   = str(r.get("firstName", "") or "").strip()
        ln   = str(r.get("lastName",  "") or "").strip()
        full = f"{fn} {ln}".strip()
        rows.append((
            tid, fn, ln, full,
            r.get("position",  None),
            None,                       # depth_role — set by NIL load
            r.get("jersey",    None),
            r.get("year",      None),
            r.get("height",    None),
            r.get("weight",    None),
            r.get("homeCity",  None),
            r.get("homeState", None),
        ))
    cur.executemany("""
        INSERT INTO players
          (team_id, first_name, last_name, full_name, position, depth_role,
           jersey, year, height, weight, home_city, home_state)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"players ({label}): {len(rows)} rows")

load_players("data/fsu/fsu_roster_2025.csv", fsu_id, "FSU")
load_players("data/uga/uga_roster_2025.csv", uga_id, "UGA")

# ── helper: find player_id ────────────────────────────────────────────────────

def find_player(full_name, tid):
    row = cur.execute(
        "SELECT player_id FROM players WHERE full_name = ? AND team_id = ?",
        (full_name, tid)
    ).fetchone()
    return row[0] if row else None

def get_or_insert_player(full_name, tid):
    pid = find_player(full_name, tid)
    if pid:
        return pid
    cur.execute("INSERT INTO players (team_id, full_name) VALUES (?,?)", (tid, full_name))
    conn.commit()
    return cur.lastrowid

# ── load nil_valuations ───────────────────────────────────────────────────────

def load_nil(csv_path, tid, label):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        full = str(r.get("Full_Name", "") or "").strip()
        pid  = get_or_insert_player(full, tid)
        # backfill depth_role onto player row
        depth = r.get("depth_role", None)
        if depth:
            cur.execute("UPDATE players SET depth_role = ? WHERE player_id = ?", (depth, pid))
        rows.append((
            pid,
            r.get("recruiting_rating", None),
            r.get("predicted_nil",     None),
            r.get("floor",             None),
            r.get("ceiling",           None),
        ))
    cur.executemany("""
        INSERT INTO nil_valuations (player_id, recruiting_rating, predicted_nil, floor, ceiling)
        VALUES (?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"nil_valuations ({label}): {len(rows)} rows")

load_nil("data/fsu/fsu_nil_valuations_final.csv", fsu_id, "FSU")
load_nil("data/uga/uga_nil_valuations_final.csv", uga_id, "UGA")

# ── load recruiting_ranks ─────────────────────────────────────────────────────

recruiting = pd.read_csv("data/training/nil_recruiting_ranks_raw.csv")
rr_rows = []
for _, r in recruiting.iterrows():
    full = str(r.get("Full_Name", r.get("name", "")) or "").strip()
    rr_rows.append((
        None, full,
        r.get("recruiting_rating", r.get("rating", None)),
        r.get("source", "247Sports"),
        r.get("year", None),
    ))
cur.executemany(
    "INSERT INTO recruiting_ranks (player_id, full_name, rating, source, year) VALUES (?,?,?,?,?)",
    rr_rows
)
conn.commit()
print(f"recruiting_ranks: {len(rr_rows)} rows")

# ── load manual_overrides ─────────────────────────────────────────────────────

overrides = pd.read_csv("data/training/manual_recruiting_overrides.csv")
mo_rows = []
for _, r in overrides.iterrows():
    full = str(r.get("Full_Name", r.get("name", "")) or "").strip()
    mo_rows.append((
        None, full,
        r.get("recruiting_rating", r.get("rating", None)),
        r.get("reason", None),
    ))
cur.executemany(
    "INSERT INTO manual_overrides (player_id, full_name, rating, reason) VALUES (?,?,?,?)",
    mo_rows
)
conn.commit()
print(f"manual_overrides: {len(mo_rows)} rows")

# ── load transfers ────────────────────────────────────────────────────────────

def load_transfers(csv_path, home_tid, label):
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        fn   = str(r.get("firstName", "") or "").strip()
        ln   = str(r.get("lastName",  "") or "").strip()
        full = f"{fn} {ln}".strip()
        pid  = find_player(full, home_tid)

        origin      = str(r.get("origin",      "") or "").strip()
        destination = str(r.get("destination", "") or "").strip()
        from_tid    = get_or_create_team(origin)      if origin      else None
        to_tid      = get_or_create_team(destination) if destination else None

        rows.append((
            pid, fn, ln,
            r.get("position",     None),
            from_tid, to_tid,
            origin, destination,        # keep raw strings for easy display
            r.get("season",       None),
            r.get("rating",       None),
            r.get("stars",        None),
            r.get("eligibility",  None),
            r.get("direction",    None),
            r.get("transferDate", None),
        ))
    cur.executemany("""
        INSERT INTO transfers
          (player_id, first_name, last_name, position,
           from_team_id, to_team_id, origin, destination,
           season, rating, stars, eligibility, direction, transfer_date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"transfers ({label}): {len(rows)} rows")

load_transfers("data/fsu/fsu_transfers_2025.csv", fsu_id, "FSU")
load_transfers("data/uga/uga_transfers_2025.csv", uga_id, "UGA")

# ── summary ───────────────────────────────────────────────────────────────────

print("\ndone — rosterEdge.db")
for table in ["teams", "players", "nil_valuations", "recruiting_ranks", "manual_overrides", "transfers"]:
    count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {count} rows")

conn.close()