import sqlite3
import pandas as pd

conn = sqlite3.connect("rosterEdge.db")

# check for orphaned or duplicate player_id joins
check = pd.read_sql("""
    SELECT p.player_id, p.full_name, p.position, p.depth_role, t.school,
           n.predicted_nil, n.recruiting_rating
    FROM players p
    JOIN teams t ON p.team_id = t.team_id
    LEFT JOIN nil_valuations n ON p.player_id = n.player_id
    WHERE p.full_name LIKE '%Sperry%'
""", conn)
print(check)