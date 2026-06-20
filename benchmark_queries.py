# benchmarks query performance for the database section of the report
# tests flat query vs join, then join with an index added

import sqlite3
import pandas as pd
import time

conn = sqlite3.connect("rosterEdge.db")
cur = conn.cursor()

N = 200  # number of times to run each query for the average

def time_query(sql, n=N):
    t0 = time.perf_counter()
    for _ in range(n):
        df = pd.read_sql(sql, conn)
    t1 = time.perf_counter()
    return (t1 - t0) / n * 1000  # ms

# query 1 - flat read, no join
flat_sql = "SELECT * FROM roster"
flat_time = time_query(flat_sql)

# query 2 - join roster + nil_valuations, no index yet
join_sql = """
    SELECT r.Full_Name, r.position, n.predicted_nil
    FROM roster r
    JOIN nil_valuations n ON r.Full_Name = n.Full_Name
"""
join_time_before = time_query(join_sql)

# add indexes on the join column
cur.execute("CREATE INDEX IF NOT EXISTS idx_roster_name ON roster(Full_Name)")
cur.execute("CREATE INDEX IF NOT EXISTS idx_nil_name ON nil_valuations(Full_Name)")
conn.commit()

# query 3 - same join, now indexed
join_time_after = time_query(join_sql)

print(f"Flat query (roster only):        {flat_time:.3f} ms avg over {N} runs")
print(f"Join, no index:                  {join_time_before:.3f} ms avg over {N} runs")
print(f"Join, indexed:                   {join_time_after:.3f} ms avg over {N} runs")
print(f"Improvement from indexing:       {(join_time_before - join_time_after) / join_time_before * 100:.1f}%")

conn.close()
