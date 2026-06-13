import requests
import pandas as pd
import time
import requests
import pandas as pd
import os

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("CFBD_API_KEY")

# -----------------------------------------------
# run 1: nil_data.csv
# run 2: fsu_roster_2025.csv
# run 3: uga_roster_2025.csv
#INPUT_FILE = "data/fsu/fsu_roster_2025.csv"
INPUT_FILE = "data/training/nil_data.csv"
OUTPUT_FILE = "data/training/uga_recruiting_ranks_raw.csv"
# -----------------------------------------------

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json"
}

# load the NIL data
nil_df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(nil_df)} players from {INPUT_FILE}")

# -----------------------------------------------
# PASS 1: check transfer portal data first
# transfer portal already has rating and stars - no API call needed
# -----------------------------------------------
print("\n--- PASS 1: checking transfer portal data ---")

# load both transfer files
# NOTE: transfer data has firstName + lastName separate, need to combine
fsu_transfers = pd.read_csv("data/fsu/fsu_transfers_2025.csv")
uga_transfers = pd.read_csv("data/uga/uga_transfers_2025.csv")

# combine them - we might need both
all_transfers = pd.concat([fsu_transfers, uga_transfers], ignore_index=True)

# make a full name column so we can match
all_transfers["Player"] = all_transfers["firstName"] + " " + all_transfers["lastName"]


print(f"Total transfer players loaded: {len(all_transfers)}")
print(all_transfers[["Player", "rating", "stars"]].head())

# build a lookup dict from the transfer data - just name -> rating/stars
# yes a dict is fine here
transfer_lookup = {}
for i, row in all_transfers.iterrows():
    name = row["Player"]
    transfer_lookup[name] = {
        "recruiting_rating": row["rating"],
        "recruiting_stars": row["stars"],
        "source": "transfer_portal"
    }

print(f"\nTransfer lookup has {len(transfer_lookup)} players")

# -----------------------------------------------
# PASS 2: for anyone NOT in transfer data, hit the recruiting API
# trying years 2021-2024 because thats when most of these guys were recruited
# -----------------------------------------------
print("\n--- PASS 2: hitting CFBD recruiting API for non-transfer players ---")
#uncomment this out for the individual school file rosters
#nil_df["Player"] = nil_df["firstName"] + " " + nil_df["lastName"]
RECRUITING_YEARS = [2026, 2025, 2024, 2023, 2022, 2021, 2020]

results = []

for i, row in nil_df.iterrows():
    player_name = row["Player"]
    school = row.get("team", row.get("School", "unknown"))
    position = row["Position"]
    #position = row["position"]
    print(f"\n[{i+1}/{len(nil_df)}] {player_name} ({school})")

    # check transfer portal first - free and fast
    if player_name in transfer_lookup:
        t = transfer_lookup[player_name]
        print(f"   found in transfer portal: rating={t['recruiting_rating']}, stars={t['recruiting_stars']}")
        results.append({
            "Player": player_name,
            "School": school,
            "Position": position,
            "recruiting_rating": t["recruiting_rating"],
            "recruiting_stars": t["recruiting_stars"],
            "recruiting_rank": None,  # portal data doesnt have national rank
            "source": "transfer_portal",
            "match_found": True
        })
        continue  # skip the API call

    # not in transfer data - try the recruiting endpoint year by year
    found = False
    for year in RECRUITING_YEARS:
        url = "https://api.collegefootballdata.com/recruiting/players"
        params = {
            "search": player_name,
            "year": year
        }

        try:
            response = requests.get(url, headers=headers, params=params)

            if response.status_code != 200:
                print(f"   ERROR {response.status_code} for year {year}")
                time.sleep(0.5)
                continue

            data = response.json()

            if not data:
                # nothing for this year, try next year
                continue

            # look for a name match in the results
            best_match = None
            for recruit in data:
                recruit_name = recruit.get("name", "")
                if recruit_name.lower() == player_name.lower():
                    best_match = recruit
                    break

            # no exact match - just go with null
            if best_match is None:
                print(f"   no exact name match in {year}, skipping")
                time.sleep(0.3)
                continue

            rating = best_match.get("rating")
            stars = best_match.get("stars")
            rank = best_match.get("ranking")

            # sometimes CFBD returns a record but rating is 0 or None - skip those
            if not rating or rating == 0:
                print(f"   year {year}: record found but rating is empty, trying next year")
                time.sleep(0.3)
                continue

            print(f"   FOUND in {year}: rank={rank}, stars={stars}, rating={rating}")
            results.append({
                "Player": player_name,
                "School": school,
                "Position": position,
                "recruiting_rating": rating,
                "recruiting_stars": stars,
                "recruiting_rank": rank,
                "source": f"cfbd_{year}",
                "match_found": True
            })
            found = True
            break  # got what we need, stop looping years

        except Exception as e:
            print(f"   EXCEPTION year {year}: {e}")

        time.sleep(0.3)

    if not found:
        print(f"   no match found in any year :(")
        results.append({
            "Player": player_name,
            "School": school,
            "Position": position,
            "recruiting_rating": None,
            "recruiting_stars": None,
            "recruiting_rank": None,
            "source": "not_found",
            "match_found": False
        })

    # slight pause between players
    time.sleep(0.5)

# -----------------------------------------------
# save and merge
# -----------------------------------------------
print("\n\n--- RESULTS ---")
recruiting_df = pd.DataFrame(results)
recruiting_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved raw results to {OUTPUT_FILE}")
print(f"Total: {len(recruiting_df)}")
print(f"Matched: {recruiting_df['match_found'].sum()}")
print(f"Not found: {(~recruiting_df['match_found']).sum()}")
print(f"\nBy source:")
print(recruiting_df["source"].value_counts())

# merge back onto original
merged = nil_df.merge(
    recruiting_df[["Player", "recruiting_rating", "recruiting_stars", "recruiting_rank", "source"]],
    on="Player",
    how="left"
)

merged.to_csv("data/uga/uga_final_training_matrix_v2.csv", index=False)
print(f"\nMerged file saved to data/uga/uga_final_training_matrix.csv")
#print(merged[["Player", "position", "NIL_Valuation", "recruiting_stars", "recruiting_rating"]].head(20))
print(merged[["Player", "Position", "recruiting_stars", "recruiting_rating"]].head(20))
# how many still have nulls - will need to impute these in nil_model.py
nulls = merged["recruiting_rating"].isna().sum()
print(f"\nPlayers still missing recruiting data: {nulls} - will impute in nil_model.py")