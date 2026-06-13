
import requests
import pandas as pd
import os
import time

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("CFBD_API_KEY")

# load the training matrix
df = pd.read_csv("data/training/nil_data.csv")
df_fsu = pd.read_csv("data/fsu/fsu_roster_processed_2025.csv")

print(f"Loaded {len(df)} players")
print(df.head())

# going to store results here
results = []

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "accept": "application/json"
}

# loop through every player - yes this is slow but I can see what's happening
for i, row in df.iterrows():
    player_name = row["Player"]
    school = row["School"]
    position = row["Position"]

    print(f"[{i + 1}/{len(df)}] Looking up: {player_name} ({school})")

    # search recruiting by name
    # NOTE: CFBD recruiting endpoint searches by name, not exact match
    # so we get back multiple results sometimes and have to filter
    url = "https://api.collegefootballdata.com/recruiting/players"

    params = {
        "search": player_name,
    }

    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"   ERROR: status {response.status_code}")
            results.append({
                "Player": player_name,
                "School": school,
                "Position": position,
                "recruiting_rank": None,
                "recruiting_stars": None,
                "recruiting_rating": None,
                "recruit_school": None,
                "recruit_year": None,
                "match_found": False
            })
            time.sleep(0.5)
            continue

        data = response.json()

        if not data:
            print(f"   no results found")
            results.append({
                "Player": player_name,
                "School": school,
                "Position": position,
                "recruiting_rank": None,
                "recruiting_stars": None,
                "recruiting_rating": None,
                "recruit_school": None,
                "recruit_year": None,
                "match_found": False
            })
            time.sleep(0.5)
            continue

        # try to find the best match - look for committed school match first
        # sometimes a player's committed school != current school (transfers)
        # so we just grab the highest rated result if no school match
        best_match = None

        for recruit in data:
            recruit_name = recruit.get("name", "")
            committed_to = recruit.get("committedTo", "") or ""

            # exact name match preferred
            if recruit_name.lower() == player_name.lower():
                if school.lower() in committed_to.lower():
                    # perfect - name and school match
                    best_match = recruit
                    break
                elif best_match is None:
                    # name matches but school doesnt (transfer) - keep as fallback
                    best_match = recruit

        # if we still dont have anything just take the first result
        if best_match is None and data:
            best_match = data[0]
            print(f"   WARNING: no exact name match, using first result: {data[0].get('name')}")

        if best_match:
            rank = best_match.get("ranking")
            stars = best_match.get("stars")
            rating = best_match.get("rating")
            recruit_school = best_match.get("committedTo")
            recruit_year = best_match.get("year")

            print(f"   FOUND: rank={rank}, stars={stars}, rating={rating}, committed={recruit_school} ({recruit_year})")

            results.append({
                "Player": player_name,
                "School": school,
                "Position": position,
                "recruiting_rank": rank,
                "recruiting_stars": stars,
                "recruiting_rating": rating,
                "recruit_school": recruit_school,
                "recruit_year": recruit_year,
                "match_found": True
            })

    except Exception as e:
        print(f"   EXCEPTION: {e}")
        results.append({
            "Player": player_name,
            "School": school,
            "Position": position,
            "recruiting_rank": None,
            "recruiting_stars": None,
            "recruiting_rating": None,
            "recruit_school": None,
            "recruit_year": None,
            "match_found": False
        })

    # be nice to the API - don't hammer it
    time.sleep(0.5)

# save results
recruiting_df = pd.DataFrame(results)
recruiting_df.to_csv("data/recruiting_ranks_raw.csv", index=False)

print("\n--- DONE ---")
print(f"Total players: {len(recruiting_df)}")
print(f"Matches found: {recruiting_df['match_found'].sum()}")
print(f"No match: {(~recruiting_df['match_found']).sum()}")
print("\nSaving to data/recruiting_ranks_raw.csv")

# quick sanity check - how many have a rank?
has_rank = recruiting_df["recruiting_rank"].notna().sum()
print(f"Players with recruiting rank: {has_rank}/{len(recruiting_df)}")

# now merge back to original
merged = df.merge(recruiting_df[["Player", "recruiting_rank", "recruiting_stars", "recruiting_rating", "recruit_year"]],
                  on="Player",
                  how="left")

merged.to_csv("data/final_training_matrix_v2.csv", index=False)
print(f"\nMerged file saved to data/final_training_matrix_v2.csv")
print(merged[["Player", "Position", "NIL_Valuation", "recruiting_rank", "recruiting_stars"]].head(20))