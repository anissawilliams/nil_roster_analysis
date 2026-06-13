# NIL valuation model for RosterEdge - FSU 2025
# v2 - added recruiting_rating as feature, expanded training set to 106 players
# still a random forest
# previous R2 was 0.24 with n=25, hoping to do better with more data + recruiting rank

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

YEAR_TARGET = 2025
TARGET_TEAM = 'Georgia'
TEAM_SLUG   = 'uga'

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------

# nil_data is the full 106-player training set from On3
df_nil_data = pd.read_csv("data/training/nil_data.csv")

# recruiting ranks pulled from CFBD + manual overrides
df_recruiting = pd.read_csv("data/training/nil_recruiting_ranks_raw.csv")

# manual overrides - players we looked up manually bc CFBD didn't find them
# format: Player, recruiting_stars, recruiting_rating
df_overrides = pd.read_csv("data/training/manual_recruiting_overrides.csv")
#df_overrides.columns = df_overrides.columns.str.strip().str.title() #where upper/lower case mismatches
# fsu roster and supporting files
df_roster = pd.read_csv(f"data/uga/uga_roster_{YEAR_TARGET}.csv")
df_depth  = pd.read_csv(f"data/uga/uga_depth_chart_{YEAR_TARGET}.csv")
df_fsu_nil = pd.read_csv(f"data/uga/uga_nil_on3_raw.csv")
df_impute = pd.read_csv("data/training/social_impute.csv")

# fsu recruiting ranks - for inference
df_fsu_recruiting = pd.read_csv("data/fsu/fsu_recruiting_ranks_raw.csv")

print(f"nil_data players: {len(df_nil_data)}")
print(f"recruiting ranks matched: {df_recruiting['match_found'].sum()}")
print(f"manual overrides: {len(df_overrides)}")

# -----------------------------------------------
# BUILD TRAINING MATRIX
# -----------------------------------------------

# apply manual overrides to recruiting ranks
# just loop through and update - yes this is slow, its fine
for _, override in df_overrides.iterrows():
    name = override['Player']
    mask = df_recruiting['Player'] == name
    if mask.any():
        # update existing row
        df_recruiting.loc[mask, 'recruiting_rating'] = override['recruiting_rating']
        df_recruiting.loc[mask, 'recruiting_stars'] = override['recruiting_stars']
        df_recruiting.loc[mask, 'match_found'] = True
        df_recruiting.loc[mask, 'source'] = 'manual_override'
        print(f"  override applied: {name}")
    else:
        # add new row - player wasn't in recruiting file at all
        new_row = {
            'Player': name,
            'School': 'unknown',
            'Position': 'unknown',
            'recruiting_rating': override['recruiting_rating'],
            'recruiting_stars': override['recruiting_stars'],
            'recruiting_rank': None,
            'source': 'manual_override',
            'match_found': True
        }
        df_recruiting = pd.concat([df_recruiting, pd.DataFrame([new_row])], ignore_index=True)
        print(f"  override added: {name}")

# merge recruiting onto nil_data
df_train = df_nil_data.merge(
    df_recruiting[['Player', 'recruiting_rating', 'recruiting_stars', 'source']],
    on='Player',
    how='left'
)

print(f"\nTraining set after merge: {len(df_train)} players")
print(f"Has recruiting_rating: {df_train['recruiting_rating'].notna().sum()}")
print(f"Missing recruiting_rating: {df_train['recruiting_rating'].isna().sum()}")

# normalize positions - training data positions are inconsistent
pos_map = {
    'OT': 'OL', 'IOL': 'OL', 'CB': 'DB', 'S': 'DB',
    'LB': 'DL', 'LS': 'OL', 'P': 'OL', 'PK': 'OL',
    'DT': 'DL', 'DE': 'DL', 'EDGE': 'DL'
}
df_train['Position'] = df_train['Position'].replace(pos_map)
df_roster['position'] = df_roster['position'].replace(pos_map)

# impute missing recruiting_rating by position median
# walk-ons and unknown players get their position's median rating
# this is a reasonable assumption - if we don't know, assume average for that position
print("\nImputing missing recruiting ratings by position median...")
position_medians = df_train.groupby('Position')['recruiting_rating'].median()
print(position_medians)

# fill nulls with position median first, then overall median as fallback
df_train['recruiting_rating'] = df_train.groupby('Position')['recruiting_rating'].transform(
    lambda x: x.fillna(x.median())
)
# overall median fallback for any positions with all nulls
overall_median = df_train['recruiting_rating'].median()
# training set medians are skewed high - unknown players are more likely
# to be average recruits, not elite ones. use a conservative default instead.
df_train['recruiting_rating'] = df_train['recruiting_rating'].fillna(0.82)
print(f"Overall median rating used as fallback: {overall_median:.4f}")

# social following - use team-level as proxy (known limitation, noted in writeup)
# individual player social would be better but we don't have it for 106 players
# social_cols = ['FACEBOOK_FOOTBALL', 'INSTAGRAM_FOOTBALL', 'TIKTOK_FOOTBALL',
#                'TWITTER_FOOTBALL', 'YOUTUBE_FOOTBALL']
#df_train['total_social'] = df_train[social_cols].sum(axis=1)

# -----------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------

# features: total_social + position dummies + recruiting_rating
# this is the big change from v1 - adding recruiting_rating
pos_dummies = pd.get_dummies(df_train['Position'], prefix='pos').astype(float)
X_train = pd.concat([
    df_train[['recruiting_rating']],
    pos_dummies
], axis=1)

y_train = df_train['NIL_Valuation'].values

print(f"\nFeatures: {X_train.columns.tolist()}")
print(f"Training samples: {len(X_train)}")

# -----------------------------------------------
# LOOCV - keeping this even though n=106 now
# n=106 is still pretty small for a regression problem
# and LOOCV gives us the most honest estimate
# -----------------------------------------------
print("\nRunning LOOCV...")
loo = LeaveOneOut()
oof_preds = np.zeros(len(df_train))

for train_idx, test_idx in loo.split(X_train):
    m = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    m.fit(X_train.iloc[train_idx], y_train[train_idx])
    oof_preds[test_idx] = m.predict(X_train.iloc[test_idx])

loocv_mae = mean_absolute_error(y_train, oof_preds)
loocv_r2  = r2_score(y_train, oof_preds)
print(f"LOOCV MAE: ${loocv_mae:,.0f} | R2: {loocv_r2:.4f}")
# v1 baseline was MAE: $537,323 | R2: 0.2428 with n=25

# -----------------------------------------------
# TRAIN FINAL MODEL ON FULL DATASET
# -----------------------------------------------
# bumped n_estimators to 100 and max_depth to 4
# with more training data we can afford a slightly deeper tree
# still keeping it conservative to avoid overfitting
model = RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)
training_cols = X_train.columns.tolist()

# -----------------------------------------------
# SHAP
# -----------------------------------------------
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
plt.tight_layout()
plt.savefig('data/fsu/shap_bar_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP bar plot saved.")

shap.summary_plot(shap_values, X_train, show=False)
plt.tight_layout()
plt.savefig('data/fsu/shap_summary_v2.png', dpi=150, bbox_inches='tight')
plt.close()
print("SHAP summary plot saved.")

# -----------------------------------------------
# FSU INFERENCE
# -----------------------------------------------

# depth chart role lookup
role_mult   = {'starter': 1.45, 'backup': 1.15, 'depth': 1.0}
role_lookup = {}
for _, row in df_depth.iterrows():
    if pd.notna(row.get('Starter')):
        role_lookup[row['Starter'].strip()] = 'starter'
    if pd.notna(row.get('Backup')):
        role_lookup[row['Backup'].strip()] = 'backup'
    if pd.notna(row.get('Depth')):
        role_lookup[row['Depth'].strip()] = 'depth'

# social imputation lookup
impute_lookup = {}
for _, row in df_impute.iterrows():
    impute_lookup[(row['position'], int(row['year']))] = int(row['social_followers'])

# on3 floors
known_nil = {}
for _, row in df_fsu_nil.iterrows():
    val = row['whisper_value'] if pd.notna(row.get('whisper_value')) else row.get('nil_value')
    if pd.notna(val):
        known_nil[row['Full_Name']] = val

# manually looked up instagram followers for top FSU players
known_social = {
    'Squirrel White':    30700,
    'Tommy Castellanos': 107000,
    'Roydell Williams':  20500,
    'Jaylen King':       2285,
    'Jaylin Lucas':      10200,
    'Earl Little Jr.':   42800,
    'Markeston Douglas': 4675,
    'Duce Robinson':     95000,
    'Mandrell Desir':    8200,
}

# apply overrides to fsu recruiting ranks too
for _, override in df_overrides.iterrows():
    name = override['Player']
    mask = df_fsu_recruiting['Player'] == name
    if mask.any():
        df_fsu_recruiting.loc[mask, 'recruiting_rating'] = override['recruiting_rating']
        df_fsu_recruiting.loc[mask, 'recruiting_stars'] = override['recruiting_stars']
        df_fsu_recruiting.loc[mask, 'match_found'] = True

# fsu recruiting lookup dict
fsu_recruit_lookup = {}
for _, row in df_fsu_recruiting.iterrows():
    if row['match_found']:
        fsu_recruit_lookup[row['Player']] = row['recruiting_rating']

# build fsu inference rows
rows = []
for _, row in df_roster.iterrows():
    name = row['Full_Name']
    pos  = row['position']
    year = int(row['year']) if not pd.isna(row['year']) else 1
    yb   = min(year, 4)

    social = known_social.get(name, impute_lookup.get((pos, yb), 3000))

    # # get recruiting rating for this fsu player
    # # fall back to position median from training data if not found
    # recruit_rating = fsu_recruit_lookup.get(name, None)
    # if recruit_rating is None:
    #     recruit_rating = position_medians.get(pos, overall_median)
    recruit_rating = fsu_recruit_lookup.get(name, None)
    if recruit_rating is None:
        recruit_rating = 0.78 if year == 1 else 0.82
    rows.append({
        'Full_Name':        name,
        'position':         pos,
        'year':             year,
        'total_social':     social,
        'recruiting_rating': recruit_rating,
        'depth_role':       role_lookup.get(name, 'depth'),
    })

df_fsu = pd.DataFrame(rows)

fsu_pos_dummies = pd.get_dummies(df_fsu['position'], prefix='pos').astype(float)
X_fsu_raw = pd.concat([df_fsu[['total_social', 'recruiting_rating']], fsu_pos_dummies], axis=1)
X_fsu = X_fsu_raw.reindex(columns=training_cols, fill_value=0.0)

df_fsu['base_nil']      = np.round(model.predict(X_fsu), -2)
df_fsu['role_mult']     = df_fsu['depth_role'].map(role_mult)
df_fsu['predicted_nil'] = np.round(df_fsu['base_nil'] * df_fsu['role_mult'], -2)

# apply on3 floors
for name, nil_val in known_nil.items():
    mask = df_fsu['Full_Name'] == name
    if mask.any():
        curr = df_fsu.loc[mask, 'predicted_nil'].values[0]
        # floor: can't go below known value
        floored = max(curr, nil_val)
        # ceiling: shouldn't go more than 2x known value
        capped = min(floored, nil_val * 2)
        df_fsu.loc[mask, 'predicted_nil'] = capped
# global position ceilings - model inflates values, these are realistic market caps
# based on On3 market data and FSU's actual budget constraints
position_ceilings = {
    'QB': 2000000, 'WR': 1200000, 'DL': 800000,
    'OL': 700000, 'DB': 700000, 'RB': 600000,
    'TE': 500000, 'ATH': 400000
}

# only apply ceiling to players WITHOUT a known On3 value
df_fsu['predicted_nil'] = df_fsu.apply(
    lambda r: min(r['predicted_nil'],
    position_ceilings.get(r['position'], 400000))
    if r['Full_Name'] not in known_nil else r['predicted_nil'],
    axis=1
)
df_out = df_fsu[['Full_Name', 'position', 'year', 'total_social', 'recruiting_rating',
                  'depth_role', 'role_mult', 'base_nil', 'predicted_nil']].copy()
df_out = df_out.sort_values('predicted_nil', ascending=False).reset_index(drop=True)
df_out['predicted_nil_fmt'] = df_out['predicted_nil'].map(lambda x: f"${x:,.0f}")

df_out.to_csv(f"data/fsu/fsu_nil_valuations_final.csv", index=False)

print(df_out[['Full_Name', 'position', 'depth_role', 'recruiting_rating', 'predicted_nil_fmt']].head(20).to_string(index=False))
print(f"\nsaved to data/fsu/fsu_nil_valuations_final.csv")