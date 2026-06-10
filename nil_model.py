# NIL valuation model for RosterEdge - FSU 2025
# predicts NIL market value using position and social following
# training data is 25 verified valuations from On3

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
TARGET_TEAM = 'Florida State'
TEAM_SLUG   = 'fsu'

# load data
df_train  = pd.read_csv("data/training/final_training_matrix.csv")
df_roster = pd.read_csv(f"data/fsu/fsu_roster_processed_{YEAR_TARGET}.csv")
df_depth  = pd.read_csv(f"data/fsu/fsu_depth_chart_{YEAR_TARGET}.csv")
df_nil    = pd.read_csv(f"data/fsu/fsu_nil_on3_raw.csv")
df_impute = pd.read_csv("data/training/social_impute.csv")  # position + year -> estimated followers

print(f"{TARGET_TEAM} roster: {len(df_roster)} players")

# normalize positions - the training data v depth chart are inconsistent
pos_map = {
    'OT': 'OL', 'IOL': 'OL', 'CB': 'DB', 'S': 'DB',
    'LB': 'DL', 'LS': 'OL', 'P': 'OL', 'PK': 'OL',
    'DT': 'DL', 'DE': 'DL'
}
df_train['Position']  = df_train['Position'].replace(pos_map)
df_roster['position'] = df_roster['position'].replace(pos_map)

# features: total social following + position
social_cols = ['FACEBOOK_FOOTBALL', 'INSTAGRAM_FOOTBALL', 'TIKTOK_FOOTBALL',
               'TWITTER_FOOTBALL', 'YOUTUBE_FOOTBALL']
df_train['total_social'] = df_train[social_cols].sum(axis=1)

# convert categorical positions to one-hot encoding
pos_dummies = pd.get_dummies(df_train['Position'], prefix='pos').astype(float)
X_train = pd.concat([df_train[['total_social']], pos_dummies], axis=1)
y_train = df_train['NIL_Valuation'].values

# LOOCV - n=25 is too small for regular k-fold
loo = LeaveOneOut()
oof_preds = np.zeros(len(df_train))

for train_idx, test_idx in loo.split(X_train):
    m = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)
    m.fit(X_train.iloc[train_idx], y_train[train_idx])
    oof_preds[test_idx] = m.predict(X_train.iloc[test_idx])

loocv_mae = mean_absolute_error(y_train, oof_preds)
loocv_r2  = r2_score(y_train, oof_preds)
print(f"LOOCV MAE: ${loocv_mae:,.0f} | R2: {loocv_r2:.4f}")
# r2 is low - rank data would help a lot here, adding in final milestone

# train on full dataset
model = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)
model.fit(X_train, y_train)
training_cols = X_train.columns.tolist()

# depth chart role lookup
role_mult = {'starter': 1.45, 'backup': 1.15, 'depth': 1.0}
# ^^ weights based on their depth chart role
role_lookup = {}
for _, row in df_depth.iterrows():
    if pd.notna(row.get('Starter')):
        role_lookup[row['Starter'].strip()] = 'starter'
    if pd.notna(row.get('Backup')):
        role_lookup[row['Backup'].strip()] = 'backup'
    if pd.notna(row.get('Depth')):
        role_lookup[row['Depth'].strip()] = 'depth'

# build imputation lookup from csv - easier to update than hardcoding
impute_lookup = {}
for _, row in df_impute.iterrows():
    impute_lookup[(row['position'], int(row['year']))] = int(row['social_followers'])

# on3 floors - whisper is closer to real deal value, nil_value is the model estimate
known_nil = {}
for _, row in df_nil.iterrows():
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

# build fsu inference rows
rows = []
for _, row in df_roster.iterrows():
    name = row['Full_Name']
    pos  = row['position']
    year = int(row['year']) if not pd.isna(row['year']) else 1
    yb   = min(year, 4)

    social = known_social.get(name, impute_lookup.get((pos, yb), 3000))

    rows.append({
        'Full_Name':    name,
        'position':     pos,
        'year':         year,
        'total_social': social,
        'depth_role':   role_lookup.get(name, 'depth'),
    })

df_fsu = pd.DataFrame(rows)

fsu_pos_dummies = pd.get_dummies(df_fsu['position'], prefix='pos').astype(float)
X_fsu_raw = pd.concat([df_fsu[['total_social']], fsu_pos_dummies], axis=1)
X_fsu = X_fsu_raw.reindex(columns=training_cols, fill_value=0.0)

df_fsu['base_nil']      = np.round(model.predict(X_fsu), -2)
df_fsu['role_mult']     = df_fsu['depth_role'].map(role_mult)
df_fsu['predicted_nil'] = np.round(df_fsu['base_nil'] * df_fsu['role_mult'], -2)

# --- SHAP Feature Contributions ---
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# # Summary beeswarm plot — shows each feature's direction + magnitude
# shap.summary_plot(shap_values, X_train, show=False)
# plt.tight_layout()
# plt.savefig('data/fsu/shap_summary.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("SHAP summary plot saved.")

# # Bar plot — mean absolute SHAP (simple, good for writeup)
# shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)
# plt.tight_layout()
# plt.savefig('data/fsu/shap_bar.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("SHAP bar plot saved.")

# use on3/whisper as floor
for name, nil_val in known_nil.items():
    mask = df_fsu['Full_Name'] == name
    if mask.any():
        curr = df_fsu.loc[mask, 'predicted_nil'].values[0]
        df_fsu.loc[mask, 'predicted_nil'] = max(curr, nil_val)

df_out = df_fsu[['Full_Name', 'position', 'year', 'total_social',
                  'depth_role', 'role_mult', 'base_nil', 'predicted_nil']].copy()
df_out = df_out.sort_values('predicted_nil', ascending=False).reset_index(drop=True)
df_out['predicted_nil_fmt'] = df_out['predicted_nil'].map(lambda x: f"${x:,.0f}")

df_out.to_csv(f"data/fsu/fsu_nil_valuations_final.csv", index=False)

print(df_out[['Full_Name', 'position', 'depth_role', 'predicted_nil_fmt']].head(20).to_string(index=False))
print(f"\nsaved to data/fsu/fsu_nil_valuations_final.csv")