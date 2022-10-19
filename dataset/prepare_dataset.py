# %%
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# %%
dfs = []

for sheet_name in ['ALL_PDR_processed', 'ALL_NPDR_processed']:

    df = pd.read_excel('data/List_patients_until_end_2020_processed.xlsx', sheet_name='ALL_PDR_processed', index_col=list(range(11)))
    df = df.dropna(axis=0)
    df = df.reset_index()
    df = df[['Patienten-Nr', 'eye', 'oct_path', 'image_type', 'acquisition_date', 'rows', 'columns', 'frames', 'image_hash', 'location', 'description', 'image_uuid']]
    df = df[(df.image_type == "Scanning Laser Ophthalmoscope") & (df.description.str.startswith("Section IR 30°"))]
    df = df.sort_values('acquisition_date', ascending=False)
    df['proliferative'] = sheet_name == 'ALL_PDR_processed'
    df['selected'] = False # none selected, we'll select next

    selection = set()

    for index, row in df.iterrows():
        exam_id = "_".join((str(row['Patienten-Nr']), row.eye, row.acquisition_date.strftime("%m-%d-%Y")))
        print(exam_id)
        if exam_id in selection:
            pass
        else:
            df.loc[index, 'selected'] = True
            selection.add(exam_id)

    dfs.append(df.copy())

pd.concat(dfs).to_csv('data/input_data_cleaned.csv')

# %%
pd.concat(dfs)

# %% [markdown]
# # Strategy
# 
# - Sort the dataframe by descending acquisition date
# - Loop through the items
# - Select only one exam per day per patient, usually the one taken as last is the best
#    - Keep track of everything in a set where (patient-id, eye, acquisition-day) are logged
#    - If an item in the set is already selected, skip and go to next
# 
# Remember: we initially selected exams acquired as far as 30 days from the patient entry date, thus we expected no errors in Fidus in reporting PDR-NPDR.

# %%

df = pd.read_csv("input_data_cleaned.csv", index_col=0)
df = df[df.selected == True]

df.proliferative.value_counts()
## well balanced: 122 proliferative, 122 non proliferative

len(df["Patienten-Nr"].unique())
## 71 patients, need to group by patient when splitting

df = df[['Patienten-Nr', 'image_uuid', 'proliferative']]

# %%

splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=55)
subsplitter = StratifiedGroupKFold(n_splits=2, shuffle=False)

train_idxs, valtest_idxs = list(splitter.split(
    df['image_uuid'], 
    df['proliferative'], 
    df['Patienten-Nr']
    ))[0]
df.iloc[train_idxs].to_csv("train.csv")

df_valtest = df.iloc[valtest_idxs]
val_idxs, test_idxs = list(subsplitter.split(
    df_valtest['image_uuid'], 
    df_valtest['proliferative'], 
    df_valtest['Patienten-Nr']
    ))[0]
df_valtest.iloc[val_idxs].to_csv("validation.csv")
df_valtest.iloc[test_idxs].to_csv("test.csv")
# %%
