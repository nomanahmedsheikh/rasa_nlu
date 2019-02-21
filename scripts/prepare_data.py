import pandas as pd
import pickle

INP_FILE = '../data/Metlife_7th Feb.xlsx'
OUT_FILE = '../data/Metlife_3files.md'
PICKLE_DUMP_FILE = '../data/Metlife_3files.pkl'


def filter_df(df, min_length, max_length=None, num_samples=None):
    df = df[df['Message'] == df['Message']]
    df = df[df['Message'].str.len() > min_length]
    if max_length is not None:
        df = df[df['Message'].str.len() < max_length]
    if num_samples is not None:
        df = df.sample(n=num_samples)
    return df


df = pd.read_excel(INP_FILE, sheet_name='Consolidated')
df = filter_df(df, 2, None, None)
df = df[['Message', 'Validation']]


df_action = pd.read_excel('../data/Sales Actionable Export.xlsx')
df_action = filter_df(df_action, 2, None, 2000)
df_action['Validation'] = 'Actionable'
df_action = df_action[['Message', 'Validation']]

df_non_action = pd.read_excel('../data/Sales_NotActionable_Export.xlsx')
df_non_action = filter_df(df_non_action, 2, None, 2000)
df_non_action['Validation'] = 'Non-actionable'
df_non_action = df_non_action[['Message', 'Validation']]

df = pd.concat([df, df_action, df_non_action])
df = df.sample(frac=1)
pickle.dump(df, open(PICKLE_DUMP_FILE, 'wb'))

intent_vs_msg = {}

for i, row in df.iterrows():
    msg = row['Message'].strip()
    msg = msg
    intent = row['Validation']
    if intent not in intent_vs_msg:
        intent_vs_msg[intent] = []
    if msg in intent_vs_msg[intent]:
        continue
    intent_vs_msg[intent].append(msg)


md = ''
for intent in intent_vs_msg.keys():
    md += '## intent:%s\n' % intent
    for msg in intent_vs_msg[intent]:
        md += '- %s\n' % msg
    md += '\n'

f = open(OUT_FILE, 'w')
f.write(md)
f.close()

pass