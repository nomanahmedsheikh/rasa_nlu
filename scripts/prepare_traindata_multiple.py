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


global_df = pickle.load(open(PICKLE_DUMP_FILE, 'rb'))


def prepare(df, out_file):
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

    f = open(out_file, 'w')
    f.write(md)
    f.close()


N = global_df.shape[0]
iterations = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, N]
for count in iterations:
    df = global_df.sample(n=count)
    print('Preparing file for %d' % count)
    prepare(df, out_file='../data/experiment/Metlife_3files_%d.md' % count)
pass
