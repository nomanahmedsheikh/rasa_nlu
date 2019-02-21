import pandas as pd
from tqdm import tqdm

"""
FOR TRAINING:
python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose
"""

from rasa_nlu.model import Interpreter

interpreter = Interpreter.load("./models/metlife_3files/nlu")

df = pd.read_excel('./data/Validation Set_12th Feb.xlsx')

ACT = 'Actionable'.lower()
NAC = 'Non-actionable'.lower()


def get_f1(precision, recall):
    if precision * recall is 0:
        return 0
    return 2 * precision * recall / (precision + recall)


def evaluate_model(model_path):
    freq = {(ACT, ACT): 0, (ACT, NAC): 0, (NAC, ACT): 0, (NAC, NAC): 0}
    print('Loading %s...' % model_path)
    interpreter = Interpreter.load(model_path)
    for i, row in tqdm(df.iterrows()):
        msg = row['Message'].strip()
        result = interpreter.parse(msg)
        true_action = row['Actionability'].lower()
        pred_action = str(result['intent']['name']).lower()
        freq[(true_action, pred_action)] += 1
    precision = {}
    recall = {}
    precision[ACT] = freq[(ACT, ACT)] / (freq[(ACT, ACT)] + freq[(NAC, ACT)])
    recall[ACT] = freq[(ACT, ACT)] / (freq[(ACT, ACT)] + freq[(ACT, NAC)])
    precision[NAC] = freq[(NAC, NAC)] / (freq[(NAC, NAC)] + freq[(ACT, NAC)])
    recall[NAC] = freq[(NAC, NAC)] / (freq[(NAC, NAC)] + freq[(NAC, ACT)])
    f1 = {ACT: get_f1(precision[ACT], recall[ACT]), NAC: get_f1(precision[NAC], recall[NAC])}
    return precision, recall, f1


iterations = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
values = []
for count in iterations:
    model_file = './models/experiment/nlu_%d' % count
    value = evaluate_model(model_file)
    values.append((count, value))

import pickle

pickle.dump(values, open('values.pkl', 'wb'))

import matplotlib.pyplot as plt
counts = []
precision = {ACT: [], NAC: []}
recall = {ACT: [], NAC: []}
f1 = {ACT: [], NAC: []}
for count, (p, r, f) in values:
    counts.append(count)
    precision[ACT].append(p[ACT])
    precision[NAC].append(p[NAC])
    recall[ACT].append(r[ACT])
    recall[NAC].append(r[NAC])
    f1[ACT].append(f[ACT])
    f1[NAC].append(f[NAC])

# plt.plot(counts, precision[ACT], 'r--', counts, f1[ACT], 'b--', counts, f1[NAC], 'g--')
plt.plot(counts, f1[ACT], 'b--', counts, f1[NAC], 'g--')
plt.show()
