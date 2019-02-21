import pandas as pd
from tqdm import tqdm

"""
FOR TRAINING:
python -m rasa_nlu.train -c nlu_config.yml --data nlu.md -o models --fixed_model_name nlu --project current --verbose
"""


from rasa_nlu.model import Interpreter
interpreter = Interpreter.load("./models/metlife_3files/nlu")

df = pd.read_excel('./data/Validation Set_12th Feb.xlsx')
rows = []
for i, row in tqdm(df.iterrows()):
    msg = row['Message'].strip()
    result = interpreter.parse(msg)
    action = row['Actionability']
    intent = str(result['intent']['name'])
    conf = result['intent']['confidence']
    line = {'ID': row['UniversalMessageId'], 'Message': msg, 'Intent': intent, 'Confidence': conf, 'Actionability': action}
    rows.append(line)

out_df = pd.DataFrame(rows)
out_df.to_excel('data/validation_metlife_12Feb.xlsx')