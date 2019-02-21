#!/usr/bin/env bash
for i in 10 20 40 80 160 320 640 1280 2560 5120 6319
do
python -m rasa_nlu.train -c nlu_config.yml --data data/experiment/Metlife_3files_$i.md -o models --fixed_model_name nlu_$i --project experiment --verbose
done
