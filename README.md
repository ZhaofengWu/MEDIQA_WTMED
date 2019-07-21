This is the official code repository for the paper

Zhaofeng Wu, Yan Song, Sicong Huang, Yuanhe Tian and Fei Xia<br/>
A Hybrid Approach to Biomedical Natural Language Inference

This package is based on the repostiory for [Multi-Task Deep Neural Networks (MT-DNN)](https://github.com/namisan/mt-dnn).

# Quickstart

0. Requirement: Python >= 3.6; `pip install -r requirements.txt`
1. Get MT-DNN pre-trained models: `sh download.sh`
2. Get the MedNLI data `mli_{train,dev,test}_v1.jsonl` and put them under `data/`
3. As the paper describes, we merge the original MedNLI train and dev sets, and treat the original test set as the new dev set: `cat mli_dev_v1.jsonl >> mli_train_v1.jsonl && mv mli_train_v1.jsonl train.json && mv mli_test_v1.jsonl dev.json`
4. Convert the MedNLI json files into tsv: `python json2tsv.py`
5. Optional: Process the MedNLI data using [this tool](https://github.com/jtourille/mimic-tools) that substitutes masked PHI tokens with pseudo-information. You might need to make some minor modifications of the script as MedNLI has slightly different PHI format from MIMIC-III
6. Preprocess tsv into a format that the MT-DNN scripts like: `python prepro.py`
7. Generate features: `python generate_domain_features.py && python generate_generic_features.py`
8. Put `glove.6B.300d.txt` under the root directory
9. `python train.py --data_dir data/mt_dnn/ --init_checkpoint mt_dnn_models/mt_dnn_base.pt --batch_size 16 --output_dir checkpoints/model --log_file checkpoints/model/log.log --answer_opt 0 --train_datasets mednli --test_datasets mednli --epochs 15 --stx_parse_dim 100 --glove_path glove.6B.300d.txt --unk_threshold 5 --feature_dim 20 --use_parse --use_generic_features --use_domain_features`
10. Postprocessing: see `postpro/postpro.py`

# Language Model Fine-tuning

1. Get some unlabeled corpus. We used MIMIC-III
2. Do some cleaning and convert the format to what step 3 below expects. We used [this tool](https://github.com/jtourille/mimic-tools) to substitute masked PHI tokens with pseudo-information
3. Use the repo [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) (now renamed to pytorch-transformers). Use the `pregenerate_training_data.py` and `finetune_on_pregenerated` scripts under their `examples/lm_finetuning` directory
4. We need to resolve some format inconsistencies between the two libraries: `python wrapping_util.py [PATH_TO_pytorch-pretrained-BERT_TRAINED_MODEL] mt_dnn_models/mt_dnn_base.pt && mv new_pytorch_model.bin mt_dnn_models/finetuned_model.pt`. This assumes that you're using the base model; change to `mt_dnn_large.pt` otherwise
5. Train the model as in Quickstart, but change `--init_checkpoint` to point to this new model
