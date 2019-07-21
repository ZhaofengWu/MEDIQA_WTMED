import json
import re

def process(input_path, output_path):
    with open(input_path, encoding='utf_8') as f:
        with open(output_path, 'w') as o:
            for i, line in enumerate(f):
                json_obj = json.loads(line)
                for k, v in json_obj.items():
                    v = re.sub(r'\t', '', v)
                    v = re.sub(r'\\t', '', v)
                    json_obj[k] = v
                o.write(f"{i}\t{json_obj['sentence1'].strip()}\t{json_obj['sentence2'].strip()}\t{json_obj['sentence1_binary_parse'].strip()}\t{json_obj['sentence2_binary_parse'].strip()}\t{json_obj['gold_label'].strip()}\n")

if __name__ == '__main__':
    process('data/train.json', 'data/train.tsv')
    process('data/dev.json', 'data/dev.tsv')
