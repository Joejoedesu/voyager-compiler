import argparse
import itertools
import os
import re
import subprocess
import sys

import pandas as pd

models = [
    'models/mobilebert_tiny_squad',
    'csarron/mobilebert-uncased-squad-v1',
    "distilbert-base-uncased-distilled-squad",
    'csarron/bert-base-uncased-squad-v1',
    'bert-large-uncased-whole-word-masking-finetuned-squad',
]

dtypes = ['posit8_1', 'fp8_e4m3']


def run_evaluation(model, dtype, log_file, gpu):
    command = [
        'python', 'examples/question_answering/run_qa_no_trainer.py',
        '--model_name_or_path', model,
        '--dataset_name', 'rajpurkar/squad',
        '--per_device_eval_batch_size', '16',
        '--max_seq_length', '384',
        '--doc_stride', '128',
        '--pad_to_max_length',
        '--bf16',
        '--activation', dtype,
        '--weight', dtype,
        '--log_file', log_file,
    ]
    if gpu is not None:
        command += ['--gpu', gpu]
    print("Running:", ' '.join(command))
    subprocess.run(command, check=True)


def extract_f1_scores(log_file, out_file):
    with open(log_file, 'r') as file, open(out_file + '.out', 'w') as out:
        scores = (re.findall(r"'f1': (\d+\.\d+)", file.read()))
        for i in range(0, len(scores), len(dtypes)):
            out.write('\t'.join(scores[i:i + len(dtypes)]) + '\n')
        return scores


def write_csv(scores, out_file):
    expected = len(models) * len(dtypes)
    assert len(scores) == expected, f"Expected {expected}, got {len(scores)}"
    rows = [
        'MobileBERT-tiny',
        'MobileBERT',
        'DistillBERT-base',
        'BERT-base',
        'BERT-large'
    ]
    columns = ['Posit8', 'E4M3']
    scores_matrix = [
        scores[i:i + len(dtypes)] for i in range(0, len(scores), len(dtypes))
    ]
    df = pd.DataFrame(scores_matrix, index=rows, columns=columns)
    df.to_csv(out_file + '.csv')


def run_experiments(args):
    for model, dtype in itertools.product(models, dtypes):
        run_evaluation(model, dtype, args.log_file, args.gpu)
        scores = extract_f1_scores(args.log_file, args.out_file)
    print("All commands executed.")
    write_csv(scores, args.out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', default='logs/squad.log')
    parser.add_argument('--out_file', default='squad_f1')
    parser.add_argument('--gpu', default=None)
    args = parser.parse_args()

    if os.path.exists(args.log_file) and os.path.getsize(args.log_file) > 0:
        print("Log file exists and is not empty. Extracting scores...")
        scores = extract_f1_scores(args.log_file, args.out_file)
        write_csv(scores, args.out_file)
        sys.exit(0)

    run_experiments(args)
