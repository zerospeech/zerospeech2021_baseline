import os
import sys
import json
import argparse
import progressbar
from pathlib import Path
from time import time
import numpy as np

import torch
from fairseq.models.roberta import RobertaModel

def writeArgs(pathArgs, args):
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export BERT features from quantized units of audio files.')
    parser.add_argument('pathQuantizedUnits', type=str,
                        help='Path to the quantized units. Each line of the input file must be'
                        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument('pathOutputDir', type=str,
                        help='Path to the output directory.')
    parser.add_argument('pathBERTCheckpoint', type=str,
                        help='Path to the trained fairseq BERT(RoBERTa) model.')
    parser.add_argument('--dict', type=str,
                       help='Path to the dictionary file (dict.txt) used to train the BERT model'
                       '(if not speficied, look for dict.txt in the model directory)')
    parser.add_argument('--hidden_level', type=int, default=-1,
                          help="Hidden layer of BERT to extract features from (default: -1, last layer).")
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    parser.add_argument('--cpu', action='store_true',
                        help="Run on a cpu machine.")
    return parser.parse_args(argv)

def main(argv):
    # Args parser
    args = parseArgs(argv)

    print("=============================================================")
    print(f"Building BERT features from {args.pathQuantizedUnits}")
    print("=============================================================")

    # Load input file
    print("")
    print(f"Reading input file from {args.pathQuantizedUnits}")
    seqNames = []
    seqInputs = []
    with open(args.pathQuantizedUnits, 'r') as f:
        for line in f:
            file_name, file_seq = line.strip().split("\t")
            # Convert sequence to the desired input form
            file_seq = file_seq.replace(",", " ")
            # Add to lists
            seqNames.append(file_name)
            seqInputs.append(file_seq)
    print(f"Found {len(seqNames)} sequences!")

    # Verify the output directory
    if os.path.exists(args.pathOutputDir):
        existing_files = set([os.path.splitext(os.path.basename(x))[0]
                            for x in os.listdir(args.pathOutputDir) if x[-4:]==".npy"])
        seqNames = [s for s in seqNames if os.path.splitext(os.path.basename(s[1]))[0] not in existing_files]
        print(f"Found existing output directory at {args.pathOutputDir}, continue to build features of {len(seqNames)} audio files left!")
    else:
        print("")
        print(f"Creating the output directory at {args.pathOutputDir}")
        Path(args.pathOutputDir).mkdir(parents=True, exist_ok=True)
    writeArgs(os.path.join(args.pathOutputDir, "_info_args.json"), args)

    # Debug mode
    if args.debug:
        nsamples=20
        print("")
        print(f"Debug mode activated, only load {nsamples} samples!")
        # shuffle(seqNames)
        seqNames = seqNames[:nsamples]
        seqInputs = seqInputs[:nsamples]

    # Load BERT model
    if args.dict is None:
        PATH_DATA = os.path.dirname(args.pathBERTCheckpoint)
    else:
        PATH_DATA = os.path.dirname(args.dict)
    assert os.path.exists(os.path.join(PATH_DATA, "dict.txt")), \
        f"Dictionary file (dict.txt) not found in {PATH_DATA}"
    print("")
    print(f"Loading RoBERTa model from {args.pathBERTCheckpoint}...")
    print(f"Path data {PATH_DATA}")
    roberta = RobertaModel.from_pretrained(os.path.dirname(args.pathBERTCheckpoint), os.path.basename(args.pathBERTCheckpoint), PATH_DATA)
    roberta.eval()  # disable dropout (or leave in train mode to finetune)
    if not args.cpu:
        roberta.cuda()
    print("Model loaded !")

    # Define BERT_feature_function
    def BERT_feature_function(input_sequence):
        sentence_tokens = roberta.task.source_dictionary.encode_line(
                            "<s> " + input_sequence,
                            append_eos=True,
                            add_if_not_exist=False).type(torch.LongTensor)
        if not args.cpu:
            sentence_tokens = sentence_tokens.cuda()

        with torch.no_grad():
            outputs = roberta.extract_features(sentence_tokens, return_all_hiddens=True)

        return outputs[args.hidden_level].squeeze(0).float().cpu().numpy()

    # Building features
    print("")
    print(f"Building BERT features and saving outputs to {args.pathOutputDir}...")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, (name_seq, input_seq) in enumerate(zip(seqNames, seqInputs)):
        bar.update(index)

        # Computing features
        BERT_features = BERT_feature_function(input_seq)

        # Save the outputs
        file_name = os.path.splitext(name_seq)[0] + ".txt"
        file_out = os.path.join(args.pathOutputDir, file_name)
        np.savetxt(file_out, BERT_features)
    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
