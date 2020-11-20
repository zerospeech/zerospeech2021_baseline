import os
import sys
import json
import argparse
import progressbar
from pathlib import Path
from time import time
import numpy as np

from utils.utils_functions import writeArgs

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export 1-hot features from quantized units of audio files.')
    parser.add_argument('pathQuantizedUnits', type=str,
                        help='Path to the quantized units. Each line of the input file must be'
                        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument('pathOutputDir', type=str,
                        help='Path to the output directory.')
    parser.add_argument('--n_units', type=int, default=50,
                        help='Number of discrete units (default: 50). If a dictionary is given,'
                        'this is automatically set as vocab size.')
    parser.add_argument('--dict', type=str,
                       help='Path to the dictionary file containing vocab of the pseudo units on the dataset'
                       '(this is required if the quantized units are not digits, i.e. multi-group case).')
    parser.add_argument('--debug', action='store_true',
                        help="Load only a very small amount of files for "
                        "debugging purposes.")
    return parser.parse_args(argv)

def main(argv):
    # Args parser
    args = parseArgs(argv)

    print("=============================================================")
    print(f"Building 1-hot features from {args.pathQuantizedUnits}")
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

    # Load 1hot dictionary in case we use it
    if seqInputs and not seqInputs[0].split()[0].isdigit(): #multi-group ie. 65-241
        assert args.dict is not None, \
            "A dictionary must be given when the quantized outputs is not digits (multi-group case)!"
    if args.dict:
        print("")
        print(f"Loading onehot dictionary from {args.dict}...")
        with open(args.dict, "r") as f:
            lines = f.read().split("\n")
        pair2idx={word.split()[0]: i for i, word in enumerate(lines) if word and not word.startwith("madeupword")}
        args.n_units = len(pair2idx)

    # Define onehot_feature_function
    def onehot_feature_function(input_sequence):
        if args.dict:
            indexes_sequence = np.array([pair2idx[item] for item in input_sequence.split()])
        else:
            indexes_sequence = np.array([int(item) for item in input_sequence.split()])

        onehotFeatures = np.eye(args.n_units)[indexes_sequence]

        return onehotFeatures

    # Building features
    print("")
    print(f"Building 1-hot features and saving outputs to {args.pathOutputDir}...")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, (name_seq, input_seq) in enumerate(zip(seqNames, seqInputs)):
        bar.update(index)

        # Computing features
        onehot_features = onehot_feature_function(input_seq)

        # Save the outputs
        file_name = os.path.splitext(name_seq)[0] + ".txt"
        file_out = os.path.join(args.pathOutputDir, file_name)
        np.savetxt(file_out, onehot_features)
    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
