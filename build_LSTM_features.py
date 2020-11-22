import os
import sys
import json
import argparse
import progressbar
from pathlib import Path
from time import time
import numpy as np
from copy import deepcopy

import torch

from utils.utils_functions import writeArgs, loadLSTMLMCheckpoint

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export LSTM features from quantized units of audio files.')
    parser.add_argument('pathQuantizedUnits', type=str,
                        help='Path to the quantized units. Each line of the input file must be'
                        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument('pathOutputDir', type=str,
                        help='Path to the output directory.')
    parser.add_argument('pathLSTMCheckpoint', type=str,
                        help='Path to the trained fairseq lstm_lm model.')
    parser.add_argument('--dict', type=str,
                       help='Path to the dictionary file (dict.txt) used to train the LSTM LM model'
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

    # Load LSTM model
    if args.dict is None:
        pathData = os.path.dirname(args.pathLSTMCheckpoint)
    else:
        pathData = os.path.dirname(args.dict)
    assert os.path.exists(os.path.join(pathData, "dict.txt")), \
        f"Dictionary file (dict.txt) not found in {pathData}"
    print("")
    print(f"Loading LSTM model from {args.pathLSTMCheckpoint}...")
    print(f"Path data {pathData}")
    model, task = loadLSTMLMCheckpoint(
                    args.pathLSTMCheckpoint, 
                    pathData)
    model.eval()  # disable dropout (or leave in train mode to finetune)
    if not args.cpu:
        model.cuda()
    print("Model loaded !")

    # Define LSTM_feature_function
    def LSTM_feature_function(input_sequence, n_hidden=-1):
        # Get the number of layers
        num_layers = len(model.decoder.layers)
        assert abs(n_hidden) <= num_layers, \
            "absolute value of n_hidden must be less than or equal to the number of hidden layers = {}".format(num_layers)

        if n_hidden < 0:
            n_hidden = num_layers + 1 + n_hidden

        # Get input tensor
        input_tensor = task.source_dictionary.encode_line(
                            "<s> " + input_sequence,
                            append_eos=True,
                            add_if_not_exist=False).type(torch.LongTensor).unsqueeze(0)
        if not args.cpu:
            input_tensor = input_tensor.cuda()
            
        # Get the output
        if n_hidden == 0: # Take the embedding layer
            with torch.no_grad():
                output_tensor = model.decoder.embed_tokens(input_tensor)

        else:
            decoder_clone = deepcopy(model.decoder)
            
            # We don't take the final fc features
            decoder_clone.fc_out = torch.nn.Identity()
            decoder_clone.additional_fc = torch.nn.Identity()
            
            # Restrict the number of hiddden layers to n_hidden
            decoder_clone.layers = decoder_clone.layers[:n_hidden]

            with torch.no_grad():
                output_tensor = decoder_clone(input_tensor)[0]

        return output_tensor[0].data.cpu().numpy()

    # Building features
    print("")
    print(f"Building LSTM features and saving outputs to {args.pathOutputDir}...")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, (name_seq, input_seq) in enumerate(zip(seqNames, seqInputs)):
        bar.update(index)

        # Computing features
        LSTM_features = LSTM_feature_function(input_seq, n_hidden=args.hidden_level)

        # Save the outputs
        file_name = os.path.splitext(name_seq)[0] + ".txt"
        file_out = os.path.join(args.pathOutputDir, file_name)
        np.savetxt(file_out, LSTM_features)
    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
