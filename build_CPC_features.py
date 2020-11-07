import os
import sys
import json
import argparse
import progressbar
from pathlib import Path
from time import time
import numpy as np

from cpc.dataset import findAllSeqs
from cpc.feature_loader import buildFeature, FeatureModule, loadModel

def writeArgs(pathArgs, args):
    with open(pathArgs, 'w') as file:
        json.dump(vars(args), file, indent=2)

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Export CPC features from audio files.')
    parser.add_argument('pathCPCCheckpoint', type=str,
                        help='Path to the CPC checkpoint.')
    parser.add_argument('pathDB', type=str,
                        help='Path to the dataset that we want to quantize.')
    parser.add_argument('pathOutputDir', type=str,
                        help='Path to the output directory.')
    parser.add_argument('--file_extension', type=str, default=".flac",
                          help="Extension of the audio files in the dataset (default: .flac).")
    parser.add_argument('--get_encoded', type=bool, default=False,
                        help='If True, get the outputs of the encoder layer only (default: False).')
    parser.add_argument('--gru_level', type=int, default=-1,
                        help='Hidden level of the LSTM autoregressive model to be taken'
                        '(default: -1, last layer).')
    parser.add_argument('--max_size_seq', type=int, default=64000,
                        help='Maximal number of frames to consider in each chunk'
                        'when computing CPC features (defaut: 64000).')
    parser.add_argument('--seq_norm', type=bool, default=False,
                        help='If True, normalize the output along the time'
                        'dimension to get chunks of mean zero and var 1 (default: False).')
    parser.add_argument('--strict', type=bool, default=True,
                        help='If True, each batch of feature '
                        'will contain exactly max_size_seq frames (defaut: True).')
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
    print(f"Building CPC features from {args.pathDB}")
    print("=============================================================")

    # Find all sequences
    print("")
    print(f"Looking for all {args.file_extension} files in {args.pathDB}")
    seqNames, _ = findAllSeqs(args.pathDB,
                                 speaker_level=1,
                                 extension=args.file_extension,
                                 loadCache=True)
    if len(seqNames) == 0 or not os.path.splitext(seqNames[0][-1])[1].endswith(args.file_extension):
        print(f"Seems like the _seq_cache.txt does not contain the correct extension, reload the file list")
        seqNames, _ = findAllSeqs(args.pathDB,
                                    speaker_level=1,
                                    extension=args.file_extension,
                                    loadCache=False)
    print(f"Done! Found {len(seqNames)} files!")

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

    # Load CPC feature maker
    print("")
    print(f"Loading CPC model from {args.pathCPCCheckpoint}")
    if args.gru_level > 0:
        updateConfig = argparse.Namespace(nLevelsGRU=args.gru_level)
    else:
        updateConfig = None
    model, nHiddenGar, nHiddenEncoder = loadModel([args.pathCPCCheckpoint], updateConfig=updateConfig)
    model.gAR.keepHidden = True
    featureMaker = FeatureModule(model, get_encoded=args.get_encoded)
    featureMaker.eval()
    if not args.cpu:
        featureMaker.cuda()
    # Define CPC_feature_function
    def CPC_feature_function(x): 
        CPC_features = buildFeature(featureMaker, x,
                                    seqNorm=args.seq_norm,
                                    strict=args.strict,
                                    maxSizeSeq=args.max_size_seq)
        return CPC_features.squeeze(0).float().cpu().numpy()
    print("CPC FeatureMaker loaded!")

    # Building features
    print("")
    print(f"Building CPC features and saving outputs to {args.pathOutputDir}...")
    bar = progressbar.ProgressBar(maxval=len(seqNames))
    bar.start()
    start_time = time()
    for index, vals in enumerate(seqNames):
        bar.update(index)

        file_path = vals[1]
        file_path = os.path.join(args.pathDB, file_path)

        # Computing features
        CPC_features = CPC_feature_function(file_path)

        # Save the outputs
        file_name = os.path.splitext(os.path.basename(file_path))[0] + ".npy"
        file_out = os.path.join(args.pathOutputDir, file_name)
        np.save(file_out, CPC_features)
    bar.finish()
    print(f"...done {len(seqNames)} files in {time()-start_time} seconds.")

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)