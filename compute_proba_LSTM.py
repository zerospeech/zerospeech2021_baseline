from pathlib import Path
from os.path import exists, join, basename, dirname, abspath
import sys
import argparse

from utils.utils_functions import loadLSTMLMCheckpoint
from utils.lm_scoring import compute_proba_LSTM

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Compute pseudo log-probabilities of quantized units with a trained BERT model.')
    parser.add_argument('pathQuantizedUnits', type=str,
                        help='Path to the quantized units. Each line of the input file must be'
                        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument('pathOutputFile', type=str,
                        help='Path to the output file containing scores.')
    parser.add_argument('pathLSTMCheckpoint', type=str,
                        help='Path to the trained fairseq LSTM model.')
    parser.add_argument('--dict', type=str,	
                       help='Path to the dictionary file (dict.txt) used to train the LSTM model'
                       '(if not speficied, look for dict.txt in the model directory)')
    parser.add_argument('--batchSize', type=int, default=128,
                        help='The number of sentences to be in each batch (defaut: 128)')
    parser.add_argument('--cpu', action='store_true',
                        help="Run on a cpu machine.")
    parser.add_argument('--resume', action='store_true',
                        help="Continue to compute score if the output file already exists.")
    return parser.parse_args(argv)

def main(argv):
    # Args parser
    args = parseArgs(argv)

    # Convert to absolute paths to get rid of exceptions
    args.pathQuantizedUnits = abspath(args.pathQuantizedUnits)
    args.pathOutputFile = abspath(args.pathOutputFile)
    args.pathLSTMCheckpoint = abspath(args.pathLSTMCheckpoint)
    if args.dict is not None:
        args.dict = abspath(args.dict)

    # Load input file
    print("")
    print(f"Reading input file from {args.pathQuantizedUnits}")
    input_file_names = []
    intput_file_seqs = []
    with open(args.pathQuantizedUnits, 'r') as f:
        for line in f:
            file_name, file_seq = line.strip().split("\t")
            # Convert sequence to the desired input form
            file_seq = file_seq.replace(",", " ")
            # Add to lists
            input_file_names.append(file_name)
            intput_file_seqs.append(file_seq)
    print(f"Found {len(input_file_names)} sequences!")

    # Check if directory exists
    pathOutputDir = dirname(args.pathOutputFile)
    if pathOutputDir and not exists(pathOutputDir):
        print("")
        print(f"Creating the output directory at {pathOutputDir}")
        Path(pathOutputDir).mkdir(parents=True, exist_ok=True)
    # writeArgs(join(pathOutputDir, "_info_args.json"), args)
    
    # Continue
    if args.resume:
        if exists(args.pathOutputFile):
            existing_file_names = []
            with open(args.pathOutputFile, 'r') as f:
                lines = [line for line in f]
            for line in lines:
                file_name, score = line.strip().split()
                existing_file_names.append(file_name)
            assert input_file_names[:len(existing_file_names)] == existing_file_names, \
                "The file names in the existing output file do not match the input file!!"
            input_file_names = input_file_names[len(existing_file_names):]
            intput_file_seqs = intput_file_seqs[len(existing_file_names):]
            print(f"Found existing output file, continue to compute scores of {len(intput_file_seqs)} sequences left!")
    else:
        assert not exists(args.pathOutputFile), \
            f"Output file {args.pathOutputFile} already exists !!! If you want to continue computing scores, please check the --resume option."

    # Load LSTM model
    if args.dict is None:
        pathData = dirname(args.pathLSTMCheckpoint)
    else:
        pathData = dirname(args.dict)
    assert exists(join(pathData, "dict.txt")), \
        f"Dictionary file (dict.txt) not found in {pathData}"
    print("")
    print(f"Loading LSTM model from {args.pathLSTMCheckpoint}...")
    print(f"Path data {pathData}")
    model, task = loadLSTMLMCheckpoint(args.pathLSTMCheckpoint, pathData)
    model.eval()
    print("Model loaded !")

    # Run and save outputs
    print("")
    print(f"Computing log-probabilities and saving results to {args.pathOutputFile}...")
    _ = compute_proba_LSTM(
                        intput_file_seqs, model, task, 
                        batch_size = args.batchSize, gpu = not args.cpu,
                        verbose=False, print_tokens=False,
                        save_to=args.pathOutputFile, file_names=input_file_names)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)