from pathlib import Path
from os.path import exists, join, basename, dirname, abspath
import sys
import argparse

from utils.utils_functions import loadRobertaCheckpoint
from utils.lm_scoring import compute_proba_BERT_mlm_span

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Compute pseudo log-probabilities of quantized units with a trained BERT model.')
    parser.add_argument('pathQuantizedUnits', type=str,
                        help='Path to the quantized units. Each line of the input file must be'
                        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument('pathOutputFile', type=str,
                        help='Path to the output file containing scores.')
    parser.add_argument('pathBERTCheckpoint', type=str,
                        help='Path to the trained fairseq BERT(RoBERTa) model.')
    parser.add_argument('--dict', type=str,
                       help='Path to the dictionary file (dict.txt) used to train the BERT model'
                       '(if not speficied, look for dict.txt in the model directory)')
    parser.add_argument('--decoding_span_size', type=int, default=15,
                        help='The decoding span size (M_d) parameter used to compute'
                        'the pseudo-probability (default: 15).')
    parser.add_argument('--temporal_sliding_size', type=int, default=5,
                        help='The temporal sliding size (Delta_t) parameter used to'
                        'compute the pseudo-probability (defaut: 5).')
    parser.add_argument('--no_overlap', action="store_true",
                        help='If specified, not overlap the masking spans when computing the'
                        'pseudo-probability (temporal_sliding_size is set to decoding_span_size)')
    parser.add_argument('--batchsen_size', type=int, default=32,
                        help='The number of sentences to be considered in each outer batch'
                        '(batch of sentences) (defaut: 32). Decrease this for longer sentences (BLIMP).')
    parser.add_argument('--inner_batch_size', type=int, default=128,
                        help='For each sentence, the model has to compute the outputs of many different'
                        'masked sequences. This parameter controls the size of the inner batches for'
                        'each outer batch (defaut: 128). Decrease this for longer sentences (BLIMP).')
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
    args.pathBERTCheckpoint = abspath(args.pathBERTCheckpoint)
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

    assert len(intput_file_seqs) > 0, \
        "No file to compute probability!"

    # Load BERT model
    if args.dict is None:
        pathData = dirname(args.pathBERTCheckpoint)
    else:
        pathData = dirname(args.dict)
    assert exists(join(pathData, "dict.txt")), \
        f"Dictionary file (dict.txt) not found in {pathData}"
    print("")
    print(f"Loading RoBERTa model from {args.pathBERTCheckpoint}...")
    print(f"Path data {pathData}")
    roberta = loadRobertaCheckpoint(
                args.pathBERTCheckpoint, 
                pathData, 
                from_pretrained=False)
    roberta.eval()  # disable dropout (or leave in train mode to finetune)
    print("Model loaded !")

    # Run and save outputs
    print("")
    print(f"Computing log-probabilities and saving results to {args.pathOutputFile}...")
    _ = compute_proba_BERT_mlm_span(
                            intput_file_seqs, roberta, tokenized=True,
                            decoding_span_size=args.decoding_span_size, temporal_sliding_size = args.temporal_sliding_size,
                            span_overlap=not args.no_overlap,
                            batchsen_size=args.batchsen_size, inner_batch_size = args.inner_batch_size,
                            gpu=not args.cpu, print_tokens=False, verbose=False, print_shape_statistics=False,
                            save_to=args.pathOutputFile, file_names=input_file_names)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
