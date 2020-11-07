import gc
from time import time
from os.path import exists, join, basename, dirname
import sys
import argparse

import torch
from fairseq.models.roberta import RobertaModel
    

def compute_proba_BERT_mlm_span(
                            sequences, model, tokenized=True,
                            decoding_span_size=15, temporal_sliding_size = 5,
                            span_overlap=True,
                            batchsen_size=32, inner_batch_size = 128, 
                            gpu=False, print_tokens=False, verbose=False,
                            save_to=None, file_names=None):
    """
    Compute the pseudo log-proba of a list of sentences with span-masked-language-model-scoring style as
    described in the baseline system of The Zero Resource Speech Benchmark 2021 (see paper for the formula).

    Parameters
    ----------
    sequences : list of strings/string
        The list of the input sentences.
    model : fairseq BERT(RoBERTa) model
        The trained RoBERTa model on the inputs.
    tokenized : boolean
        Wether the input sentences are already tokenized (separated by spaces). If False, use the model encoder
        to encode the input sentences (set this to False for pretrained RoBERTa models).
    decoding_span_size: int
        The decoding span size (M_d) parameter used to compute the pseudo-probability.
    temporal_sliding_size : int
        The temporal sliding size (Delta_t) parameter used to compute the pseudo-probability.
    span_overlap : bool
        Wether to overlap the masking spans when computing the pseudo-probability. If False, the 
        temporal_sliding_size is set to decoding_span_size.
    batchsen_size : int
        The number of sentences to be considered in each outer batch (batch of sentences).
    inner_batch_size : int
        For each sentence, the model has to compute the outputs of many different masked sequences.
        This parameter controls the size of the inner batches for each outer batch.
    gpu : bool
        Wether to use GPU.
    print_tokens : bool
        Wether to print explicitly the input tokens to the BERT model. Should only be used for debugging.
    verbose : bool
        Wether to print the scores of the input sequences. Should only be used for debugging.
    save_to (optional) : str 
        Path to save the outputs.
    file_names (optional) : list of strings 
        If save_to is not None, a list of corresponding file names must be given.

    Return
    -------
    logproba_all : list of floats
        The pseudo log-probabilities of the input sentences.

    """
    def compute_proba_batchsen(sentences):
        # Compute the id of the mask
        masked_token =  '<mask>'
        masked_idx = model.task.source_dictionary.indices[masked_token]
        pad_idx = model.task.source_dictionary.pad()

        # Compute the input sequences of tokens
        masked_sentence_tokens_list = []
        sequences_list = []         # to retrieve the sentences when computing logproba
        for sentence in sentences:
            if tokenized:
                sentence_tokens = model.task.source_dictionary.encode_line("<s> " + sentence, append_eos=True, add_if_not_exist=False)
                
                if print_tokens:
                    print("|".join([model.task.source_dictionary.symbols[tok] for tok in sentence_tokens]))
            else:
                bpe_sentence = '<s> ' + model.bpe.encode(sentence) + ' </s>'
                sentence_tokens = model.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
                
                if print_tokens:
                    print("|".join([model.decode(tok.unsqueeze(0)) for tok in sentence_tokens]))

            ln = len(sentence_tokens) - 2

            # Get actual span size, in case of short sequences
            actual_span_size = min(decoding_span_size, ln)

            # Overlap ?
            if span_overlap:
                step_size = temporal_sliding_size
            else:
                step_size  = max(actual_span_size,1)

            # Start masking out tokens
            for idx in range(0, ln-actual_span_size+1, step_size):
                masked_sentence_tokens = sentence_tokens.clone().long()
                masked_sentence_tokens[idx+1:idx+1+actual_span_size] = masked_idx
                masked_sentence_tokens_list.append(masked_sentence_tokens.clone())

            sequences_list.append(sentence_tokens)

        # Divide the sequences into batches
        if inner_batch_size > 0:
            inputs_chunks = []
            n_batch = len(masked_sentence_tokens_list)//inner_batch_size

            for i in range(n_batch):
                sequences_inputs = torch.nn.utils.rnn.pad_sequence(masked_sentence_tokens_list[i*inner_batch_size : (i+1)*inner_batch_size], 
                                                                    batch_first = False, padding_value = pad_idx).t()
                inputs_chunks.append(sequences_inputs)

            if len(masked_sentence_tokens_list) % inner_batch_size != 0:
                sequences_inputs = torch.nn.utils.rnn.pad_sequence(masked_sentence_tokens_list[n_batch*inner_batch_size : ], 
                                                                    batch_first = False, padding_value = pad_idx).t()
                inputs_chunks.append(sequences_inputs)
        else:
            sequences_inputs = torch.nn.utils.rnn.pad_sequence(masked_sentence_tokens_list, batch_first = False, padding_value = pad_idx).t()
            inputs_chunks = [sequences_inputs]

        # Compute the output by batch
        outputs_list = []
        shape_statistics = ""
        for i, inputs_chk in enumerate(inputs_chunks):
            shape_statistics += "{} - ({}, {}) | ".format(inputs_chk.shape[0] * inputs_chk.shape[1], inputs_chk.shape[0], inputs_chk.shape[1])
            if verbose:
                print("{} - ({}, {}) | ".format(inputs_chk.shape[0] * inputs_chk.shape[1], inputs_chk.shape[0], inputs_chk.shape[1]))

            if gpu:
                inputs_chk = inputs_chk.cuda()
            
            outputs_chk = model.model(inputs_chk)[0]
            
            # Release all GPU memory
            if gpu:
                # outputs_chk = outputs_chk.cpu()
                # inputs_chk = inputs_chk.cpu()
                gc.collect()
                torch.cuda.empty_cache()

            outputs_list.extend(outputs_chk.unbind())

        # Compute log proba
        logproba_list = []
        i = 0
        for sentence_tokens in sequences_list:
            ln = len(sentence_tokens) - 2

            # Get actual span size, in case of short sequences
            actual_span_size = min(decoding_span_size, ln)

            # Overlap ?
            if span_overlap:
                step_size = temporal_sliding_size
            else:
                step_size  = max(actual_span_size,1)

            logproba = 0.
            for idx in range(0, ln-actual_span_size+1, step_size):
                score = 0.
                for idx_masked in range(idx+1, idx+1+actual_span_size):
                    score += outputs_list[i][idx_masked].softmax(0)[sentence_tokens[idx_masked]].log()
                if verbose:
                    print(score)
                logproba += score
                i += 1

            if logproba != 0.:
                logproba_list.append(logproba.data.item())
            else:
                logproba_list.append(0.)

        return logproba_list, shape_statistics

    try:  # In case of any errors, release GPU memory (useful for notebooks)
        if gpu:
            model = model.cuda()

        if type(sequences) == str:
            sequences = [sequences]

        if save_to is not None:
            assert file_names is not None and len(file_names) == len(sequences), \
                "Number of input sequences and number of files must be equal!"
            addEndLine = False # to add end line (\n) to first line or not
            if exists(save_to):
                with open(save_to, 'r') as f:
                    lines = [line for line in f]
                if len(lines) > 0 and not lines[-1].endswith("\n"):
                    addEndLine = True

        print("")
        print(f"Parameters: decoding_span_size={decoding_span_size}, temporal_sliding_size={temporal_sliding_size}, span_overlap={span_overlap}")
        print("Number of sequences: {}".format(len(sequences)))
        if batchsen_size > 0:
            logproba_all = []
            n_batch = len(sequences)//batchsen_size
            if len(sequences) % batchsen_size != 0:
                n_batch += 1

            start_time_batch_zero = time()
            for i in range(n_batch):
                start_time = time()
                sequences_batch = sequences[i*batchsen_size : min((i+1)*batchsen_size, len(sequences))]
                with torch.no_grad():
                    logproba_batch, shape_statistics = compute_proba_batchsen(sequences_batch)
                logproba_all.extend(logproba_batch)

                if save_to is not None:
                    file_names_batch = file_names[i*batchsen_size : min((i+1)*batchsen_size, len(sequences))]
                    outLines = []
                    for fname, score in zip(file_names_batch, logproba_batch):
                        outLines.append("\t".join([fname, str(score)]))
                    outLines = "\n".join(outLines)
                    with open(save_to, 'a') as f:
                        if addEndLine:
                            f.write("\n"+outLines)
                        else:
                            f.write(outLines)
                            addEndLine = True
                
                print("Batch {:d}/{:d}. Input shapes: {} Done in {:4f} s.\t\t\t".format(
                                                                    i+1, n_batch,
                                                                    shape_statistics,
                                                                    time() - start_time), end = "\r")
            print("\nDone all in {:4f} s.".format(time() - start_time_batch_zero))
        else:
            start_time = time()
            logproba_all, shape_statistics = compute_proba_batchsen(sequences)
            print("Done all in {:4f} s.".format(time() - start_time))

        # Release all GPU memory
        if gpu:
            model = model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
    
    except:
        # Release all GPU memory
        if gpu:
            model = model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        
        raise

    return logproba_all

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Compute pseudo log-probabilities of quantized units with a trained BERT model.')
    parser.add_argument('input', type=str,
                        help='Path to the quantized units. Each line of the input file must be'
                        'of the form file_name[tab]pseudo_units (ex. hat  1,1,2,3,4,4)')
    parser.add_argument('output', type=str,
                        help='Path to the output file containing scores.')
    parser.add_argument('checkpointBERTmodel', type=str,
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

    # Load input file
    print("")
    print(f"Reading input file from {args.input}")
    input_file_names = []
    intput_file_seqs = []
    with open(args.input, 'r') as f:
        for line in f:
            file_name, file_seq = line.strip().split("\t")
            # Convert sequence to the desired input form
            file_seq = file_seq.replace(",", " ")
            # Add to lists
            input_file_names.append(file_name)
            intput_file_seqs.append(file_seq)
    print(f"Found {len(input_file_names)} sequences!")
    
    # Continue
    if args.resume:
        if exists(args.output):
            existing_file_names = []
            with open(args.output, 'r') as f:
                lines = [line for line in f]
            for line in lines:
                file_name, score = line.strip().split("\t")
                existing_file_names.append(file_name)
            assert input_file_names[:len(existing_file_names)] == existing_file_names, \
                "The file names in the existing output file do not match the input file!!"
            input_file_names = input_file_names[len(existing_file_names):]
            intput_file_seqs = intput_file_seqs[len(existing_file_names):]
            print(f"Found existing output file, continue to compute scores of {len(intput_file_seqs)} sequences left!")
    else:
        assert not exists(args.output), \
            f"Output file {args.output} already exists !!! If you want to continue computing scores, please check the --resume option."

    # Load BERT model
    if args.dict is None:
        PATH_DATA = dirname(args.checkpointBERTmodel)
    else:
        PATH_DATA = dirname(args.dict)
    assert exists(join(PATH_DATA, "dict.txt")), \
        f"Dictionary file (dict.txt) not found in {PATH_DATA}"
    print("")
    print(f"Loading RoBERTa model from {args.checkpointBERTmodel}...")
    print(f"Path data {PATH_DATA}")
    roberta = RobertaModel.from_pretrained(dirname(args.checkpointBERTmodel), basename(args.checkpointBERTmodel), PATH_DATA)
    roberta.eval()  # disable dropout (or leave in train mode to finetune)
    print("Model loaded !")

    # Run and save outputs
    print("")
    print(f"Computing log-probabilities and saving results to {args.output}...")
    _ = compute_proba_BERT_mlm_span(
                            intput_file_seqs, roberta, tokenized=True,
                            decoding_span_size=args.decoding_span_size, temporal_sliding_size = args.temporal_sliding_size,
                            span_overlap=not args.no_overlap,
                            batchsen_size=args.batchsen_size, inner_batch_size = args.inner_batch_size, 
                            gpu=not args.cpu, print_tokens=False, verbose=False,
                            save_to=args.output, file_names=input_file_names)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)