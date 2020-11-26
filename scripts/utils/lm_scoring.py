import gc
from time import time
from os.path import exists

import torch

def compute_proba_BERT_mlm_span(
                            sequences, roberta, tokenized=True,
                            decoding_span_size=15, temporal_sliding_size = 5,
                            span_overlap=True,
                            batchsen_size=32, inner_batch_size = 128,
                            gpu=False, print_tokens=False, verbose=False,
                            print_shape_statistics=False,
                            save_to=None, file_names=None):
    """
    Compute the pseudo log-proba of a list of sentences with span-masked-language-model-scoring style as
    described in the baseline system of The Zero Resource Speech Benchmark 2021 (see paper for the formula).

    Parameters
    ----------
    sequences : list of strings/string
        The list of the input sentences.
    roberta : fairseq's RobertaHubInterface
        The trained RoBERTa model on the inputs.
    tokenized : boolean
        Wether the input sentences are already tokenized (separated by spaces). If False, use the roberta encoder
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
    print_shape_statistics : bool
        Wether to print shape statistics when printing batches. Should only be used for debugging.
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
        masked_idx = roberta.task.source_dictionary.indices[masked_token]
        pad_idx = roberta.task.source_dictionary.pad()

        # Compute the input sequences of tokens
        masked_sentence_tokens_list = []
        sequences_list = []         # to retrieve the sentences when computing logproba
        for sentence in sentences:
            if tokenized:
                sentence_tokens = roberta.task.source_dictionary.encode_line("<s> " + sentence, append_eos=True, add_if_not_exist=False)

                if print_tokens:
                    print("|".join([roberta.task.source_dictionary.symbols[tok] for tok in sentence_tokens]))
            else:
                bpe_sentence = '<s> ' + roberta.bpe.encode(sentence) + ' </s>'
                sentence_tokens = roberta.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)

                if print_tokens:
                    print("|".join([roberta.decode(tok.unsqueeze(0)) for tok in sentence_tokens]))

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

            outputs_chk = roberta.model(inputs_chk)[0]

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
            roberta = roberta.cuda()

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

            start_time = time()
            for i in range(n_batch):
                sequences_batch = sequences[i*batchsen_size : min((i+1)*batchsen_size, len(sequences))]
                with torch.no_grad():
                    logproba_batch, shape_statistics = compute_proba_batchsen(sequences_batch)
                logproba_all.extend(logproba_batch)

                if save_to is not None:
                    file_names_batch = file_names[i*batchsen_size : min((i+1)*batchsen_size, len(sequences))]
                    outLines = []
                    for fname, score in zip(file_names_batch, logproba_batch):
                        outLines.append(" ".join([fname, str(score)]))
                    outLines = "\n".join(outLines)
                    with open(save_to, 'a') as f:
                        if addEndLine:
                            f.write("\n"+outLines)
                        else:
                            f.write(outLines)
                            addEndLine = True

                if print_shape_statistics:
                    print("Batch {:d}/{:d}. Input shapes: {} Done in {:4f} s.\t\t\t".format(
                        i+1, n_batch,
                        shape_statistics,
                        time() - start_time))
                else:
                    print("Batch {:d}/{:d}. Done in {:4f} s.\t\t\t".format(
                        i+1, n_batch,
                        time() - start_time), 
                        end="\r")
            print("\nDone all in {:4f} s.".format(time() - start_time))
        else:
            start_time = time()
            logproba_all, shape_statistics = compute_proba_batchsen(sequences)

            if save_to is not None:
                outLines = []
                for fname, score in zip(file_names, logproba_all):
                    outLines.append(" ".join([fname, str(score)]))
                outLines = "\n".join(outLines)
                with open(save_to, 'a') as f:
                    if addEndLine:
                        f.write("\n"+outLines)
                    else:
                        f.write(outLines)
                        addEndLine = True

            print("Done all in {:4f} s.".format(time() - start_time))

        # Release all GPU memory
        if gpu:
            roberta = roberta.cpu()
            gc.collect()
            torch.cuda.empty_cache()

    except:
        # Release all GPU memory
        if gpu:
            roberta = roberta.cpu()
            gc.collect()
            torch.cuda.empty_cache()

        raise

    return logproba_all

def compute_proba_LSTM(
        sequences, model, task, 
        batch_size = 128, gpu = True,
        verbose=False, print_tokens=False,
        save_to=None, file_names=None):
    """
    Compute the pseudo log-proba of a list of sentences from a LSTM language model estimated with the chain rule.
    P(abcd)=P(a)P(b|a)P(c|ab)P(d|abc)

    Parameters
    ----------
    sequences : list of strings/string
        The list of the input sentences.
    model : fairseq's lstm_lm model
        The trained LSTM model on the inputs.
    task : fairseq's language_modeling task
        The task of the model, used to encode the input sequence with the corresponding task.dictionary.
    batch_size : int
        The number of sentences to be considered in each batch.
    gpu : bool
        Wether to use GPU.
    print_tokens : bool
        Wether to print explicitly the input tokens to the BERT model. Should only be used for debugging.
    verbose : bool
        Wether to print the scores of the input sequences. Should only be used for debugging.
    print_shape_statistics : bool
        Wether to print shape statistics when printing batches. Should only be used for debugging.
    save_to (optional) : str
        Path to save the outputs.
    file_names (optional) : list of strings
        If save_to is not None, a list of corresponding file names must be given.

    Return
    -------
    logproba_all : list of floats
        The pseudo log-probabilities of the input sentences.

    """
    try:  # In case of any errors, release GPU memory
        if gpu:
            model = model.cuda()
        
        def compute_proba_batch(sequences):
            pad_idx = task.source_dictionary.pad()

            # Load tensor
            input_sequences = []
            input_lengths = []
            for sequence in sequences:
                # Convert from string to list of units
                sentence_tokens = task.source_dictionary.encode_line("<s> " + sequence, append_eos=True, add_if_not_exist=False).long()

                if print_tokens:
                    print("|".join([task.source_dictionary.symbols[tok] for tok in sentence_tokens]))

                # Update
                input_lengths.append(len(sentence_tokens))
                input_sequences.append(sentence_tokens)
            
            sequences_inputs = torch.nn.utils.rnn.pad_sequence(input_sequences, batch_first = False, padding_value = pad_idx).t()

            if gpu:
                sequences_inputs = sequences_inputs.cuda()

            # Compute output & probabilities
            output_ts, _ = model(sequences_inputs)
            output_ts = output_ts.softmax(dim=-1)

            # Compute scores
            logproba_list = []
            for j, sequence in enumerate(sequences):
                logproba=0.
                if verbose:
                    sequence = ["BOS"] + tokenize(sequence) + ["EOS"]
                for i, ch_idx in enumerate(sequences_inputs[j][1:]):
                    score = output_ts[j,i,ch_idx].log()
                    if verbose:
                        print(sequence[:i+1], sequence[i+1], score)
                    logproba += score
                    if i == input_lengths[j] - 2:
                        break
                if verbose:
                    if is_list:
                        print("Log proba score for '"+" ".join(sequence[1:-1])+"' :", logproba)
                    else:
                        print("Log proba score for '"+sequence[1:-1]+"' :", logproba)
                logproba_list.append(logproba.data.item())
                
            return logproba_list
        
        if save_to is not None:
            assert file_names is not None and len(file_names) == len(sequences), \
                "Number of input sequences and number of files must be equal!"
            addEndLine = False # to add end line (\n) to first line or not
            if exists(save_to):
                with open(save_to, 'r') as f:
                    lines = [line for line in f]
                if len(lines) > 0 and not lines[-1].endswith("\n"):
                    addEndLine = True
        
        print("Number of sequences: {}".format(len(sequences)))
        if batch_size > 0:
            logproba_all = []
            n_batch = len(sequences)//batch_size
            if len(sequences) % batch_size != 0:
                n_batch += 1

            start_time = time()
            for i in range(n_batch):
                sequences_batch = sequences[i*batch_size : min((i+1)*batch_size, len(sequences))]
                logproba_batch = compute_proba_batch(sequences_batch)

                if save_to is not None:
                    file_names_batch = file_names[i*batch_size : min((i+1)*batch_size, len(sequences))]
                    outLines = []
                    for fname, score in zip(file_names_batch, logproba_batch):
                        outLines.append(" ".join([fname, str(score)]))
                    outLines = "\n".join(outLines)
                    with open(save_to, 'a') as f:
                        if addEndLine:
                            f.write("\n"+outLines)
                        else:
                            f.write(outLines)
                            addEndLine = True

                logproba_all.extend(logproba_batch)
                print("Batch {:d}/{:d}. Done in {:4f} s.\t\t\t".format(
                                                                    i+1, n_batch,
                                                                    time() - start_time), end = "\r")

            print("\nDone all in {:4f} s.".format(time() - start_time))
        else:
            start_time = time()
            logproba_all = compute_proba_batch(sequences)

            if save_to is not None:
                outLines = []
                for fname, score in zip(file_names, logproba_all):
                    outLines.append(" ".join([fname, str(score)]))
                outLines = "\n".join(outLines)
                with open(save_to, 'a') as f:
                    if addEndLine:
                        f.write("\n"+outLines)
                    else:
                        f.write(outLines)
                        addEndLine = True

            print("Done all in {:4f} s.".format(time() - start_time))

        # Release all GPU memory
        if gpu:
            gc.collect()
            torch.cuda.empty_cache()
    
    except:
        # Release all GPU memory
        if gpu:
            gc.collect()
            torch.cuda.empty_cache()
        
        raise
    
    return logproba_all