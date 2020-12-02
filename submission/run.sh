#!/bin/bash
## Usage: ./run.sh DATASET OUTPUT_DIR [SBATCH_OPTIONS]
##
## Generates the BERT and LSTM baseline submissions for ZeroSpeech 2021 on a
## SLURM cluster. Runs one sbatch job per task to compute. SBATCH_OPTIONS allows
## to specify partitions and resources. Writes results to OUTPUT_DIR/{bert,lstm}
## and logs to OUTPUT_DIR/log.
##
## Example:
##
## ./run.sh /path/to/dataset ./output --partition=gpu --nodelist=puck5 --gres=gpu:rtx8000:1
##
## Parameters:
##
##   DATASET         path to the zerospeech2021 dataset
##   OUTPUT_DIR      directory to write results on
##   SBATCH_OPTIONS  options to pass to sbatch
##
## See https://zerospeech.com/2021 for more details on the baseline.

# absolute path to the directory where this script is
here="$(cd $(dirname "${BASH_SOURCE[0]}") > /dev/null 2>&1 && pwd)"


function usage
{
    sed -nr 's/^## ?//p' ${BASH_SOURCE[0]}
    exit 0
}


# input arguments
[ "$1" == "-h" -o "$1" == "-help" -o "$1" == "--help" ] && usage
[ $# -lt 2 ] && usage

dataset=$1
output_dir=$2
shift; shift
sbatch_options=$@

[ ! -d $dataset ] && error "DATASET not found: $dataset"
output_dir=$(readlink -f $output_dir)


mkdir -p $output_dir/{log,bert,lstm,quantized}


# {bert,lstm}/code folder
mkdir -p $output_dir/{bert,lstm}/code
cat <<EOF >  $output_dir/bert/code/README.txt
The source code for the baseline is available at
https://github.com/bootphon/zerospeech2021_baseline
EOF
cp $output_dir/bert/code/README.txt $output_dir/lstm/code


# bert/meta.yaml
cat <<EOF > $output_dir/bert/meta.yaml
author: BERT Baseline
affiliation: EHESS, ENS, PSL Research Univerity, CNRS and Inria
description: >
  CPC-big (trained on librispeech 960), kmeans (trained on librispeech 100),
  BERT (trained on librispeech 960 encoded with the quantized units). See
  https://zerospeech.com/2021 for more details.
open_source: true
train_set: librispeech 100 and 960
gpu_budget: 1536
parameters:
  phonetic:
    metric: cosine
    frame_shift: 0.01
  semantic:
    metric: cosine
    pooling: max
EOF


# lstm/meta.yaml
cat <<EOF > $output_dir/lstm/meta.yaml
author: LSTM Baseline
affiliation: EHESS, ENS, PSL Research Univerity, CNRS and Inria
description: >
  CPC-big (trained on librispeech 960), kmeans (trained on librispeech 100),
  LSTM. See https://zerospeech.com/2021 for more details.
open_source: true
train_set: librispeech 100 and 960
gpu_budget: 60
parameters:
  phonetic:
    metric: cosine
    frame_shift: 0.01
  semantic:
    metric: cosine
    pooling: min
EOF


# phonetic
for kind in dev test
do
    for corpus in clean other
    do
        sbatch \
            $sbatch_options \
            --job-name=phonetic_${kind}_${corpus} \
            --output=$output_dir/log/%x.log \
            $here/run_phonetic.sh $dataset $output_dir $kind $corpus
    done
done


# lexical and syntactic
for task in lexical syntactic
do
    for kind in dev test
    do
        sbatch \
            $sbatch_options \
            --job-name=${task}_${kind} \
            --output=$output_dir/log/%x.log \
            $here/run_lexical_syntactic.sh $dataset $output_dir $task $kind
    done
done


# semantic
for kind in dev test
do
    for corpus in synthetic librispeech
    do
        sbatch \
            $sbatch_options \
            --job-name=semantic_${kind}_${corpus} \
            --output=$output_dir/log/%x.log \
            $here/run_semantic.sh $dataset $output_dir $kind $corpus
        exit 0
    done
done
