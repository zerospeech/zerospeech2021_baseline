#!/bin/bash
# Generate a complete submission to ZR2021 from the challenge baseline
# On a cluster using SLURM, simply run:
#    sbatch ./generate_submission.sh /path/to/dataset /path/to/output
# This will create /path/to/output/zerospeech2021-baseline-submission.zip
#
#SBATCH --partition=gpu
#SBATCH --nodelist=puck5
#SBATCH --gres=gpu:rtx8000:1

# path to the challenge dataset
dataset=$1

# where to write intermediate data and final scores
output_directory=$2


function usage
{
    echo "Usage: $0 <dataset> <output-directory>"
    exit 0
}


function error
{
    if [ -z "$1" ]
    then
        message="fatal error"
    else
        message="fatal error: $1"
    fi

    echo $message
    exit 1
}


# check input arguments
[ "$1" == "-h" -o "$1" == "-help" -o "$1" == "--help" ] && usage
[ $# -ne 2 ] && usage
[ ! -d $dataset ] && error "dataset not found: $dataset"


start_time=$(date +%s)

# code folder
mkdir -p $output_directory/submission/code
cat <<EOF >  $output_directory/submission/code/README.txt
The source code for the baseline is available at
https://github.com/bootphon/zerospeech2021-baseline
EOF


# meta.yaml
cat <<EOF > $output_directory/submission/meta.yaml
author: Zero Speech Challenge Organizers
affiliation: EHESS, ENS, PSL Research Univerity, CNRS and Inria
description: >
  CPC (trained on librispeech 960), kmeans (trained on librispeech 100),
  BERT (trained on librispeech 960 encoded with the quantized units).
  See https://zerospeech.com/2021 for more details.
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


# lexical and syntactic
for task in lexical syntactic
do
    for kind in dev test
    do
        python ./quantize_audio.py \
               ./checkpoints/CPC-big-kmeans50-BERT/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
               $dataset/$task/$kind \
               $output_directory/quantized/$task/$kind \
               --file_extension .wav \
            || error "task $task $kind failed"

        python ./compute_proba_BERT.py \
               $output_directory/quantized/$task/$kind/quantized_outputs.txt \
               $output_directory/submission/$task/$kind.txt \
               checkpoints/CPC-big-kmeans50-BERT/BERT/BERT_CPC_big_kmeans50.pt \
            || error "task $task $kind failed"
    done
done


# semantic
for kind in dev test
do
    for corpus in synthetic librispeech
    do
        python ./quantize_audio.py \
               ./checkpoints/CPC-big-kmeans50-BERT/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
               $dataset/semantic/$kind/$corpus \
               $output_directory/quantized/semantic/$kind/$corpus \
               --file_extension .wav \
            || error "task semantic $kind $corpus failed"

        python ./build_BERT_features.py \
               $output_directory/quantized/semantic/$kind/$corpus/quantized_outputs.txt \
               $output_directory/submission/semantic/$kind/$corpus \
               checkpoints/CPC-big-kmeans50-BERT/BERT/BERT_CPC_big_kmeans50.pt \
               --hidden_level 4 \
            || error "task semantic $kind $corpus failed"
    done
done


# phonetic
for kind in dev test
do
    for corpus in clean other
    do
        python ./quantize_audio.py \
               ./checkpoints/CPC-big-kmeans50-BERT/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
               $dataset/phonetic/${kind}-${corpus} \
               $output_directory/quantized/phonetic/${kind}-${corpus} \
               --file_extension .wav \
            || error "task phonetic $kind $corpus failed"

        python ./build_1hot_features.py \
               $output_directory/quantized/phonetic/${kind}-${corpus}/quantized_outputs.txt \
               $output_directory/submission/phonetic/${kind}-${corpus} \
            || error "task phonetic $kind $corpus failed"
    done
done


# cleanup and archive creation
find $output_directory/submission -type f -name _info_args.json -delete
cd $output_directory/submission
zip -q -r ../zerospeech2021-submission-baseline-bert.zip .
cd -
# rm -rf $output_directory/{submission,quantized}

stop_time=$(date +%s)
total_time=$(date -u -d "0 $stop_time seconds - $start_time seconds" +"%H:%M:%s")
echo "total time: $total_time"

exit 0
