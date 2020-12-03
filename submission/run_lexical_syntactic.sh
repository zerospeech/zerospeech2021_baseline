#!/bin/bash
## Usage: run_lexical_syntactic.sh DATASET OUTPUT_DIR TASK KIND
##
## Generates the lexical or syntactic submission for BERT and LSTM baselines.
## Writes OUTPUT_DIR/{bert,lstm}/TASK/KIND.txt
##
## Parameters:
##  DATASET     path to the zerospeech2021 dataset
##  OUTPUT_DIR  directory to write results on
##  TASK        must be 'lexical' or 'syntactic'
##  KIND        must be 'dev' or 'test'


function here
{
    if [ -n $SLURM_JOB_ID ]
    then
        echo $(dirname $(scontrol show job $SLURM_JOBID \
                             | awk -F= '/Command=/{print $2}' \
                             | cut -d' ' -f1))
    else
        echo $(cd $(dirname "${BASH_SOURCE[0]}") > /dev/null 2>&1 && pwd)
    fi
}


function usage
{
    sed -nr 's/^## ?//p' ${BASH_SOURCE[0]}
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
    echo "finished at $(date)"
    exit 1
}


if [ -n $SLURM_JOBID ]
then
    module load anaconda/3
    source activate zerospeech2021_baseline
else
    conda activate zerospeech2021_baseline
fi


scripts_dir=$(readlink -f $(here)/../scripts)
checkpoints_dir=$(readlink -f $(here)/../checkpoints)


dataset=$1    # path to zerospeech2021 dataset
output_dir=$2 # path to the output directory
task=$3       # lexical or syntactic
kind=$4       # dev or test


# check input arguments
[ "$1" == "-h" -o "$1" == "-help" -o "$1" == "--help" ] && usage
[ $# -ne 4 ] && usage

[ ! -d $dataset ] \
    && error "DATASET not found: $dataset"

[ $task != "lexical" -a $task != "syntactic" ] \
    && error "TASK must be lexical or syntactic, it is: $task"

[ $kind != "dev" -a $kind != "test" ] \
    && error "KIND must be dev or test, it is: $kind"


echo "running on $(hostname)"
echo "started at $(date)"
echo "python: $(which python)"
echo "dataset: $dataset"
echo "output: $output_dir"
echo "parameters: $task $kind"


# quantization
python $scripts_dir/quantize_audio.py \
       $checkpoints_dir/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
       $dataset/$task/$kind \
       $output_dir/quantized/$task/$kind \
       --file_extension .wav \
    || error "$task quantization failed ($kind)"

# pseudo probabilities for bert and lstm
for baseline in bert lstm
do
    BASELINE=$(echo $baseline | tr [a-z] [A-Z])
    python $scripts_dir/compute_proba_${BASELINE}.py \
           $output_dir/quantized/$task/$kind/quantized_outputs.txt \
           $output_dir/$baseline/$task/$kind.txt \
           $checkpoints_dir/CPC-big-kmeans50/${BASELINE}/${BASELINE}_CPC_big_kmeans50.pt \
        || error "$task probabilities failed ($baseline $kind)"
done


echo "finished at $(date)"

exit 0
