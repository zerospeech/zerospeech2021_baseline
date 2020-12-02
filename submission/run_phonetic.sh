#!/bin/bash
## Usage: run_phonetic.sh DATASET OUTPUT_DIR KIND CORPUS
##
## Generates the phonetic baseline submission (same for BERT and LSTM). Writes
## OUTPUT_DIR/{bert,lstm}/phonetic/KIND-CORPUS/*.txt
##
## Parameters:
##  DATASET     path to the zerospeech2021 dataset
##  OUTPUT_DIR  directory to write results on
##  KIND        must be 'dev' or 'test'
##  CORPUS      must be 'clean' or 'other'


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
kind=$3       # dev or test
corpus=$4     # clean or other


# check input arguments
[ "$1" == "-h" -o "$1" == "-help" -o "$1" == "--help" ] && usage
[ $# -ne 4 ] && usage

[ ! -d $dataset ] \
    && error "DATASET not found: $dataset"

[ $kind != "dev" -a $kind != "test" ] \
    && error "KIND must be dev or test, it is: $kind"

[ $corpus != "clean" -a $corpus != "other" ] \
    && error "CORPUS must be clean or other, it is: $corpus"


echo "running on $(hostname)"
echo "started at $(date)"
echo "python: $(which python)"
echo "dataset: $dataset"
echo "output: $output_dir"
echo "parameters: $kind $corpus"


# quantization
python $scripts_dir/quantize_audio.py \
       $checkpoints_dir/CPC-big-kmeans50/clustering_kmeans50/clustering_CPC_big_kmeans50.pt \
       $dataset/phonetic/${kind}-${corpus} \
       $output_dir/quantized/phonetic/${kind}-${corpus} \
       --file_extension .wav \
    || error "phonetic quantization failed (${kind}-${corpus})"


# features building
python $scripts_dir/build_1hot_features.py \
       $output_dir/quantized/phonetic/${kind}-${corpus}/quantized_outputs.txt \
       $output_dir/bert/phonetic/${kind}-${corpus} \
    || error "phonetic features failed (${kind}-${corpus})"


# cleanup
find $output_dir/bert/phonetic/${kind}-${corpus} -type f -name _info_args.json -delete


# symlink bert to lstm
mkdir -p $output_dir/lstm/phonetic
ln -s $output_dir/bert/phonetic/${kind}-${corpus} $output_dir/lstm/phonetic/${kind}-${corpus}


echo "finished at $(date)"

exit 0
