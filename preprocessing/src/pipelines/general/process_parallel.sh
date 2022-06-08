#! /bin/sh
#
# process_list.sh
# Copyright (C) 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.
#

# Parse arguments
POSITIONAL=()
while [[ $# -gt 0 ]]
do
    key="$1"

    case $key in
        -l|--list)
            file_list="$2"
            shift # past argument
            shift # past value
            ;;
        -s|--sizes)
            s_x="$2"
            s_y="$3"
            s_z="$4"
            shift # past argument
            shift # past value
            shift # past value
            shift # past value
            ;;
        -f|--factor)
            downsampling_factor="$2"
            shift # past argument
            shift # past value
            ;;
        -b|--blur_sigma)
            blur_sigma="$2"
            shift # past argument
            shift # past value
            ;;
        *)    # unknown option
            POSITIONAL+=("$1") # save it in an array for later
            shift # past argument
            ;;
    esac
done

#Get number of available GPUS
available_gpus=($(echo $CUDA_VISIBLE_DEVICES | sed -e "s:,: :g"))
if [ -z "$available_gpus" ]
then
    n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    # The count starts at zero, so we need to stop one earlier
    let n_gpus=$n_gpus-1
    available_gpus=($(seq 0 $n_gpus) )
fi
n_gpus=${#available_gpus[@]}
echo Using gpus ${available_gpus[@]} for a total of $n_gpus GPUs

# Time to get and break the list
mkdir  -p tmp/lists
split ${file_list} --number=l/${n_gpus} tmp/lists/list

# Now for each list, we process in a different file

i=0
for list in $(ls tmp/lists/list*); do 
    gpu=${available_gpus[i]}
    CUDA_VISIBLE_DEVICES=${gpu} bash src/pipelines/general/process_serial.sh  -l $list  -s $s_x $s_y $s_z -f $downsampling_factor -b $blur_sigma >& logs/process_$i.txt &
    let i=$i+1
done

wait

# A bit of cleanup
# rm -r tmp
