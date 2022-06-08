#! /bin/sh #
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
       -l|--list_brains)
            file_list_brains="$2"
            shift # past argument
            shift # past value
            ;;
        -m|--list_masks)
            file_list_masks="$2"
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
        -n|--ncpus)
            n_cpus="$2"
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
if [ -z "$n_cpus" ]
then
    n_cpus=$( nproc --all)
    let n_cpus=${n_cpus}
fi
echo Using $n_cpus CPUs

# Time to get and break the list
mkdir  -p tmp/lists_brains
mkdir  -p tmp/lists_masks
split ${file_list_brains} --number=l/${n_cpus} tmp/lists_brains/list
split ${file_list_masks} --number=l/${n_cpus} tmp/lists_masks/list

# Now for each list, we process in a different file

i=0
for list_brain in $(ls tmp/lists_brains/list*); do 
    list_mask=$(ls tmp/lists_masks/list*)[$i]
    bash src/pipelines/dhcp/process_serial.sh  -l $list_brain -m $list_mask -s $s_x $s_y $s_z -f $downsampling_factor -b $blur_sigma >& logs/process_$i.txt &
    let i=$i+1
done

wait

# A bit of cleanup
# rm -r tmp
