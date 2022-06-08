#! /bin/sh
#
# process_serial.sh
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
            echo downsampling_factor "$2"
            shift # past argument
            shift # past value
            ;;
        -b|--blur_sigma)
            echo blur_sigma "$2"
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

mkdir -p logs

for name in $(cat $file_list); do
    bash src/pipelines/general/process_single.sh $name $s_x $s_y $s_z $downsampling_factor $blur_sigma || exit 1
done

echo "Finished properly"
