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

i=1
for name_brain in $(cat $file_list_brains ); do
    name_mask=$(sed -n -e ${i}p $file_list_masks )
    echo ""
    echo "################################################################"
    echo $name_brain
    echo $name_mask
    bash src/pipelines/dhcp/process_single.sh $name_brain $name_mask $s_x $s_y $s_z $downsampling_factor $blur_sigma || exit 1
    let i=$i+1
done

echo "Finished properly"
