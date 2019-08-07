#!/usr/bin/env bash

which vimdiff > /dev/null
if [[ $? -ne 0 ]]; then
    echo "vimdiff not installed"
    exit 1
fi

usage="$0 mod|vmr hr"
mod_or_vmr="$1"
hr="$2"

if [[ $mod_or_vmr == "mod" ]]; then
    dir="mod_files/fpit/oc/vertical"
    fname="FPIT_20180101_${hr}00Z_36.60N_97.49W.mod"
elif [[ $mod_or_vmr == "vmr" ]]; then
    dir="vmr_files/fpit/"
    fname="JL1_20180101${hr}_37N_097W.vmr"
else
    echo $usage
    echo "$mod_or_vmr is not a valid first argument"
    exit 1
fi

file_in="test_input_data/$dir/$fname"
file_out="test_output_data/$dir/$fname"

vimdiff "$file_in" "$file_out"
