#!/bin/bash

make clean
make

if [ "$#" = "2" ]; then
    if [ "$1" = "0" ];then
        thorq --add --mode single --device gpu/1080 ./vr_seq data/video"$2".bin result"$2".out
    elif [ "$1" = "1" ];then
        thorq --add --mode single --device gpu/1080 ./vr_opencl data/video"$2".bin result"$2".out
    fi
fi
