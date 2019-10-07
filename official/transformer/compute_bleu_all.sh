#!/bin/bash

source activate ompi_hvd

if [ -z "$1" ]; then
  echo "Translation directory is not provided"
  exit 1
elif [ -z "$2" ]; then
  echo "Reference file is not provided"
  exit 1
fi

#echo "functionality doesn't work"

TRANSLATION_DIR=$1
reference=$2
CODE_DIR=$3
output_file=$TRANSLATION_DIR/all_bleu_results.o

rm $output_file

export PYTHONPATH="$PYTHONPATH:$HOME/models"

echo "Computing BLEU scores" >$output_file
echo "\n" >> $output_file

for filename in $TRANSLATION_DIR/*; do
  echo $filename | sed -e 's/.*model.\(.*\)_translation.*/\1/' >> $output_file
  #echo "Filename: $filename" >> $output_file 
  python $CODE_DIR/compute_bleu.py --reference=$reference --translation=$filename &>> $output_file

done
