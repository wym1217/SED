#!/bin/bash

DATA="./sound_event_detection_data"
mkdir {dev,eval,metadata}

# development set
echo "Preparing development set"
python prepare_wav_csv.py "$DATA/audio/train/weak" "dev/wav.csv"
ln -s $(realpath "$DATA/label/train/weak.csv") "dev/label.csv"

# evaluation set
echo "Preparing evaluation set"
python prepare_wav_csv.py "$DATA/audio/eval" "eval/wav.csv"
ln -s $(realpath "$DATA/label/eval/eval.csv") "eval/label.csv"

cp "$DATA/label/class_label_indices.txt" "metadata/class_label_indices.txt"

