#!/bin/bash

set -e

command -v curl >/dev/null 2>&1 || { echo 'curl is missing' >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo 'unzip is missing' >&2; exit 1; }

mkdir -p _datasets/FSD50K && cd _datasets/FSD50K

set -x
set +e

curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.metadata.zip'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.doc.zip'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z01'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z02'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z03'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z04'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z05'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01'
curl -fOL -C - 'https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip'

unzip 'FSD50K.ground_truth.zip'
unzip 'FSD50K.metadata.zip'
unzip 'FSD50K.doc.zip'
zip -s- 'FSD50K.dev_audio.zip' -O 'FSD50K.dev_audio.full.zip'
unzip 'FSD50K.dev_audio.full.zip'
rm 'FSD50K.dev_audio.full.zip'
zip -s- 'FSD50K.eval_audio.zip' -O 'FSD50K.eval_audio.full.zip'
unzip 'FSD50K.eval_audio.full.zip'
rm 'FSD50K.eval_audio.full.zip'
