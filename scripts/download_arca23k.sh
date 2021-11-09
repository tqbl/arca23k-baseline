#!/bin/bash

set -e

command -v curl >/dev/null 2>&1 || { echo 'curl is missing' >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo 'unzip is missing' >&2; exit 1; }

mkdir -p _datasets/ARCA23K && cd _datasets/ARCA23K

curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K.ground_truth.zip'
curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K-FSD.ground_truth.zip'
curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K.metadata.zip'
curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K.audio.z01'
curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K.audio.z02'
curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K.audio.z03'
curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K.audio.z04'
curl -fOL 'https://zenodo.org/record/5117901/files/ARCA23K.audio.zip'
curl -fOL 'https://zenodo.org/record/5117901/files/ATTRIBUTION'
curl -fOL 'https://zenodo.org/record/5117901/files/LICENSE'

unzip 'ARCA23K.ground_truth.zip'
unzip 'ARCA23K-FSD.ground_truth.zip'
unzip 'ARCA23K.metadata.zip'
zip -s- 'ARCA23K.audio.zip' -O 'ARCA23K.audio.full.zip'
unzip 'ARCA23K.audio.full.zip'
rm 'ARCA23K.audio.full.zip'
