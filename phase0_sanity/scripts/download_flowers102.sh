#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
dataset_root="${1:-${repo_root}/data/flowers102}"
raw_root="${dataset_root}/raw"
extracted_root="${dataset_root}/extracted"
python_bin="${IBKD_SEG_DOWNLOAD_PYTHON:-python3}"
connections="${IBKD_SEG_DOWNLOAD_CONNECTIONS:-8}"

mkdir -p "${raw_root}" "${extracted_root}"

download() {
  local url="$1"
  local output="$2"
  local expected_size="$3"
  PYTHONPATH="${repo_root}/src" "${python_bin}" -m ibkd_seg.phase0.download \
    --url "${url}" \
    --output "${output}" \
    --size "${expected_size}" \
    --connections "${connections}"
}

download \
  "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz" \
  "${raw_root}/102flowers.tgz" \
  "344862509"
download \
  "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz" \
  "${raw_root}/102segmentations.tgz" \
  "203577493"
download \
  "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat" \
  "${extracted_root}/imagelabels.mat" \
  "502"
download \
  "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat" \
  "${extracted_root}/setid.mat" \
  "14989"

tar -xzf "${raw_root}/102flowers.tgz" -C "${extracted_root}"
tar -xzf "${raw_root}/102segmentations.tgz" -C "${extracted_root}"

echo "Flowers-102 extracted under ${extracted_root}"
