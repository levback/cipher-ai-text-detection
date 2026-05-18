#!/bin/bash
# Save as download_all.sh and run: bash download_all.sh
cd literature

# Surveys
wget -q https://arxiv.org/pdf/2310.14724 -O surveys/wu2023_survey.pdf
wget -q https://arxiv.org/pdf/2306.05871 -O surveys/weber_wulff2023.pdf
wget -q https://arxiv.org/pdf/2210.07321 -O surveys/crothers2023.pdf
wget -q https://arxiv.org/pdf/2310.15654 -O surveys/tang2024.pdf

# Detection methods
wget -q https://arxiv.org/pdf/2301.11305 -O detection/mitchell2023_detectgpt.pdf
wget -q https://arxiv.org/pdf/2310.05130 -O detection/bao2023_fast_detectgpt.pdf
wget -q https://arxiv.org/pdf/1906.04043 -O detection/gehrmann2019_gltr.pdf
wget -q https://arxiv.org/pdf/2305.17359 -O detection/yang2023_dnagpt.pdf
wget -q https://arxiv.org/pdf/2401.12070 -O detection/hans2024_binoculars.pdf
wget -q https://arxiv.org/pdf/2305.15047 -O detection/verma2024_ghostbuster.pdf
wget -q https://arxiv.org/pdf/2306.04723 -O detection/tulchinskii2024_phd.pdf
wget -q https://arxiv.org/pdf/2306.05540 -O detection/su2023_detectllm.pdf
wget -q https://arxiv.org/pdf/2307.03838 -O detection/hu2023_radar.pdf
wget -q https://arxiv.org/pdf/2304.02819 -O detection/liang2023_bias.pdf

# Watermarking
wget -q https://arxiv.org/pdf/2301.10226 -O watermarking/kirchenbauer2023.pdf
wget -q https://arxiv.org/pdf/2410.02307 -O watermarking/synthid2024.pdf
wget -q https://arxiv.org/pdf/2305.15060 -O watermarking/kuditipudi2023.pdf
wget -q https://arxiv.org/pdf/2306.09194 -O watermarking/christ2024_undetectable.pdf

# Benchmarks
wget -q https://arxiv.org/pdf/2405.07940 -O benchmarks/dugan2024_raid.pdf
wget -q https://arxiv.org/pdf/2306.15666 -O benchmarks/weber_wulff2023_testing.pdf
wget -q https://arxiv.org/pdf/2305.14902 -O benchmarks/wang2024_m4.pdf

# Adversarial
wget -q https://arxiv.org/pdf/2303.11156 -O adversarial/sadasivan2023_impossibility.pdf
wget -q https://arxiv.org/pdf/2303.13408 -O adversarial/krishna2024_dipper.pdf
wget -q https://arxiv.org/pdf/2310.05095 -O adversarial/shi2024_evasive.pdf

# Vendor
wget -q https://arxiv.org/pdf/2304.12966 -O vendor/chaka2023_turnitin_eval.pdf

echo "Download complete. Verify PDFs with: file literature/**/*.pdf"
