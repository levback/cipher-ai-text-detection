#!/bin/bash
cd /Users/levent.ozparlak/Projects/AITextDetection
exec .venv-1/bin/jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=7200 \
  --ExecutePreprocessor.kernel_name=cipher-venv \
  notebooks/full_model_eval.ipynb
