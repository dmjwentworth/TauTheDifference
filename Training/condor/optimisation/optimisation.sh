#!/bin/bash
set -euo pipefail

echo "Beginning job"
cd /vols/cms/dmw25/Lucas_TauTheDifference/Training/python/
mkdir -p hyperlogs
echo "Ready to run hyperparameter optimization"
nvidia-smi
/vols/cms/dmw25/miniforge3/bin/conda run -n TauTheDifference --no-capture-output \
	python searchBDTparams.py --channel=tt --n_trials=750 --n_jobs=6 --study_name=hyper_opt --gpu
echo "Finished"