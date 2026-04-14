#!/bin/bash
set -euo pipefail

echo "Beginning job"
cd /vols/cms/dmw25/TauTheDifference/Training/python/
echo "Ready to train BDT"
/vols/cms/dmw25/miniforge3/bin/conda run -n TauTheDifference --no-capture-output \
	python train_BDT.py --channel=tt --config=BDTconfig_separate
echo "Finished"