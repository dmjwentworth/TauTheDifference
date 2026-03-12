#!/bin/bash

# Apply training to model:
python ../python/apply_BDTtraining.py
echo ""
echo "------------------------------------"
echo ""
# Plot evaluation metrics
python ../python/plot_optimised_binning.py
echo ""
echo "------------------------------------"
