#!/bin/bash

# Apply training to model:
python ../python/apply_DNNtraining.py
echo ""
echo "------------------------------------"
echo ""
# Plot evaluation metrics
python ../python/plot_scores.py
echo ""
echo "------------------------------------"
echo ""
python ../python/plot_optimised_binning.py
echo ""
echo "------------------------------------"
echo ""
python ../python/plot_ROC.py
echo ""
echo "------------------------------------"
echo ""
python ../python/plot_confusion.py
echo ""
echo "------------------------------------"
echo ""