# Tau The Difference

Separate Higgs to Tau Tau decays from Genuine and Fake Backgrounds – now with separate ggH and qqH (= VBF + VH) categories. 

Channels currently supported:
- Fully hadronic ($\tau_h\tau_h$)

Models currently supported:
- XGBClassifier 

Workflow for training the BDT:
    - ``cd Production/python``
    - ``python PreSelect.py <some argparse stuff>``
    - ``python Process.py <some argparse stuff>``
    - ``python ShuffleMerge.py <some argparse stuff>``
    - ``cd ../../Training/python``
    - ``python trainBDT.py <some argparse stuff>``
    - ``cd ../../Evaluation/python``
    - ``python apply_BDTtraining.py <some argparse stuff>``
    - ``python plot_optimised_binning.py <some argparse stuff>``

Visit the repo upstream of this one <URL> for a more informative README.
