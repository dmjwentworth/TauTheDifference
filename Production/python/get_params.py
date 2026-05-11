# populate the configs for different eras, extracting automatically from HiggsDNA

import yaml
from utils import get_logger
import argparse
import os


def initialise(dictionary, process, dataset):
    try:
        dictionary[process][dataset] = {
            'n_eff': 0,
            'filter_eff': 0,
            'x_sec': 0,
        }
    except KeyError:
        dictionary[process] = {}
        dictionary[process][dataset] = {
            'n_eff': 0,
            'filter_eff': 0,
            'x_sec': 0,
        }


def create_cfg_out(cfg_in):
    os.makedirs("../config", exist_ok=True)
    process_dict = {
        'Electron_DATA': {},
        'Muon_DATA': {},
        'Tau_DATA': {},
    }

    print(
        "\033[91mWARNING: DATA samples have to be added by hand\033[0m"
    )
    
    for dataset in cfg_in.keys():
        # Ignore all of Irene's samples
        if 'BBH' in dataset:
            continue
        if '2HDM' in dataset:
            continue
        if 'GluGluHto2Tau_M' in dataset:
            continue
        elif 'DYto2E' in dataset:
            initialise(process_dict, 'DYto2E', dataset)
        elif 'DYto2Mu' in dataset:
            initialise(process_dict, 'DYto2Mu', dataset)
        elif 'DYto2Tau' in dataset:
            initialise(process_dict, 'DYto2Tau', dataset)
        elif 'DYto2L' in dataset:
            initialise(process_dict, 'DYto2L', dataset)
        elif (
            'GluGluHTo2Tau_UncorrelatedDecay' in dataset
            and 'UnFiltered' not in dataset
        ):
            initialise(process_dict, 'ggH', dataset)
        elif 'ST_tW' in dataset:
            initialise(process_dict, 'ST', dataset)
        elif 'TTto' in dataset:
            initialise(process_dict, 'TTBar', dataset)
        elif dataset == 'VBFHToTauTau_UncorrelatedDecay_Filtered':
            initialise(process_dict, 'VBFH', dataset)
        elif dataset in ['WW', 'WZ', 'ZZ']:
            initialise(process_dict, 'Diboson', dataset)
        elif 'HToTauTau_UncorrelatedDecay_Filtered' in dataset:
            initialise(process_dict, 'VH', dataset)
        elif 'Wto' in dataset:
            initialise(process_dict, 'WJets', dataset)
    
    cfg_out = {
        'Params': {
            'Luminosity': cfg_in['lumi'],
        },
        'Process': process_dict
    }
    
    return cfg_out


def main(eras):  
    path_to_HiggsDNA = '/vols/cms/dmw25/HiggsDNA/scripts/ditau/config'
    for era in eras:
        print(f'\nUpdating config for era: {era}')
        # Load configuration for the desired era
        cfg_in = yaml.safe_load(open(f"{path_to_HiggsDNA}/{era}/params.yaml"))
        try:
            cfg_out = yaml.safe_load(open(f"../config/{era}.yaml"))
        except FileNotFoundError:
            cfg_out = create_cfg_out(cfg_in)
        # Load processes:
        for process in cfg_out['Process']:
            if 'DATA' not in process:
                print(f"Process: {process}")
                for ds in cfg_out['Process'][process]:
                    print(ds)
                    cfg_out['Process'][process][ds]['x_sec'] = cfg_in[ds]['xs']
                    cfg_out['Process'][process][ds]['n_eff'] = cfg_in[ds]['eff']
                    cfg_out['Process'][process][ds]['filter_eff'] = cfg_in[ds]['filter_efficiency']
                # for ggH we need to sum everything up
                if process == 'ggH':
                    neff_sum = sum([cfg_out['Process'][process][ds]['n_eff'] for ds in cfg_out['Process'][process]]) # total eff across the ggH samples
                    for ds in cfg_out['Process'][process]:
                        cfg_out['Process'][process][ds]['n_eff'] = neff_sum # set the neff to be the sum
        save_path = f"../config/{era}.yaml"
        with open(save_path, 'w') as file:
            documents = yaml.dump(cfg_out, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract parameters from HiggsDNA configs")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode")
    parser.add_argument('--era', type=str, default='all', help="Era to process (e.g., 'Run3_2024', 'Early_Run3', or 'all')")
    args = parser.parse_args()
    logger = get_logger(debug=args.debug)

    if args.era == 'all':
        eras = [
            'Run3_2022', 'Run3_2022EE', 'Run3_2023', 'Run3_2023BPix', 'Run3_2024'
        ]
    elif args.era == 'Early_Run3':
        eras = ['Run3_2022', 'Run3_2022EE', 'Run3_2023', 'Run3_2023BPix']
    elif args.era == 'Run3_2024':
        eras = ['Run3_2024']
    else:
        raise ValueError(f"Unknown era: {args.era}")

    main(eras)