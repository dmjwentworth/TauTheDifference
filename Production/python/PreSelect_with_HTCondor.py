import os
import yaml
import argparse
import subprocess


def find_python():
    result = subprocess.run(
        ['which', 'python'], capture_output=True, text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        raise RuntimeError("Python executable not found in PATH")


def split_era_yaml(era, channel):
    with open(f"../config/{era}.yaml") as f:
        era_cfg = yaml.safe_load(f)
    processes = era_cfg['Process'].keys()

    for process in processes:
        os.makedirs(f"temp_{era}_{channel}/{process}", exist_ok=True)
        process_cfg = {
            'Process': {
                process: era_cfg['Process'][process]
            }
        }
        with open(f"temp_{era}_{channel}/{process}/config.yaml", 'w') as f:
            yaml.dump(process_cfg, f)

    return processes


def submit(era, channel, process):
    print(f"\033[92mSubmitting {process} for era {era} and channel {channel}...\033[0m")
    process_dir = f"temp_{era}_{channel}/{process}"
    yaml_path = os.path.join(process_dir, "config.yaml")
    sub_file_path = os.path.join(process_dir, "submit.sub")
    request_cpus = 4
    if "DATA" in process:
        print(f"\033[93mRequesting more CPUs for {process} since it's data...\033[0m")
        request_cpus = 8

    with open(sub_file_path, 'w') as f:
        f.write(
f"""
executable = {find_python()}
getenv = True
arguments = PreSelect.py --channel {channel} --yaml {yaml_path} --debug

output = {process_dir}/logs/PreSelect.$(ClusterId).out
error = {process_dir}/logs/PreSelect.$(ClusterId).err
log = {process_dir}/logs/PreSelect.$(ClusterId).log

request_cpus = {request_cpus}
request_memory = 1024M

+MaxRuntime = 3600
queue
"""
        )
    
    subprocess.run(['condor_submit', sub_file_path], check=True)


def main(channel):
    with open(f"../config/config_{channel}.yaml") as f:
        cfg = yaml.safe_load(f)
    for era in cfg['Setup']['eras']:
        processes = split_era_yaml(era, channel)
        for process in processes:
            submit(era, channel, process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preselect HiggsDNA outputs for classifier training with HTCondor")
    parser.add_argument('--channel', type=str, help="Channel to process", required=True)
    args = parser.parse_args()

    main(args.channel)

