import subprocess

# Define the commands to be executed
commands = [
    "python src/data_prep.py",
    "python src/myTabddpm/pipeline.py --config configuration/datasets/uci_german/config.toml --train --sample",
    "python src/myTabddpm/pipeline.py --config configuration/datasets/hmeq/config.toml --train --sample",
    "python src/myTabddpm/pipeline.py --config configuration/datasets/gmsc/config.toml --train --sample",
    "python src/myTabddpm/pipeline.py --config configuration/datasets/pakdd/config.toml --train --sample",
    "python src/myTabddpm/pipeline.py --config configuration/datasets/uci_taiwan/config.toml --train --sample"
]

def run_commands():
    for command in commands:
        print(f"Running: {command}")
        process = subprocess.run(command, shell=True, capture_output=True, text=True)
        if process.returncode != 0:
            print(f"Error occurred while running: {command}")
            print(f"Error message: {process.stderr}")
        else:
            print(f"Successfully completed: {command}")
            print(f"Output: {process.stdout}")

if __name__ == "__main__":
    run_commands()