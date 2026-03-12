import os
import json
import glob

# Define the specific HuggingFace cache directory for Gemma Scope SAEs
sae_dir = "/root/.cache/huggingface/hub/models--google--gemma-scope-2b-pt-att"

# Search for json configuration files specifically in the SAE directory
config_files = glob.glob(os.path.join(
    sae_dir, "**", "params.json"), recursive=True)
config_files += glob.glob(os.path.join(sae_dir, "**",
                          "config.json"), recursive=True)

# Output the content of the found configuration files
if config_files:
    for target in config_files:
        print(f"Reading: {target}\n")
        with open(target, "r", encoding="utf-8") as file:
            try:
                config_data = json.load(file)
                print(json.dumps(config_data, indent=4))
                print("-" * 50)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON in {target}")
else:
    print("No configuration files found in the specified SAE directory.")
