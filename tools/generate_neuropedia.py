import json
import argparse
import os

def generate_neuronpedia_links(json_path, top_k=10):
    if not os.path.exists(json_path):
        print(f"Error: File '{json_path}' not found.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data.get("ranked_features", [])
    if not features:
        print("No features found in the JSON file.")
        return

    syc_promoters = [f for f in features if f["type"] == "sycophancy_promoter"]
    fac_promoters = [f for f in features if f["type"] == "factuality_promoter"]

    def print_links(feature_list, category_name):
        print(f"\n### Top {min(top_k, len(feature_list))} {category_name}")
        print("-" * 60)
        
        for i, feat in enumerate(feature_list[:top_k]):
            submod = feat["submodule"]
            index = feat["index"]
            effect = feat["effect"]
            
            mod_parts = submod.split('_')
            mod_type = "res" if mod_parts[0] == "resid" else "mlp"
            layer = mod_parts[1]
            
            url = f"https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-{mod_type}-16k/{index}"
            
            print(f"{i+1}. **[{submod} / Feature {index}]({url})** | Effect: `{effect:.4f}`")

    print(f"Loaded {len(features)} total features from {json_path}")
    print_links(syc_promoters, "Sycophancy Promoters (Induces Sycophancy)")
    print_links(fac_promoters, "Factuality Promoters (Induces Honesty/Factuality)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Neuronpedia links from SFC JSON output.")
    parser.add_argument("--file", type=str, required=True, help="Path to the extracted JSON file.")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top features to display per category.")
    
    args = parser.parse_args()
    generate_neuronpedia_links(args.file, args.top_k)