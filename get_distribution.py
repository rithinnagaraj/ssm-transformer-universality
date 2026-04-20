import json
from pathlib import Path

JSON_PATH = Path("all_mapped_features.json")

def main():
    with open(JSON_PATH, "r") as f:
        features = json.load(f)

    total_features = len(features)
    
    counts = {
        "Alien (< 0.25)": 0,
        "Buffer (0.25 - 0.279)": 0,
        "Grey Matter (0.28 - 0.35)": 0,
        "Aligned (> 0.35)": 0
    }

    for feat in features:
        score = feat["max_jaccard_similarity"]
        
        if score < 0.25:
            counts["Alien (< 0.25)"] += 1
        elif 0.25 <= score < 0.28:
            counts["Buffer (0.25 - 0.279)"] += 1
        elif 0.28 <= score <= 0.35:
            counts["Grey Matter (0.28 - 0.35)"] += 1
        else:
            counts["Aligned (> 0.35)"] += 1

    print("\n=== GLOBAL ALIGNMENT DISTRIBUTION ===")
    print(f"Total Active Features: {total_features}\n")
    
    for category, count in counts.items():
        percentage = (count / total_features) * 100
        print(f"{category:25}: {count:5} features ({percentage:.2f}%)")
        
    print("=====================================\n")

if __name__ == "__main__":
    main()