import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Analyze routing errors for a specific fine class.")
    parser.add_argument("--fine_class", type=str, default="Breast cancer cells",
                        help="The fine class to analyze, e.g., 'Breast cancer cells', 'Myeloid cells'")
    parser.add_argument("--results_dir", type=str, default="/hpc/group/jilab/rz179/MorphPT_MOE/results/moe_e2e")
    args = parser.parse_args()

    results_dir = args.results_dir
    fine_class = args.fine_class

    try:
        df = pd.read_parquet(f"{results_dir}/predictions.parquet")
    except FileNotFoundError:
        print(f"Could not find predictions.parquet in {results_dir}.")
        print("Please run eval_moe_e2e.py to generate evaluation outputs first.")
        return

    df_class = df[df["label"] == fine_class].copy()

    if len(df_class) == 0:
        print(f"No cells found for fine class '{fine_class}'.")
        return

    print(f"\n=============================================")
    print(f"Analysis for: {fine_class}")
    print(f"Total cells:  {len(df_class)}")
    print(f"=============================================")
    
    df_class["router_correct"] = df_class["router_pred"] == df_class["coarse_label"]
    overall_acc = df_class["router_correct"].mean()
    print(f"Overall Coarse Routing Accuracy: {overall_acc:.4f}\n")

    print("--- Routing Accuracy by Tissue ---")
    if "tissue" in df_class.columns:
        tissue_stats = df_class.groupby("tissue").agg(
            total=("router_correct", "count"),
            correct=("router_correct", "sum")
        )
        tissue_stats["accuracy"] = tissue_stats["correct"] / tissue_stats["total"]
        tissue_stats = tissue_stats.sort_values("accuracy")
        print(tissue_stats.to_string())
    else:
        print("No 'tissue' column found in predictions metadata.")

    print("\n--- Gate Distribution Analysis (2.5x vs 10x) ---")
    if "gate_2_5x" in df_class.columns and "gate_10x" in df_class.columns:
        correct_mask = df_class["router_correct"]
        
        print(f"{'Routing Status':<20} | {'Gate 2.5x (Avg)':<18} | {'Gate 10x (Avg)':<18} | {'N'}")
        print("-" * 70)
        
        g1_mean_all = df_class["gate_2_5x"].mean()
        g2_mean_all = df_class["gate_10x"].mean()
        print(f"{'All Cells':<20} | {g1_mean_all:<18.4f} | {g2_mean_all:<18.4f} | {len(df_class)}")
        
        if correct_mask.sum() > 0:
            g1_mean_c = df_class.loc[correct_mask, "gate_2_5x"].mean()
            g2_mean_c = df_class.loc[correct_mask, "gate_10x"].mean()
            print(f"{'Correct (Router)':<20} | {g1_mean_c:<18.4f} | {g2_mean_c:<18.4f} | {correct_mask.sum()}")
            
        if (~correct_mask).sum() > 0:
            g1_mean_w = df_class.loc[~correct_mask, "gate_2_5x"].mean()
            g2_mean_w = df_class.loc[~correct_mask, "gate_10x"].mean()
            print(f"{'Wrong (Router)':<20} | {g1_mean_w:<18.4f} | {g2_mean_w:<18.4f} | {(~correct_mask).sum()}")
    else:
        print("Gate values not found in predictions.parquet.")
        print("Hint: eval_moe_e2e.py needs to safely save router_gates.")

    print("\n--- Misclassification Analysis ---")
    wrong_df = df_class[~df_class["router_correct"]]
    if len(wrong_df) > 0:
        misclass_counts = wrong_df["router_pred"].value_counts()
        misclass_pct = misclass_counts / len(wrong_df) * 100
        print(f"{'Predicted Coarse Class':<25} | {'Count':<8} | {'Percentage'}")
        print("-" * 55)
        for pred_class, count in misclass_counts.items():
            pct = misclass_pct[pred_class]
            print(f"{pred_class:<25} | {count:<8} | {pct:.2f}%")
    else:
        print("No misclassifications found!")

if __name__ == "__main__":
    main()
