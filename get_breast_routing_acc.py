import pandas as pd
from pathlib import Path

# Paths based on the script we saw
ROUTER_SHARDS_DIR = Path("/hpc/group/jilab/rz179/MorphPT_MOE/prepared/splits_v3_seed1337/router_shards")
TEST_SHARDS_DIR = Path("/hpc/group/jilab/rz179/MorphPT_MOE/prepared/splits_v3_seed1337/test_shards")

SLIDES = [
    "Xenium_V1_FFPE_Human_Breast_IDC_With_Addon",
    "Xenium_V1_FFPE_Human_Breast_ILC",
    "Xenium_V1_FFPE_Human_Breast_IDC",
    "Xenium_V1_FFPE_Human_Breast_ILC_With_Addon",
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_1",
    "Xenium_V1_FFPE_Human_Breast_IDC_Big_2",
]

# For routing accuracy according to eval_moe_e2e.py:
# We need to evaluate routing on `core_test` using router_ckpt "experiments/router_best/best.pt"
# However, the user simply asks: 
# "对那 6 个 breast slides，把 router_shards 和 test_shards 分别算一遍 breast 的 routing acc（按 tissue 和按 patch_id）"
# "for those 6 breast slides, calculate the router_shards and test_shards breast routing acc separately (by tissue and by patch_id)"
#
# Wait, routing accuracy requires the router model, OR does it mean we should run the router on these shards?
# Or do these shards already contain router predictions?
