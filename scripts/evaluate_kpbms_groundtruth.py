import argparse
import json
import os

from kpsaliency.datasets import SaliencyBoxDataset
from kpsaliency.metrics.bboxes import DatasetEvaluator


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", "-d", type=str,
                        help="/path/to/PVDN/day")
    parser.add_argument("--out-dir", "-o", type=str, help="Output directory to store performance.json files. Only necessary if you want to save the scores.", default=None)
    opts = parser.parse_args()

    if opts.out_dir:
        os.makedirs(opts.out_dir, exist_ok=True)
        print("Output directory:", opts.out_dir)
    else:
        print("Not saving the scores.")

    for split in ("train", "val", "test"):
        print("Split:", split)
        base_path = os.path.join(opts.data_dir, split)
        bbox_path = os.path.join(
            base_path, "labels/kpbms_boxes"
        )
        dataset = SaliencyBoxDataset(path=base_path, load_images=False,
                                  bbox_path=bbox_path)
        evaluator = DatasetEvaluator()
        prec, rec, fsc, kp_quality, kp_quality_std, box_quality, box_quality_std, combined = evaluator.evaluate_dataset(dataset, verbose=True)

        if opts.out_dir:
            out_dict = {
                "precision": prec,
                "recall": rec,
                "fscore": fsc,
                "qk": kp_quality,
                "qk_std": kp_quality_std,
                "qb": box_quality,
                "qb_std": box_quality_std,
                "q": combined
            }

            out_path = os.path.join(opts.out_dir, f"performance_{split}.json")
            with open(out_path, "w") as f:
                json.dump(out_dict, f)
            print("Written scores to", out_path)