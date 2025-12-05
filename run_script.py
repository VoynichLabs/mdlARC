from pathlib import Path
import argparse
import importlib
import utils, tinytransformer, train

importlib.reload(utils)  # pick up code changes during iteration
importlib.reload(tinytransformer)
importlib.reload(train)

args = {
    # run config
    "num_workers": 1,
    "device": "cuda",  # 'cuda' | 'mps' | 'cpu'
    # paths - must pass as Path("<path_to_dir>")
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": Path("runs/tiny.pt"),  # or None to start from scratch
    "data_path": Path(
        "assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_train/challenges.json"
    ),  # this dataset has dihedral augments only on the train set
    # hyperparameters
    "epochs": 1,
    "batch_size": 110,
    "val_batch_size": 60,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "seed": 42,
    # Visibility toggles
    "log_train_strings": False,
    "log_train_limit": 10,
    "log_inference_prompt": False,
}
cfg = argparse.Namespace(**args)

model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)

# Training only
train.train_model(
    cfg,
    model=model,
    dataloader=dataloader,
    dataset=dataset,
    device=device,
    data_path=data_path,
)

cfg.data_path = Path(
    "assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_both/challenges.json"
)  # note that for inference we switch to a dataset with dihedral augments on both train AND test to make AAIVR work
cfg.checkpoint_path = cfg.save_path

model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)

# # Full dataset evaluation
import inference

importlib.reload(inference)

EVAL_BATCH_SIZE = 1000
splits = ["train", "test"]

evaluation = inference.evaluate_model_on_dataset(
    model=model,
    dataset=dataset,
    device=device,
    batch_size=EVAL_BATCH_SIZE,
    log_prompts=args["log_inference_prompt"],
    splits=splits,
)

# # AAIVR voting on augmented test predictions
# import importlib
# import utils

importlib.reload(utils)

test_results = evaluation.get("test", {}).get("results", [])
aaivr_results = utils.run_aaivr_on_results(test_results)

print("\nAAIVR selections (pass@2) for test split:")
if not aaivr_results:
    print("  no test results available for AAIVR voting")

summary = utils.summarize_aaivr_pass_at_k(aaivr_results)
evaluated = summary.get("evaluated", 0)
hits = summary.get("hits", 0)

print("AAIVR pass@2 with targets:", f"{hits} / {evaluated} original test pairs")
x = set([])
for sel in aaivr_results:
    if sel.pass_at_k:
        x.add(sel.task_id)

print("Unique tasks: ", len(set(x)))
