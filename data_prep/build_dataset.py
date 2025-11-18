# build_dataset.py (adapted to call your generator scripts with supported CLI args)
"""
Robust dataset builder. Tries to import generator functions if they exist,
otherwise calls generator scripts using minimal compatible CLI arguments.

Edit the PATHS / SETTINGS below, then run:
    python build_dataset.py
"""

import os
import sys
import glob
import importlib
import subprocess
from natsort import natsorted

# -------------------------
# EDIT THESE PATHS & PARAMS
# -------------------------
TRAIN_GENERATOR_MODULE = "random_patches_train"  # filename without .py
VAL_GENERATOR_MODULE   = "fixed_sigmas_val"      # filename without .py

TRAIN_GENERATOR_SCRIPT = TRAIN_GENERATOR_MODULE + ".py"
VAL_GENERATOR_SCRIPT   = VAL_GENERATOR_MODULE + ".py"

# Source HR image folders (relative to this file if running from data_prep)
TRAIN_HR_IMAGES = '../data/DIV2K_train_HR'
VAL_HR_IMAGES   = '../data/DIV2K_valid_HR'

# Output dataset root where train/ and val/ will be created
OUT_DATASET_DIR = '../datasets'

# Train generation settings
PATCH_SIZE = 256
PATCHES_PER_IMAGE = 10

# Val generation settings
CROPS_PER_IMAGE = 3

# -------------------------
# Derived paths
# -------------------------
TRAIN_INPUT_DIR  = os.path.join(OUT_DATASET_DIR, "train", "input")
TRAIN_TARGET_DIR = os.path.join(OUT_DATASET_DIR, "train", "target")
VAL_INPUT_DIR    = os.path.join(OUT_DATASET_DIR, "val", "input")
VAL_TARGET_DIR   = os.path.join(OUT_DATASET_DIR, "val", "target")

for d in (TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, VAL_INPUT_DIR, VAL_TARGET_DIR):
    os.makedirs(d, exist_ok=True)

# -------------------------
# Helper: try to import module function
# -------------------------
def try_import_and_get(module_name, func_name):
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"[import] Could not import module '{module_name}': {e}")
        return None
    func = getattr(mod, func_name, None)
    if func is None:
        print(f"[import] Module '{module_name}' has no attribute '{func_name}'.")
    return func

# -------------------------
# Helper: run script via CLI
# -------------------------
def run_script_cli(script_path, cli_args):
    cmd = [sys.executable, script_path] + cli_args
    print("[cli] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        print("[cli] STDERR:\n", proc.stderr)
        print(f"[cli] Script {script_path} failed with exit code {proc.returncode}")
        return False
    return True

# -------------------------
# 1) TRAIN
# -------------------------
print("\n=== GENERATING TRAIN PATCHES ===")
train_files = natsorted(glob.glob(os.path.join(TRAIN_HR_IMAGES, "*.png")) + glob.glob(os.path.join(TRAIN_HR_IMAGES, "*.jpg")))
print(f"Found {len(train_files)} train HR images in {TRAIN_HR_IMAGES}")

# Preferred: if module has process_train_image, call it per image
train_func = try_import_and_get(TRAIN_GENERATOR_MODULE, "process_train_image")
if train_func:
    total = 0
    for idx, path in enumerate(train_files):
        try:
            produced = train_func(path, idx, TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, PATCH_SIZE, PATCHES_PER_IMAGE, True, 1234)
            # The function signature may vary; we try to handle return value gracefully.
            if isinstance(produced, int):
                total += produced
            else:
                total += 1
        except TypeError:
            # try calling with fewer args if signature differs
            try:
                train_func(path, idx, TRAIN_INPUT_DIR, TRAIN_TARGET_DIR, PATCH_SIZE, PATCHES_PER_IMAGE)
                total += 1
            except Exception as e:
                print(f"[train] Error processing {path}: {e}")
        except Exception as e:
            print(f"[train] Error processing {path}: {e}")
    print(f"[train] Total train patches produced (approx): {total}")
else:
    # Fallback to CLI with only args your scripts accept
    if os.path.exists(TRAIN_GENERATOR_SCRIPT):
        cli_args = [
            "--src_dir", TRAIN_HR_IMAGES,
            "--out_target_dir", TRAIN_TARGET_DIR,
            "--out_input_dir", TRAIN_INPUT_DIR,
            "--patch_size", str(PATCH_SIZE),
            "--patches_per_image", str(PATCHES_PER_IMAGE)
        ]
        success = run_script_cli(TRAIN_GENERATOR_SCRIPT, cli_args)
        if not success:
            print("[train] CLI fallback failed. Check script usage with 'python random_patches_train.py -h'")
    else:
        print(f"[train] Neither module nor script found: {TRAIN_GENERATOR_MODULE}, {TRAIN_GENERATOR_SCRIPT}")

# -------------------------
# 2) VAL
# -------------------------
print("\n=== GENERATING VAL PATCHES ===")
val_files = natsorted(glob.glob(os.path.join(VAL_HR_IMAGES, "*.png")) + glob.glob(os.path.join(VAL_HR_IMAGES, "*.jpg")))
print(f"Found {len(val_files)} val HR images in {VAL_HR_IMAGES}")

val_func = try_import_and_get(VAL_GENERATOR_MODULE, "process_val_image")
if val_func:
    total = 0
    for idx, path in enumerate(val_files):
        try:
            produced = val_func(path, idx, VAL_INPUT_DIR, VAL_TARGET_DIR, PATCH_SIZE, CROPS_PER_IMAGE, True, 1234)
            if isinstance(produced, int):
                total += produced
            else:
                total += 1
        except TypeError:
            try:
                val_func(path, idx, VAL_INPUT_DIR, VAL_TARGET_DIR, PATCH_SIZE, CROPS_PER_IMAGE)
                total += 1
            except Exception as e:
                print(f"[val] Error processing {path}: {e}")
        except Exception as e:
            print(f"[val] Error processing {path}: {e}")
    print(f"[val] Total val patches produced (approx): {total}")
else:
    if os.path.exists(VAL_GENERATOR_SCRIPT):
        cli_args = [
            "--src_dir", VAL_HR_IMAGES,
            "--out_input_dir", VAL_INPUT_DIR,
            "--out_target_dir", VAL_TARGET_DIR,
            "--patch_size", str(PATCH_SIZE),
            "--crops_per_image", str(CROPS_PER_IMAGE)
        ]
        success = run_script_cli(VAL_GENERATOR_SCRIPT, cli_args)
        if not success:
            print("[val] CLI fallback failed. Check script usage with 'python fixed_sigmas_val.py -h'")
    else:
        print(f"[val] Neither module nor script found: {VAL_GENERATOR_MODULE}, {VAL_GENERATOR_SCRIPT}")

print("\n=== BUILD COMPLETE ===")
print("Check the output folders:")
print(" -", TRAIN_INPUT_DIR)
print(" -", TRAIN_TARGET_DIR)
print(" -", VAL_INPUT_DIR)
print(" -", VAL_TARGET_DIR)
