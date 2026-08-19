"""Sequential Pilot runner: B -> U40 -> S40 -> U60 -> S60 -> Q"""
import os
import sys
import subprocess

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PY = sys.executable
SCRIPTS = os.path.join(ROOT, "scripts")


def run(args, name):
    print("\n" + "=" * 70)
    print(f"STARTING: {name}")
    print("=" * 70)
    cmd = [PY, "-u"] + args
    ret = subprocess.call(cmd, cwd=ROOT)
    if ret != 0:
        print(f"FAILED: {name} (exit {ret})")
        sys.exit(ret)
    print(f"DONE: {name}")


def main():
    print("Pilot order: B -> U40 -> S40 -> U60 -> S60 -> Q")
    run([os.path.join(SCRIPTS, "train_baseline.py")], "Baseline FP32 (5 epochs)")
    run([os.path.join(SCRIPTS, "prune_unstructured.py"), "--amount", "0.4"], "U40")
    run([os.path.join(SCRIPTS, "prune_structured.py"), "--amount", "0.4"], "S40")
    run([os.path.join(SCRIPTS, "prune_unstructured.py"), "--amount", "0.6"], "U60")
    run([os.path.join(SCRIPTS, "prune_structured.py"), "--amount", "0.6"], "S60")
    run([os.path.join(SCRIPTS, "quantize_int8.py")], "INT8 PTQ")
    print("\nAll pilot configs finished. See results/*.json")


if __name__ == "__main__":
    main()
