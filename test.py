import os
import sys
import json
import time
from describe import describe_vrp
from code_generation import code_gen
import re
import subprocess
from complete_code_revise import revise_code
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def _run_one(params):
    out_file, full_vrp_path, iteration = params
    t0 = time.time()
    inicost, best_cost, err = run_generated_code(out_file, full_vrp_path, iteration)
    dt = time.time() - t0
    return (full_vrp_path, inicost, best_cost, err, dt)


def run_generated_code(out_file, full_vrp_path,iteration=100):
    try:
        # execute python {out_file} --path {full_vrp_path}
        result = subprocess.run(
            ["python", out_file, "--path", full_vrp_path, "--iteration", str(iteration)],
            capture_output=True, text=True, timeout=1500000
        )

        if result.returncode != 0:  
            error_output = result.stderr.strip()
            print("❌ Program Error:")
            print(result.stdout.strip()+error_output[:5000]) 
            return None, None, result.stdout.strip()+error_output[:5000]

        stdout = result.stdout.strip()
        print("Program Output:\n", stdout)

        # 提取 best_cost
        match1 = re.search(r"the process is successful, the best cost is (\d+(\.\d+)?)", stdout)
        match2 = re.search(r"the initial process is successful, the initial cost is (\d+(\.\d+)?)", stdout)
        if match1 and match2:
            best_cost = float(match1.group(1))
            ini_cost=float(match2.group(1))
            print("✅ best_cost:", best_cost, "✅ initial_cost:", ini_cost)
            return ini_cost, best_cost, None
        else:
            msg = "cannot find 'the process is successful, the best cost is' in output. Full stdout:\n" + stdout
            print("⚠️", msg[:5000])  
            return None, None, msg

    except Exception as e:
        print("⚠️ Error:", str(e))
        return None, None, str(e)
#python run.py --path /common/home/users/n/ni.zhang.2025/routefinder-main/vrp/evrp/small/ --problem ECVRP --iteration 1000

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process all VRP files in a folder (parallel)")
    parser.add_argument("--path", type=str, default="./vrp/cvrp/50/", help="Path to the folder containing .vrp files")
    parser.add_argument("--problem", type=str, default="CVRP")
    parser.add_argument("--iteration", type=int, default=500, help="Solver iterations passed to generated code")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()

    folder_path = args.path
    iteration = args.iteration
    specific_name = args.problem

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    out_dir = os.path.join(os.getcwd(), "code_reevo")
    out_file = os.path.join(out_dir, f"{specific_name}.py")


    vrp_files = [os.path.join(folder_path, f) for f in sorted(os.listdir(folder_path)) if (f.endswith(".vrp") or f.endswith(".atsp")or f.endswith(".sop")or f.endswith(".json")or f.endswith(".csv"))]
    total = len(vrp_files)
    if total == 0:
        print("No .vrp files found.")
        sys.exit(0)

    print(f"Found {total} .vrp files. Running with workers={args.workers}")
    start_time_describe = time.time()


    tasks = [(out_file, p, iteration) for p in vrp_files]


    num_ok = 0
    ini_cost_sum = 0.0
    best_cost_sum = 0.0
    errors = []

    results_path = os.path.join(os.getcwd(), "results.txt")
    with open(results_path, "w") as fout:
        fout.write("VRP Results Log\n")
        fout.write("==============================\n")

    # 并行执行
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, t): t[1] for t in tasks}
        for fut in tqdm(as_completed(futures), total=total, desc="Processing VRP files (parallel)"):
            full_vrp_path, inicost, best_cost, err, dt = fut.result()

            with open(results_path, "a") as fout:
                if best_cost is None:
                    errors.append((full_vrp_path, err))
                    print(f"❌ Failed: {full_vrp_path}\n{(err or '')[:5000]}")
                    fout.write(f"❌ Failed: {full_vrp_path}\nError: {(err or '')[:5000]}\n")
                    continue

                num_ok += 1
                ini_cost_sum += inicost
                best_cost_sum += best_cost

                avg_ini = ini_cost_sum / num_ok
                avg_best = best_cost_sum / num_ok
                elapsed_m = (time.time() - start_time_describe) / 60.0

                print(f"[OK] {full_vrp_path}")
                print(f"    initial={inicost:.6f}, best={best_cost:.6f}, "
                      f"avg_initial={avg_ini:.6f}, avg_best={avg_best:.6f}, "
                      f"took={dt:.2f}s, elapsed={elapsed_m:.2f}m")


                fout.write(f"{full_vrp_path}     {inicost:.6f}     {best_cost:.6f}\n")


    with open(results_path, "a") as fout:
        fout.write("\n========== Summary ==========\n")
        fout.write(f"Total files: {total}\n")
        fout.write(f"Succeeded : {num_ok}\n")
        fout.write(f"Failed    : {len(errors)}\n")
        if num_ok > 0:
            fout.write(f"Average initial cost: {ini_cost_sum/num_ok:.6f}\n")
            fout.write(f"Average best cost   : {best_cost_sum/num_ok:.6f}\n")
        if errors:
            fout.write("\nErrors (first few):\n")
            for p, e in errors[:5]:
                fout.write(f"- {p} :: {(e or '')[:300]}\n")

    print(f"\nall results are stored to {results_path}")

    
        
