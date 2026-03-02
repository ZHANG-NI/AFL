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

def run_generated_code(out_file, full_vrp_path,iteration=100):
    try:
        result = subprocess.run(
            ["python", out_file, "--path", full_vrp_path, "--iteration", str(iteration)],
            capture_output=True, text=True, timeout=50
        )

        if result.returncode != 0:
            error_output = result.stderr.strip()
            print("❌ Program Error:")
            print(result.stdout.strip()[:5000]+error_output[:5000])  
            return None, None, result.stdout.strip()[:5000]+error_output[:5000]

        # 正常输出
        stdout = result.stdout.strip()
        print("Program Output:\n", stdout)

        # 提取 best_cost
        match1 = re.search(r"the process is successful, the best cost is (\d+(\.\d+)?)", stdout)
        match2 = re.search(r"the initial process is successful, the initial cost is (\d+(\.\d+)?)", stdout)
        if match1 and match2:
            best_cost = float(match1.group(1))
            ini_cost=float(match2.group(1))
            print("✅ best_cost:", best_cost, "✅initial_cost:", ini_cost)
            return ini_cost, best_cost, None
        else:
            msg = "cannot find 'the process is successful, the best cost is' in output. Full stdout:\n" + stdout
            print("⚠️", msg[:5000])  
            return None, None, msg

    except Exception as e:
        print("⚠️ Error:", str(e))
        return None, None, str(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process all VRP files in a folder")
    parser.add_argument("--path", type=str, default="./vrp/cvrp/50/", help="Path to the folder containing .vrp files")
    parser.add_argument("--iteration", type=int, default=500, help="Path to the folder containing .vrp files")
    args = parser.parse_args()
    folder_path = args.path
    iteration= args.iteration
    save_file = "vrp_meta.json" 

    # ========= Step 1: read local file =========
    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        vrp_list = saved_data.get("vrp_list", [])
        problem_desc_list = saved_data.get("problem_desc_list", {})
        constraints_list = saved_data.get("constraints_list", {})
        input_def_list = saved_data.get("input_def_list", {})
        output_def_list = saved_data.get("output_def_list", {})
        objective_list = saved_data.get("objective_list", {})
    else:
        vrp_list = []
        problem_desc_list, constraints_list = {}, {}
        input_def_list, output_def_list, objective_list = {}, {}, {}

    flag=0
    iter=0
    best_cost_all=0
    ini_cost_all=0
    # ========= Step 2: go through all .vrp file =========
    num=0
    for vrp_path in tqdm(sorted(os.listdir(folder_path)), desc="Processing VRP files"):
        num+=1
        if not (vrp_path.endswith(".vrp") or vrp_path.endswith(".sop")or vrp_path.endswith(".json")or vrp_path.endswith(".csv")):
            continue
        full_vrp_path = os.path.join(folder_path, vrp_path)
        print(full_vrp_path)

        print("----------------------Describe_Vrp-----------------------")
        vrp_list1 = vrp_list.copy()
        if flag==0:
            start_time_describe = time.time()
            _, problem_desc, constraints, specific_name, input_def, output_def, objective, vrp_list = describe_vrp( full_vrp_path, vrp_list=vrp_list)
            flag=1
            end_time_describe = time.time()
        print(f"describe_vrp took: {end_time_describe - start_time_describe:.2f}s")
        print("problem_description:", problem_desc or "❌ Not found")
        print("constraints:", constraints or "❌ Not found")
        print("specific_name:", specific_name or "❌ Not found")
        print("input:", input_def or "None")
        
        if specific_name not in vrp_list1:
            problem_desc_list[specific_name] = problem_desc
            constraints_list[specific_name] = constraints
            input_def_list[specific_name] = input_def
            output_def_list[specific_name] = output_def
            objective_list[specific_name] = objective

        # ========= Step 3: code_gen =========
        if specific_name not in vrp_list1:
            start_time_generate = time.time()
            print("----------------------Code Generation-----------------------")
            code_gen(full_vrp_path, problem_desc, constraints, specific_name, input_def, output_def, objective)
            end_time_generate = time.time()
            print(f"generate_code took: {end_time_generate - start_time_generate:.2f}s, "
              f"now_took: {end_time_generate - start_time_describe:.2f}s")
          
        out_dir = os.path.join(os.getcwd(), "code_reevo")
        out_file = os.path.join(out_dir, f"{specific_name}.py")
        if os.path.exists(out_file):
            with open(out_file, "r", encoding="utf-8") as f:
                code = f.read()
        else:
            print(f"❌ {out_file} not exist")
            

        # ========= Step 4: store data =========
        if specific_name not in vrp_list1:
            to_save = {
                "vrp_list": vrp_list,
                "problem_desc_list": problem_desc_list,
                "constraints_list": constraints_list,
                "input_def_list": input_def_list,
                "output_def_list": output_def_list,
                "objective_list": objective_list,
            }
            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(to_save, f, indent=2, ensure_ascii=False)


        # ========= Step 5: run code =========
        print("----------------------Run Code-----------------------")
        start_time_run = time.time()
        while True:
            inicost, best_cost,stout = run_generated_code(out_file, full_vrp_path,iteration)
            if best_cost is not None:
                break
            iter1=revise_code(full_vrp_path, problem_desc, constraints, specific_name, input_def, output_def, objective, code, stout)
            iter+=iter1
        end_time_run = time.time()
        best_cost_all+=best_cost
        ini_cost_all+=inicost
        print(ini_cost_all/num, best_cost_all/num, best_cost, iter, f"run_code took: {end_time_run - start_time_run:.2f}s, ", f"now_took: {end_time_run - start_time_describe:.2f}s")

    
        
