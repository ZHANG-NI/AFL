# An Agentic Framework with LLMS for Solving Complex Vehicle Routing Problems

Our paper *“[An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems](https://openreview.net/forum?id=BMOgYw4EhQ)”* has been accepted to ICLR 2026.

![Framework](.\Framework.png)

#### Generate code for solving VRPs

```bash
python main.py --path "path_to_instance_floder"
```

The generated code will be stored in the `code` directory. For example, to generate code for CVRP, you can run:

```bash
x python main.py --path vrp/cvrp/50
```

#### Execute Generated Code for solving VRPs

```bash
python test.py --path "path_to_instance_folder" --problem "problem_name" --iteration <num_iterations>
```

For example, to solve the CVRP, you can run:

```bash
python test.py --path vrp/cvrp/50 --problem CVRP --iteration 1000
```

## Reference

If you find this codebase useful, please consider citing the paper:

```
@inproceedings{zhang2026afl,
  title     = {An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems},
  author    = {Zhang, Ni and Cao, Zhiguang and Zhou, Jianan and Zhang, Cong and Ong, Yew-Soon},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=BMOgYw4EhQ}
}

```
