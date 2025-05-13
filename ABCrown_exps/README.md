Make sure to follow the instructions for installing `alpha-beta-CROWN` [here](https://github.com/Verified-Intelligence/alpha-beta-CROWN/blob/main/complete_verifier/docs/vnn_comp.md).

It will be necessary to set up a conda environment as they have described. In our examples, neither the Gurobi or CPLEX tools were used, so you can skip setting up their licenses.

---

Before proceeding, make sure the `alpha-beta-CROWN` tool has been cloned into the `SSN_Reach/ABCrown_exps` directory and then installed according to the directions linked above.

The configuration file named `custom_cifar100.yaml` needs to be moved to the following location:
```
alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp24/custom_cifar100.yaml
```

You can run the experiments by:

1. Make sure the alpha-beta-CROWN tool is installed correctly
2. Make sure the conda environment for alpha-beta-CROWN is activated
3. `cd alpha-beta-CROWN/complete_verifier`
4. `python abcrown.py --config exp_configs/vnncomp24/custom_cifar100.yaml`

---

If you want to run the original CIFAR100 benchmark from VNN-COMP 2024, then you will need to clone [this repository](https://github.com/ChristopherBrix/vnncomp2024_benchmarks.git).

Also, if you would like to create new custom benchmarks using the code in `vnncomp2024_cifar100_benchmark`, then you will need to create a `venv/` and install the necessary dependencies as discussed in the directory's README file.
