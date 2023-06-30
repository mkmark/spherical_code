# spherical code

## usage

### example

For single solver, compile:

```sh
g++ -std=c++11 -g src/solver/SolverMinPotentialNaive.cpp -o src/solver/SolverMinPotentialNaive.cpp.o3.out -O3 -I .
```

With n=100, random seed as 42, result points output path (optional) as "output/path/".

```sh
src/solver/SolverMinPotentialNaive.cpp.o3.out 100 42 output/path/
```

The random seed here is mandatory for stable reproductivity.

## project structure

- .vscode: vscode settings
- src: all the current programs
  - batch_solver: a celery implementation of distributed computing framework
  - solver: the actual algorithm
  - validator: for result analysis and validation
- out: results (raw results are too large to put into a git repo thus gitignored)
  - [ver]/[target]/txtdump/[n].txt: the best configuration of the n points found for the target
- include: 3rd party libraries
- release: version stamped programs for batch solver to use, see below for detail

## releases

Note that unless stated otherwise, results derived by different releases are different even with the same random seed. For this reason, releases are preserved for reproducibility. Using git version control alone is not enough as there are requirements to compare between versions, thus the release folder is created for this purpose.

### v1.0.0

- initial version with SolverMinPotentialNative

### v1.0.1

- use adaptive alpha
  - this avoids large step count near n=80 in previous version

### v1.0.2

- use simulated crystallization
  - this significantly increases the probability of finding the global best
