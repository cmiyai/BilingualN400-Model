# `run_simulations.py` Execution Flow

## Overview

Running `python run_simulations.py` does four main things:

1. Sets up the environment and loads the lexicon
2. Builds stimulus sets for each experiment condition
3. Runs a batch of predictive-coding simulations
4. Analyzes results, makes plots, and exports CSVs

## 1. Initial Setup

When the script starts, it:

- imports helper modules
- changes the working directory to the script folder
- sets fixed random seeds for reproducibility

This happens at the top of `run_simulations.py`.

## 2. Helper Function Definition

The script defines `run_simulation(**kwargs)`.

This function:

- creates a `Simulation` object
- extracts only the summary outputs needed later
- deletes the full simulation object to save memory
- returns a lightweight dictionary containing:
  - `simulation_data`
  - `sim_filename`
  - `sim_input_bottomup`
  - `sim_input_topdown`

## 3. Lexicon Construction

The script creates:

- `lexicon = Lexicon()`

Inside `Lexicon()`:

- the lexical CSV is loaded
- unique words are extracted
- orthographic one-hot representations are created
- lexical identity matrix is built
- word frequency values are loaded and scaled
- semantic feature matrix is loaded
- orthographic neighborhood size is computed

Outputs created here include:

- `self.words`
- `self.orthmatrix`
- `self.lexicalmatrix`
- `self.frequency`
- `self.semfeatmatrix`
- `self.ONsize`

## 4. Stimulus Loading

The script loads precomputed stimulus index arrays:

- `standard_stims_word_idx.npy`
- `fully_unrelated_word_idx_1.npy`
- `sem_related_word_idx_nonneighbor.npy`

These are used to index into `wordlist` and produce:

- `standard_stims`
- `unrelated_stims_1`
- `semrelated_stims`

Then it verifies that:

- semantically related pairs share features
- unrelated pairs share zero semantic features

## 5. Simulation Families

The script runs several experiment types.

### Standard simulation

Runs bottom-up presentation of the standard stimuli.

### Semantic priming

Runs:

- semantically unrelated primes
- semantically related primes

### Repetition priming

Runs:

- unrepeated condition
- repeated condition

### Cloze / predictability simulations

For several cloze levels, it runs:

1. top-down preactivation
2. bottom-up target presentation after that preactivation

### Lexical prediction violation

Runs unexpected words after low-constraint and high-constraint contexts.

### Semantic prediction overlap

Compares semantically related vs unrelated unexpected continuations under strong constraint.

## 6. What Happens Inside Each `Simulation`

Creating `Simulation(**kwargs)` does this:

### A. Load cached data if available

If `./data/<sim_filename>.pbz2` already exists, the simulation is loaded from disk.

### B. Otherwise initialize from scratch

If no cached file exists:

- parameters are copied from `kwargs`
- if `prevSim` is provided, state is loaded from the previous simulation
- otherwise weights and initial state are built from scratch

### C. Convert inputs into model representations

Depending on mode:

- bottom-up words are converted into orthographic input
- top-down words are converted into contextual probability input

### D. Run iterative updates

The simulation loops through:

- blank iterations
- clamped input iterations

At each iteration it updates:

- orthographic state
- lexical state
- semantic state
- contextual state
- reconstructions
- prediction errors
- top-down biases

### E. Save summary information

After each iteration, the script stores summary metrics such as:

- max lexical activation
- total lexical-semantic prediction error
- total lexical-semantic bias
- identity of the most active lexical unit

### F. Save to disk

If `sim_filename` is set, the full simulation state is saved as a compressed `.pbz2` file.

## 7. Postprocessing

After all simulations are complete, the script computes:

- threshold-crossing accuracy
- threshold-crossing iteration
- whether the correct lexical item crossed threshold
- which word crossed threshold

It does this across multiple thresholds.

## 8. Plotting

The script generates plots for:

- mean threshold crossing iteration vs threshold
- total lexical-semantic prediction error over time
- total lexical-semantic bias over time
- most active lexical state over time

Plots are saved into `./plots/` as:

- `.png`
- `.svg`

## 9. CSV Export

The script creates trial-level CSV files in `./simulation_csv_files/`.

These CSVs include values like:

- lexical-semantic error
- threshold crossing iteration
- orthographic neighborhood size
- frequency
- number of semantic features
- whether threshold was crossed
- whether the correct word crossed
- which word crossed
- condition labels and codes
- input word

## 10. Current Debugging Behavior

Right now the script does not run straight through because there are active debugging breakpoints such as:

- `pdb.set_trace()` in `run_simulations.py`
- `pdb.set_trace()` in `PredictiveCoding_Model.py`

So execution will pause unless you manually continue in `pdb`.

There are also bilingual/shape-related changes in the current code that can produce matrix dimension mismatches if:

- lexical vocabulary size
- semantic matrix columns
- concept/context dimensions

are not aligned.

## Mental Model

A short way to think about the script is:

1. Build the lexicon
2. Build the stimulus lists
3. Run many predictive-coding experiments
4. Save compressed simulation outputs
5. Compute threshold-based behavioral summaries
6. Plot the dynamics
7. Export analysis CSVs

