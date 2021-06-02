# ft-mod

Please read the instructions in [howtorun.pdf](https://github.com/FT-Submit/ft-mod/blob/main/howtorun.pdf) for further information on how to run the modified versions of the code, and how we prepared our data.

## Contents

The following repository contains:

- The source code for the modified version of fastText which we use in our paper.
- The preprocessed text corpora in 3 languages in `tests/data`.
- The precomputed BPE tokens for 3 languages in `tests/pretrained_embeddings`
- The file [ftscript.sh](https://github.com/FT-Submit/ft-mod/blob/main/ftscript.sh) containing the script that calculates the embeddings for different algorithmic variants.
- The file [hftokenize.py](https://github.com/FT-Submit/ft-mod/blob/main/hftokenize.py) containing the script used to tokenize the training corpora with the [HuggingFace](https://github.com/huggingface/tokenizers) tokenizer.
- The code in [synthetic.cpp](https://github.com/FT-Submit/ft-mod/blob/main/synthetic.cpp) used for the experiments on synthetic data.

## Hardware

For the purpose of our paper, the code has been run on three CPUs:

- Dual-socket Intel(R) Xeon(R) Silver 4114 CPU.
- Intel Xeon Phi 7290.
- **NEW**: AMD EPYC 7742.

## Requirements

To compile and run the code for Intel experiments, [Cmake](https://cmake.org/), [Intel C++ Compiler](https://software.intel.com/content/dam/develop/external/us/en/documents/iss-icc-download-install-cmdline-780679.pdf) and AVX-512 support are required. Please check if path to the compiler is set correcly in `CMakeLists.txt`.

**NEW:** We have additionally provided the code to run experiments for AMD discussed in the paper revision. The source files are located in src/amd. Please replace the relevant files in src with these files, uncomment line 19 in CMakeLists.txt, and comment-out line 18. Note that this solution allows the code to run under setups with no AVX-512 support (AVX and AVX2 are still required) and GCC.

## Compilation

Compile with

```
cd ft-mod
cmake .
make
```

## Data and precomputed BPE tokens

In the directory `tests`, we provide:

- `tests/data`: the preprocessed text corpora (please unzip before use!)
- `tests/pretrained_tokens`: the pretrained BPE tokens for different laguages (vocab and merges)

Note that we do not provide the large English corpus used in the experiments for paper revision. It can be created by downloading a full English Wikipedia dump from the [Wikimedia database backup dumps](https://dumps.wikimedia.org/backup-index.html) and processed with the [wikifil.pl script](http://mattmahoney.net/dc/textdata#appendixa) provided by Matt Mahoney. Note that the execution takes a few hours.

## Running

To run, please type `./fastText` with the desired arguments: please refer to the output of the helper message or to Section 4 of [howtorun.pdf](https://github.com/FT-Submit/ft-mod/blob/main/howtorun.pdf) for details).

Example run (English, skipgram code_opt with 20 threads):
```
./fasttext skipgram -input tests/data/enwiki9  -output english_skipgram_codeopt_20 -thread 20 -dim 300
```

To train all the embeddings we use in our paper, please run [ftscript.sh](https://github.com/FT-Submit/ft-mod/blob/main/ftscript.sh). The following comments apply:

- Please modify the line 10 in order to change the number of threads used in the experiments. For example, to use 1, 20 and 64 threads, please write `for thr in 1 20 64`.
- The lines 153-157 used to train our code on large corpora have been commented out due to long execution time (10 hours each with 20 threads on our Skylake machine). Please uncomment to run these experiments.
- The lines 14-25 used to train on the original fastText code must be provided with correct path to the original code. Additionally, to get measurements for the originalinal fastText code, we modify the original function void FastText::train(const Args& args) in src/fasttext.cc in the same fashion as it is modified in ft-mod.
- All the lines used to train embeddings with negative sharing (NS) are commented out. These numbers are not reported in the paper.

The full set of experiments on a single machine may take up to a few days to complete. Please refer to the "time" columns in Tables 1-2 in the paper for the execution time of each variant.

## Evaluation

All evaluation has been done with the evaluation tools prepared by the corresponding task creators and providers.

- To test on QW (sem., syn.), we use the `compute-accuracy.c` code provided [HERE](https://github.com/svn2github/word2vec) by Thomas Mikolov alongside with the `questions-words.txt` test data.
- To test on BATS, we use the [Vecto](https://pypi.org/project/vecto/) Python package:
```
    python -m vecto benchmark analogy [embedding file name] BATS_3.0 --path_out [results file name] --method 3CosAdd
```

- To test on MUSE, we use the code provided [HERE](https://github.com/facebookresearch/MUSE) by Facebook Research. Please see Section 1 in [howtorun.pdf](https://github.com/FT-Submit/ft-mod/blob/main/howtorun.pdf) on which datasets we use and where we download them from. We evaluate with the `evaluate.py` script, e.g., to test on English, we run:
```
    python evaluate.py --src_lang en --src_emb [path to embedding file]
```


- To test on Battig, we use [word-embeddings-benchmarks](https://github.com/kudkudak/word-embeddings-benchmarks), we run the full evaluation script and pick the Battig score. Note that the other scores either repeat the experiments we had already done, or operate on smaller data. We do not report them in the paper.
```
    python ./scripts/evaluate_on_all.py -p word2vec -f [embedding file name] -o [results file name]
```

The evaluation often occurs in a single-threaded fashion and may take multiple days to complete.
