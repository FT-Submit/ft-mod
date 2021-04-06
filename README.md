# ft-mod

To compile the code, Cmake, Intel ICPC compiler and AVX-512 support are required. Please check if path to ICPC compiler is set correcly in CMakeLists.txt.

Compile with
```
cd ft-mod
cmake .
make
```

Please read the instructions in `howtorun.pdf` on how to run the modified versions of the code.

The file `hftokenize.py` contains the script used to tokenize the training corpora with the [HuggingFace](https://github.com/huggingface/tokenizers) tokenizer.

The file `ftscript.sh` contains the script that calculates the embeddings for different algorithmic variants.

All evaluation has been done with the evaluation tools prepared by the corresponding task creators and providers.

The code used for the experiments on synthetic datals can be found in the file `synthetic.cpp`.
