#run this from the huggingface tokenizer dir

import time
import textwrap
from tokenizers import ByteLevelBPETokenizer as bpe

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/ruwiki9-utf"], vocab_size=200000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-ru-200k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/ruwiki9-utf"], vocab_size=40000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-ru-40k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/ruwiki9-utf"], vocab_size=20000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-ru-20k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/dewiki9-utf"], vocab_size=200000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-de-200k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/dewiki9-utf"], vocab_size=40000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-de-40k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/dewiki9-utf"], vocab_size=20000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-de-20k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/fil9"], vocab_size=200000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-200k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/fil9"], vocab_size=40000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-40k")

tokenizer = bpe()
tokenizer.train(["/home/USERNAME/fastText/tests/data/fil9"], vocab_size=20000)
tokenizer.save("/home/USERNAME/tokenizers", "totally-tokenized-20k")