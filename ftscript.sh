# This is a script to run the tests provided in the paper on the Skylake
# machine. Please modify accordingly.

# The runs with negative sharing (ns) have been commented out.

# To get measurements for the originalinal fastText code, we modify the
# original function void FastText::train(const Args& args)
# in src/fasttext.cc in the same fashion as it is modified in ft-mod.

for thr in 20 1 # thread setups to test
do

#~ # original fastText runs: please modify the path accordingly.
echo "english skipgram original ${thr}"
../fastText/fasttext skipgram -input tests/data/enwiki9 -output results/res-en/english_skipgram_original_${thr} -thread ${thr} -dim 300
echo "english cbow original ${thr}"
../fastText/fasttext cbow -input tests/data/enwiki9 -output results/res-en/english_cbow_original_${thr} -thread ${thr} -dim 300
echo "german skipgram original ${thr}"
../fastTextoriginal/fastText/fasttext skipgram -input tests/data/dewiki9-output results/res-de/german_skipgram_original_${thr} -thread ${thr} -dim 300
echo "german cbow original ${thr}"
../fastTextoriginal/fastText/fasttext cbow -input tests/data/dewiki9-output results/res-de/german_cbow_original_${thr} -thread ${thr} -dim 300
echo "russian skipgram original ${thr}"
../fastTextoriginal/fastText/fasttext skipgram -input tests/data/ruwiki9-output results/res-ru/russian_skipgram_original_${thr} -thread ${thr} -dim 300
echo "russian cbow original ${thr}"
../fastTextoriginal/fastText/fasttext cbow -input tests/data/ruwiki9-output results/res-ru/russian_cbow_original_${thr} -thread ${thr} -dim 300

# runs over our code_opt-based modifications.

echo "english skipgram code opt ${thr}"
./fasttext skipgram -input tests/data/enwiki9  -output results/res-en/english_skipgram_codeopt_${thr} -thread ${thr} -dim 300
echo "english skipgram batch ${thr}"
./fasttext skipgram -mode batched -input tests/data/enwiki9  -output results/res-en/english_skipgram_batch_${thr} -thread ${thr} -dim 300
#~ echo "english skipgram ns ct ${thr}"
#~ ./fasttext skipgram -mode ns -input tests/data/enwiki9  -output results/res-en/english_skipgram_ns_ct_${thr} -thread ${thr} -dim 300
echo "english cbow code opt ${thr}"
./fasttext cbow -input tests/data/enwiki9  -output results/res-en/english_cbow_codeopt_${thr} -thread ${thr} -dim 300
#~ echo "english cbow ns 11 ${thr}"
#~ ./fasttext cbow -mode ns -shared 0 -input tests/data/enwiki9  -output results/res-en/english_cbow_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "english cbow ns 80 ${thr}"
#~ ./fasttext cbow -mode ns -shared 80 -input tests/data/enwiki9  -output results/res-en/english_cbow_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "english cbow ns 160 ${thr}"
#~ ./fasttext cbow -mode ns -shared 160 -input tests/data/enwiki9  -output results/res-en/english_cbow_ns_160_${thr} -thread ${thr} -dim 300
echo "english cbow dh ${thr}"
./fasttext cbow -mode dh -input tests/data/enwiki9  -output results/res-en/english_cbow_dh_${thr} -thread ${thr} -dim 300
#~ echo "english cbow dh ns 11 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 0 -input tests/data/enwiki9  -output results/res-en/english_cbow_dh_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "english cbow dh ns 80 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 80 -input tests/data/enwiki9  -output results/res-en/english_cbow_dh_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "english cbow dh ns 160 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 160 -input tests/data/enwiki9  -output results/res-en/english_cbow_dh_ns_160_${thr} -thread ${thr} -dim 300
echo "english cbow dhf ${thr}"
./fasttext cbow -mode dhf -input tests/data/enwiki9  -output results/res-en/english_cbow_dhf_${thr} -thread ${thr} -dim 300
#~ echo "english cbow dhf ns 11 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 0 -input tests/data/enwiki9  -output results/res-en/english_cbow_dhf_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "english cbow dhf ns 80 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 80 -input tests/data/enwiki9  -output results/res-en/english_cbow_dhf_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "english cbow dhf ns 160 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 160 -input tests/data/enwiki9  -output results/res-en/english_cbow_dhf_ns_160_${thr} -thread ${thr} -dim 300

echo "german skipgram code opt ${thr}"
./fasttext skipgram -input tests/data/dewiki9 -output results/res-de/german_skipgram_codeopt_${thr} -thread ${thr} -dim 300
echo "german skipgram batch ${thr}"
./fasttext skipgram -mode batched -input tests/data/dewiki9 -output results/res-de/german_skipgram_batch_${thr} -thread ${thr} -dim 300
#~ echo "german skipgram ns ct ${thr}"
#~ ./fasttext skipgram -mode ns -input tests/data/dewiki9 -output results/res-de/german_skipgram_ns_ct_${thr} -thread ${thr} -dim 300
echo "german cbow code opt ${thr}"
./fasttext cbow -input tests/data/dewiki9 -output results/res-de/german_cbow_codeopt_${thr} -thread ${thr} -dim 300
#~ echo "german cbow ns 11 ${thr}"
#~ ./fasttext cbow -mode ns -shared 0 -input tests/data/dewiki9 -output results/res-de/german_cbow_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "german cbow ns 80 ${thr}"
#~ ./fasttext cbow -mode ns -shared 80 -input tests/data/dewiki9 -output results/res-de/german_cbow_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "german cbow ns 160 ${thr}"
#~ ./fasttext cbow -mode ns -shared 160 -input tests/data/dewiki9 -output results/res-de/german_cbow_ns_160_${thr} -thread ${thr} -dim 300
echo "german cbow dh ${thr}"
./fasttext cbow -mode dh -input tests/data/dewiki9 -output results/res-de/german_cbow_dh_${thr} -thread ${thr} -dim 300
#~ echo "german cbow dh ns 11 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 0 -input tests/data/dewiki9 -output results/res-de/german_cbow_dh_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "german cbow dh ns 80 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 80 -input tests/data/dewiki9 -output results/res-de/german_cbow_dh_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "german cbow dh ns 160 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 160 -input tests/data/dewiki9 -output results/res-de/german_cbow_dh_ns_160_${thr} -thread ${thr} -dim 300
echo "german cbow dhf ${thr}"
./fasttext cbow -mode dhf -input tests/data/dewiki9 -output results/res-de/german_cbow_dhf_${thr} -thread ${thr} -dim 300
#~ echo "german cbow dhf ns 11 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 0 -input tests/data/dewiki9 -output results/res-de/german_cbow_dhf_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "german cbow dhf ns 80 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 80 -input tests/data/dewiki9 -output results/res-de/german_cbow_dhf_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "german cbow dhf ns 160 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 160 -input tests/data/dewiki9 -output results/res-de/german_cbow_dhf_ns_160_${thr} -thread ${thr} -dim 300

echo "russian skipgram code opt ${thr}"
./fasttext skipgram -input tests/data/ruwiki9 -output results/res-ru/russian_skipgram_codeopt_${thr} -thread ${thr} -dim 300
echo "russian skipgram batch ${thr}"
./fasttext skipgram -mode batched -input tests/data/ruwiki9 -output results/res-ru/russian_skipgram_batch_${thr} -thread ${thr} -dim 300
#~ echo "russian skipgram ns ct ${thr}"
#~ ./fasttext skipgram -mode ns -input tests/data/ruwiki9 -output results/res-ru/russian_skipgram_ns_ct_${thr} -thread ${thr} -dim 300
echo "russian cbow code opt ${thr}"
./fasttext cbow -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_codeopt_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow ns 11 ${thr}"
#~ ./fasttext cbow -mode ns -shared 0 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow ns 80 ${thr}"
#~ ./fasttext cbow -mode ns -shared 80 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow ns 160 ${thr}"
#~ ./fasttext cbow -mode ns -shared 160 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_ns_160_${thr} -thread ${thr} -dim 300
echo "russian cbow dh ${thr}"
./fasttext cbow -mode dh -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dh_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow dh ns 11 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 0 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dh_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow dh ns 80 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 80 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dh_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow dh ns 160 ${thr}"
#~ ./fasttext cbow -mode dh-ns -shared 160 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dh_ns_160_${thr} -thread ${thr} -dim 300
echo "russian cbow dhf ${thr}"
./fasttext cbow -mode dhf -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dhf_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow dhf ns 11 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 0 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dhf_ns_10_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow dhf ns 80 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 80 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dhf_ns_80_${thr} -thread ${thr} -dim 300
#~ echo "russian cbow dhf ns 160 ${thr}"
#~ ./fasttext cbow -mode dhf-ns -shared 160 -input tests/data/ruwiki9 -output results/res-ru/russian_cbow_dhf_ns_160_${thr} -thread ${thr} -dim 300

# runs over no_subword
echo "english skipgram no subword ${thr}"
./fasttext skipgram -input tests/data/enwiki9 -no-subwords -output results/res-en/english_skipgram_no_subword_${thr} -thread ${thr} -dim 300
echo "german skipgram no subword ${thr}"
./fasttext skipgram -input tests/data/dewiki9-no-subwords -output results/res-de/german_skipgram_no_subword_${thr} -thread ${thr} -dim 300
echo "russian skipgram no subword ${thr}"
./fasttext skipgram -input tests/data/ruwiki9-no-subwords -output results/res-ru/russian_skipgram_no_subword_${thr} -thread ${thr} -dim 300
echo "english cbow no subword ${thr}"
./fasttext cbow -input tests/data/enwiki9 -no-subwords -output results/res-en/english_cbow_no_subword_${thr} -thread ${thr} -dim 300
echo "german cbow no subword ${thr}"
./fasttext cbow -input tests/data/dewiki9-no-subwords -output results/res-de/german_cbow_no_subword_${thr} -thread ${thr} -dim 300
echo "russian cbow no subword ${thr}"
./fasttext cbow -input tests/data/ruwiki9-no-subwords -output results/res-ru/russian_cbow_no_subword_${thr} -thread ${thr} -dim 300

# runs over BPE
for j in 20 40 200
do
echo "english bpe ${j}k cbow ${thr}"
./fasttext cbow -input tests/data/enwiki9 -output results/res-en/english_cbow_bpe_${j}k_${thr} -token-merges ./tests/pretrained_tokens/totally-tokenized-en-${j}k-merges.txt -token-vocab ./tests/pretrained_tokens/totally-tokenized-en-${j}k-vocab.json -thread ${thr} -dim 300
echo "german bpe ${j}k cbow ${thr}"
./fasttext cbow -input tests/data/dewiki9-output results/res-de/german_cbow_bpe_${j}k_${thr} -token-merges ./tests/pretrained_tokens/totally-tokenized-de-${j}k-merges.txt -token-vocab ./tests/pretrained_tokens/totally-tokenized-de-${j}k-vocab.json -thread ${thr} -dim 300
echo "russian bpe ${j}k  cbow ${thr}"
./fasttext cbow -input tests/data/ruwiki9-output results/res-ru/russian_cbow_bpe_${j}k_${thr} -token-merges ./tests/pretrained_tokens/totally-tokenized-ru-${j}k-merges.txt -token-vocab ./tests/pretrained_tokens/totally-tokenized-ru-${j}k-vocab.json -thread ${thr} -dim 300
echo "english bpe ${j}k skipgram ${thr}"
./fasttext skipgram -input tests/data/enwiki9 -output results/res-en/english_skipgram_bpe_${j}k_${thr} -token-merges ./tests/pretrained_tokens/totally-tokenized-en-${j}k-merges.txt -token-vocab ./tests/pretrained_tokens/totally-tokenized-en-${j}k-vocab.json -thread ${thr} -dim 300
echo "german bpe ${j}k skipgram ${thr}"
./fasttext skipgram -input tests/data/dewiki9-output results/res-de/german_skipgram_bpe_${j}k_${thr} -token-merges ./tests/pretrained_tokens/totally-tokenized-de-${j}k-merges.txt -token-vocab ./tests/pretrained_tokens/totally-tokenized-de-${j}k-vocab.json -thread ${thr} -dim 300
echo "russian bpe ${j}k  skipgram ${thr}"
./fasttext skipgram -input tests/data/ruwiki9-output results/res-ru/russian_skipgram_bpe_${j}k_${thr} -token-merges ./tests/pretrained_tokens/totally-tokenized-ru-${j}k-merges.txt -token-vocab ./tests/pretrained_tokens/totally-tokenized-ru-${j}k-vocab.json -thread ${thr} -dim 300
done

# English experiments over large data
#~ echo "english skipgram code opt large ${thr}"
#~ ./fasttext skipgram -input tests/data/enwiki_large  -output results/res-en/english_skipgram_codeopt_large_${thr} -thread ${thr} -dim 300
#~ echo "english cbow code opt large ${thr}"
#~ ./fasttext cbow -input tests/data/enwiki_large  -output results/res-en/english_cbow_codeopt_large_${thr} -thread ${thr} -dim 300

done
