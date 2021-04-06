thr_val=(20 1)
for thr in "${thr_val[@]}"
do

echo ${thr}

echo "english skipgram normal ${thr}"
./fasttext skipgram -input tests/data/fil9  -output results/res-en/english_skipgram_normal_${thr} -thread ${thr} -dim 300
echo "english skipgram batched ${thr}"
./fasttext skipgram -mode batched -input tests/data/fil9  -output results/res-en/english_skipgram_batched_${thr} -thread ${thr} -dim 300
echo "english skipgram shared ctx ${thr}"
./fasttext skipgram -mode ns -input tests/data/fil9  -output results/res-en/english_skipgram_shared_ctx_${thr} -thread ${thr} -dim 300
echo "english cbow normal ${thr}"
./fasttext cbow -input tests/data/fil9  -output results/res-en/english_cbow_normal_${thr} -thread ${thr} -dim 300
echo "english cbow shared 11 ${thr}"
./fasttext cbow -mode ns -shared 0 -input tests/data/fil9  -output results/res-en/english_cbow_negative_sharing_10_${thr} -thread ${thr} -dim 300
echo "english cbow shared 80 ${thr}"
./fasttext cbow -mode ns -shared 80 -input tests/data/fil9  -output results/res-en/english_cbow_negative_sharing_80_${thr} -thread ${thr} -dim 300
echo "english cbow shared 160 ${thr}"
./fasttext cbow -mode ns -shared 160 -input tests/data/fil9  -output results/res-en/english_cbow_negative_sharing_160_${thr} -thread ${thr} -dim 300
echo "english cbow dh ${thr}"
./fasttext cbow -mode dh -input tests/data/fil9  -output results/res-en/english_cbow_dh_${thr} -thread ${thr} -dim 300
echo "english cbow dh shared 11 ${thr}"
./fasttext cbow -mode dh-ns -shared 0 -input tests/data/fil9  -output results/res-en/english_cbow_dh_negative_sharing_10_${thr} -thread ${thr} -dim 300
echo "english cbow dh shared 80 ${thr}"
./fasttext cbow -mode dh-ns -shared 80 -input tests/data/fil9  -output results/res-en/english_cbow_dh_negative_sharing_80_${thr} -thread ${thr} -dim 300
echo "english cbow dh shared 160 ${thr}"
./fasttext cbow -mode dh-ns -shared 160 -input tests/data/fil9  -output results/res-en/english_cbow_dh_negative_sharing_160_${thr} -thread ${thr} -dim 300
echo "english cbow dh fixed ${thr}"
./fasttext cbow -mode dhf -input tests/data/fil9  -output results/res-en/english_cbow_dh_fixed_${thr} -thread ${thr} -dim 300
echo "english cbow dh shared 11 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 0 -input tests/data/fil9  -output results/res-en/english_cbow_dh_negative_sharing_10_fixed_${thr} -thread ${thr} -dim 300
echo "english cbow dh shared 80 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 80 -input tests/data/fil9  -output results/res-en/english_cbow_dh_negative_sharing_80_fixed_${thr} -thread ${thr} -dim 300
echo "english cbow dh shared 160 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 160 -input tests/data/fil9  -output results/res-en/english_cbow_dh_negative_sharing_160_fixed_${thr} -thread ${thr} -dim 300

echo "german skipgram normal ${thr}"
./fasttext skipgram -input tests/data/dewiki9-utf  -output results/res-de/german_skipgram_normal_${thr} -thread ${thr} -dim 300
echo "german skipgram batched ${thr}"
./fasttext skipgram -mode batched -input tests/data/dewiki9-utf  -output results/res-de/german_skipgram_batched_${thr} -thread ${thr} -dim 300
echo "german skipgram shared ctx ${thr}"
./fasttext skipgram -mode ns -input tests/data/dewiki9-utf  -output results/res-de/german_skipgram_shared_ctx_${thr} -thread ${thr} -dim 300
echo "german cbow normal ${thr}"
./fasttext cbow -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_normal_${thr} -thread ${thr} -dim 300
echo "german cbow shared 11 ${thr}"
./fasttext cbow -mode ns -shared 0 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_negative_sharing_10_${thr} -thread ${thr} -dim 300
echo "german cbow shared 80 ${thr}"
./fasttext cbow -mode ns -shared 80 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_negative_sharing_80_${thr} -thread ${thr} -dim 300
echo "german cbow shared 160 ${thr}"
./fasttext cbow -mode ns -shared 160 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_negative_sharing_160_${thr} -thread ${thr} -dim 300
echo "german cbow dh ${thr}"
./fasttext cbow -mode dh -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_${thr} -thread ${thr} -dim 300
echo "german cbow dh shared 11 ${thr}"
./fasttext cbow -mode dh-ns -shared 0 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_negative_sharing_10_${thr} -thread ${thr} -dim 300
echo "german cbow dh shared 80 ${thr}"
./fasttext cbow -mode dh-ns -shared 80 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_negative_sharing_80_${thr} -thread ${thr} -dim 300
echo "german cbow dh shared 160 ${thr}"
./fasttext cbow -mode dh-ns -shared 160 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_negative_sharing_160_${thr} -thread ${thr} -dim 300
echo "german cbow dh fixed ${thr}"
./fasttext cbow -mode dhf -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_fixed_${thr} -thread ${thr} -dim 300
echo "german cbow dh shared 11 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 0 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_negative_sharing_10_fixed_${thr} -thread ${thr} -dim 300
echo "german cbow dh shared 80 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 80 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_negative_sharing_80_fixed_${thr} -thread ${thr} -dim 300
echo "german cbow dh shared 160 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 160 -input tests/data/dewiki9-utf  -output results/res-de/german_cbow_dh_negative_sharing_160_fixed_${thr} -thread ${thr} -dim 300

echo "russian skipgram normal ${thr}"
./fasttext skipgram -input tests/data/ruwiki9-utf  -output results/res-ru/russian_skipgram_normal_${thr} -thread ${thr} -dim 300
echo "russian skipgram batched ${thr}"
./fasttext skipgram -mode batched -input tests/data/ruwiki9-utf  -output results/res-ru/russian_skipgram_batched_${thr} -thread ${thr} -dim 300
echo "russian skipgram shared ctx ${thr}"
./fasttext skipgram -mode ns -input tests/data/ruwiki9-utf  -output results/res-ru/russian_skipgram_shared_ctx_${thr} -thread ${thr} -dim 300
echo "russian cbow normal ${thr}"
./fasttext cbow -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_normal_${thr} -thread ${thr} -dim 300
echo "russian cbow shared 11 ${thr}"
./fasttext cbow -mode ns -shared 0 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_negative_sharing_10_${thr} -thread ${thr} -dim 300
echo "russian cbow shared 80 ${thr}"
./fasttext cbow -mode ns -shared 80 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_negative_sharing_80_${thr} -thread ${thr} -dim 300
echo "russian cbow shared 160 ${thr}"
./fasttext cbow -mode ns -shared 160 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_negative_sharing_160_${thr} -thread ${thr} -dim 300
echo "russian cbow dh ${thr}"
./fasttext cbow -mode dh -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_${thr} -thread ${thr} -dim 300
echo "russian cbow dh shared 11 ${thr}"
./fasttext cbow -mode dh-ns -shared 0 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_negative_sharing_10_${thr} -thread ${thr} -dim 300
echo "russian cbow dh shared 80 ${thr}"
./fasttext cbow -mode dh-ns -shared 80 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_negative_sharing_80_${thr} -thread ${thr} -dim 300
echo "russian cbow dh shared 160 ${thr}"
./fasttext cbow -mode dh-ns -shared 160 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_negative_sharing_160_${thr} -thread ${thr} -dim 300
echo "russian cbow dh fixed ${thr}"
./fasttext cbow -mode dhf -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_fixed_${thr} -thread ${thr} -dim 300
echo "russian cbow dh shared 11 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 0 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_negative_sharing_10_fixed_${thr} -thread ${thr} -dim 300
echo "russian cbow dh shared 80 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 80 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_negative_sharing_80_fixed_${thr} -thread ${thr} -dim 300
echo "russian cbow dh shared 160 fixed ${thr}"
./fasttext cbow -mode dhf-ns -shared 160 -input tests/data/ruwiki9-utf  -output results/res-ru/russian_cbow_dh_negative_sharing_160_fixed_${thr} -thread ${thr} -dim 300

echo "english skipgram orig ${thr}"
../fastTextOrig/fastText/fasttext skipgram -input tests/data/fil9 -output results/res-en/english_skipgram_orig_${thr} -thread ${thr} -dim 300
echo "english cbow orig ${thr}"
../fastTextOrig/fastText/fasttext cbow -input tests/data/fil9 -output results/res-en/english_cbow_orig_${thr} -thread ${thr} -dim 300
echo "german skipgram orig ${thr}"
../fastTextOrig/fastText/fasttext skipgram -input tests/data/dewiki9-utf -output results/res-de/german_skipgram_orig_${thr} -thread ${thr} -dim 300
echo "german cbow orig ${thr}"
../fastTextOrig/fastText/fasttext cbow -input tests/data/dewiki9-utf -output results/res-de/german_cbow_orig_${thr} -thread ${thr} -dim 300
echo "russian skipgram orig ${thr}"
../fastTextOrig/fastText/fasttext skipgram -input tests/data/ruwiki9-utf -output results/res-ru/russian_skipgram_orig_${thr} -thread ${thr} -dim 300
echo "russian cbow orig ${thr}"
../fastTextOrig/fastText/fasttext cbow -input tests/data/ruwiki9-utf -output results/res-ru/russian_cbow_orig_${thr} -thread ${thr} -dim 300

echo "english skipgram nosub ${thr}"
./fasttext skipgram -input tests/data/fil9 -no-subwords -output results/res-en/english_skipgram_nosub_${thr} -thread ${thr} -dim 300
echo "german skipgram nosub ${thr}"
./fasttext skipgram -input tests/data/dewiki9-utf -no-subwords -output results/res-de/german_skipgram_nosub_${thr} -thread ${thr} -dim 300
echo "russian skipgram nosub ${thr}"
./fasttext skipgram -input tests/data/ruwiki9-utf -no-subwords -output results/res-ru/russian_skipgram_nosub_${thr} -thread ${thr} -dim 300
echo "english cbow nosub ${thr}"
./fasttext cbow -input tests/data/fil9 -no-subwords -output results/res-en/english_cbow_nosub_${thr} -thread ${thr} -dim 300
echo "german cbow nosub ${thr}"
./fasttext cbow -input tests/data/dewiki9-utf -no-subwords -output results/res-de/german_cbow_nosub_${thr} -thread ${thr} -dim 300
echo "russian cbow nosub ${thr}"
./fasttext cbow -input tests/data/ruwiki9-utf -no-subwords -output results/res-ru/russian_cbow_nosub_${thr} -thread ${thr} -dim 300


for j in 20 40 200
do
echo "english tokenized ${j}k cbow ${thr}"
./fasttext cbow -input tests/data/fil9 -output results/res-en/english_cbow_tokenized_${j}k_${thr} -token-merges ../tokenizers/totally-tokenized-${j}k-merges.txt -token-vocab ../tokenizers/totally-tokenized-${j}k-vocab.json -thread ${thr} -dim 300
echo "german tokenized ${j}k cbow ${thr}"
./fasttext cbow -input tests/data/dewiki9-utf -output results/res-de/german_cbow_tokenized_${j}k_${thr} -token-merges ../tokenizers/totally-tokenized-de-${j}k-merges.txt -token-vocab ../tokenizers/totally-tokenized-de-${j}k-vocab.json -thread ${thr} -dim 300
echo "russian tokenized ${j}k  cbow ${thr}"
./fasttext cbow -input tests/data/ruwiki9-utf -output results/res-ru/russian_cbow_tokenized_${j}k_${thr} -token-merges ../tokenizers/totally-tokenized-ru-${j}k-merges.txt -token-vocab ../tokenizers/totally-tokenized-ru-${j}k-vocab.json -thread ${thr} -dim 300
echo "english tokenized ${j}k skipgram ${thr}"
./fasttext skipgram -input tests/data/fil9 -output results/res-en/english_skipgram_tokenized_${j}k_${thr} -token-merges ../tokenizers/totally-tokenized-${j}k-merges.txt -token-vocab ../tokenizers/totally-tokenized-${j}k-vocab.json -thread ${thr} -dim 300
echo "german tokenized ${j}k skipgram ${thr}"
./fasttext skipgram -input tests/data/dewiki9-utf -output results/res-de/german_skipgram_tokenized_${j}k_${thr} -token-merges ../tokenizers/totally-tokenized-de-${j}k-merges.txt -token-vocab ../tokenizers/totally-tokenized-de-${j}k-vocab.json -thread ${thr} -dim 300
echo "russian tokenized ${j}k  skipgram ${thr}"
./fasttext skipgram -input tests/data/ruwiki9-utf -output results/res-ru/russian_skipgram_tokenized_${j}k_${thr} -token-merges ../tokenizers/totally-tokenized-ru-${j}k-merges.txt -token-vocab ../tokenizers/totally-tokenized-ru-${j}k-vocab.json -thread ${thr} -dim 300
done

done
