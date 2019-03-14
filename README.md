# Avrae NN Stuff
This repository consists of multiple helper scripts, data
preprocessors, and test stubs to assist me in training Avrae's
neural networks.

## Results
How to read these results:
- t1: The number of times the correct result was the top match for a given query.
- t2: The number of times the correct result was #2 for a given query.
- t3: The number of times the correct result was #3 for a given query.
- t10: The number of times the correct result was in #4-10 for a given query.
- f: The number of times the correct result was not in the top 10.
- t: How long it took to evaluate all queries, in seconds.

Results are roughly sorted by accuracy.
> naive_partial: t1=3326 t2=456 t3=187 t10=323 f=12635 t=5.05

> levenshtein: t1=10788 t2=1464 t3=927 t10=1150 f=2598 t=73.93

> magic1_dense Pure: t1=9621 t2=1600 t3=730 t10=1657 f=3319 t=17.03
>   - Magic1 -> Dense 128 -> Dense 501

> magic2_conv_smaller Pure: t1=10777 t2=1096 t3=513 t10=1265 f=3276 t=38.46
>
> magic2_conv_smaller Mixed: t1=11369 t2=1547 t3=699 t10=1553 f=1759 t=143.3
>   - Magic2 -> Conv1D 25, 2 -> MaxPool -> Dense 501

> magic1_embedding_conv_smaller Pure: t1=12979 t2=1172 t3=470 t10=976 f=1330 t=39.28
>
> magic1_embedding_conv_smaller Mixed: t1=12814 t2=961 t3=495 t10=1546 f=1111 t=163.57
>   - Embedding 29x16 -> Dropout 0.2 -> Conv1D 75, 3 -> GlobAvgPool -> Dropout 0.2 -> Dense 501

> magic1_embedding_conv Pure: t1=13852 t2=944 t3=335 t10=686 f=1110 t=39.88
>
> magic1_embedding_conv Mixed: t1=13821 t2=809 t3=390 t10=1125 f=782 t=149.48
>   - Embedding 29x16 -> Dropout 0.2 -> Conv1D 25, 3 -> AvgPool -> Dropout 0.2 -> Dense 501

> magic1_embedding_conv_maxpool Pure: t1=13945 t2=949 t3=346 t10=626 f=1061 t=39.90
>
> magic1_embedding_conv_maxpool Mixed: t1=13901 t2=815 t3=380 t10=1095 f=736 t=154.63
>   - Embedding 29x16 -> Dropout 0.2 -> Conv1D 25, 3 -> MaxPool -> Dropout 0.2 -> Dense 501

