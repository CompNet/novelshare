#!/bin/bash 

NOVELS=("Moby_Dick" "Frankenstein" "Pride_and_Prejudice")

# params xp
for novel in ${NOVELS[@]}; do

    python xp_edition_mlm_params.py\
           --id="xp_edition_mlm_params_n=${novel}_h=2"\
           with\
           novel="$novel"\
           hash_len=2\
           window_range='[16, 32, 64, 128]'\
           device=cuda

    python xp_edition_retokenize_params.py\
           --id="xp_edition_split_params_n=${novel}_h=2"\
           with\
           novel="$novel"\
           hash_len=2\
           max_token_len_range='[8, 16, 32]'\
           max_splits_nb_range='[8, 16, 32]'

done

# main xp with chosen params
for novel in ${NOVELS[@]}; do
    python xp_edition.py\
           --id="xp_edition_n=${novel}_h=2"\
           with\
           novel="$novel"\
           hash_len=2\
           device=cuda
done

# add a few xp to plot the influence of hash length
HASH_LEN_ARRAY=(1 3 4 64) # NOTE: hash_len=2 is already done above so we skip it
for novel in ${NOVELS[@]}; do
    for hash_len in ${HASH_LEN_ARRAY[@]}; do
        python xp_edition.py\
            --id="xp_edition_n=${novel}_h=${hash_len}"\
            with\
            novel="$novel"\
            hash_len=$hash_len\
            device=cuda
    done
done
