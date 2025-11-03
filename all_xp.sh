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

    python xp_edition_split_params.py\
           --id="xp_edition_split_params_n=${novel}_h=2"\
           with\
           novel="$novel"\
           hash_len=2\
           max_token_len_range='[8, 16, 32]'\
           max_splits_nb_range='[8, 16, 32]'

    python xp_edition_propagate_order.py\
           --id="xp_edition_propagate_order_n=${novel}_h=2"\
           with\
           novel="$novel"\
           hash_len=2\
           device=cuda

done

# synthetic errors
python xp_synthetic_errors.py\
        --id="xp_synthetic_errors_h=2"\
        with\
        hash_len=2\
        min_error_ratio=0.0\
        max_error_ratio=0.15\
        errors_step=0.01\
        jobs_nb=2\
        device=cuda

python xp_synthetic_errors_ocr.py\
        --id="xp_synthetic_errors_ocr_h=2"\
        with\
        hash_len=2\
        wer_grid='[0.0, 0.05, 0.1, 0.15, 0.2]'\
        cer_grid='[0.0, 0.05, 0.1, 0.15, 0.2]'\
        jobs_nb=2\
        device=cuda
           

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
            --id="xp_edition_n=${novel}_h=2"\
            with\
            novel="$novel"\
            hash_len=$hash_len\
            device=cuda
    done
done
