#!/bin/bash

python xp_edition_ner_novelties.py\
       --id="xp_edition_ner_novelties"\
       with\
       hash_len=2\
       device=cuda

NER_CORPORA=('conll2003' 'wnut2017')

for ner_corpus in ${NER_CORPORA[@]}; do

    python xp_synthetic_errors.py\
            --id="xp_synthetic_errors_c=${ner_corpus}_h=2"\
            with\
            corpus_id="${ner_corpus}"\
            hash_len=2\
            min_error_ratio=0.0\
            max_error_ratio=0.2\
            error_ratio_step=0.02\
            jobs_nb=2\
            device=cuda

    python xp_synthetic_errors_ocr.py\
            --id="xp_synthetic_errors_ocr_c=${ner_corpus}_h=2"\
            with\
            corpus_id="${ner_corpus}"\
            hash_len=2\
            wer_grid='[0.0, 0.1,   0.2,  0.3,   0.4,  0.5,  0.6]'\
            cer_grid='[0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.175]'\
            jobs_nb=2\
            device=cuda

done
