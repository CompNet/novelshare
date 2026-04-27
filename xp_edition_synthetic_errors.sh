#!/bin/bash 

# synthetic errors
python xp_synthetic_errors.py\
        --id="xp_synthetic_errors_h=2"\
        with\
        hash_len=2\
        min_error_ratio=0.0\
        max_error_ratio=0.2\
        error_ratio_step=0.02\
        jobs_nb=2\
        device=cuda

# explanation for WER/CER values: we refer to Figure 3 of the
# supplementary materials of the scrambledtext paper (Bourne
# 2025). Our WER/CER grid visually follows the typical values
# presented for the BLN600, CA and SMH datasets.
python xp_synthetic_errors_ocr.py\
        --id="xp_synthetic_errors_ocr_h=2"\
        with\
        hash_len=2\
        wer_grid='[0.0, 0.1,   0.2,  0.3,   0.4,  0.5,  0.6]'\
        cer_grid='[0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.175]'\
        jobs_nb=2\
        device=cuda
