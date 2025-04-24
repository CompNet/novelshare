from novelties_bookshare.encrypt import encrypt_tokens
from novelties_bookshare.decrypt import decrypt_tokens


def test_substitution():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C D X E".split()
    tags = "B-PER O O O B-PER I-PER".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, user_tokens)
    assert pred_tokens == ref_tokens


def test_deletion():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C E E".split()
    tags = "B-PER O O O B-PER I-PER".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, user_tokens)
    assert pred_tokens == "A B C [UNK] E E".split()


def test_addition():
    ref_tokens = "A B C D E E".split()
    user_tokens = "A B C X D E E".split()
    tags = "B-PER O O O B-PER I-PER".split()
    pred_tokens = decrypt_tokens(encrypt_tokens(ref_tokens), tags, user_tokens)
    assert pred_tokens == ref_tokens
