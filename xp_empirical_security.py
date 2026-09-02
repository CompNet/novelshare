from typing import Literal
import hashlib, re
import pathlib as pl
from collections import defaultdict
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.commands import print_config
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds
import torch
from nltk.corpus import words
from transformers import (
    LogitsProcessor,
    PreTrainedTokenizer,
    LogitsProcessorList,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from novelshare.conll import load_conll2002_bio
from novelshare.hash import hash_token, hash_tokens
from novelshare.align import (
    align_tokens,
    make_plugin_mlm,
    make_plugin_propagate,
    make_plugin_retokenize,
    make_plugin_case,
)
from novelshare.experiments.data import iter_book_chapters, EDITION_SETS
from novelshare.experiments.metrics import errors_percent


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

    def _build_tree_string(self, level: int = 0, token_id: int | str = "ROOT") -> str:
        indent = "    " * level
        marker = "└── " if level > 0 else ""

        end_tag = " [END]" if self.is_end_of_word else ""

        lines = [f"{indent}{marker}Token: {token_id}{end_tag}"]

        for child_id in sorted(self.children.keys()):
            child_node = self.children[child_id]
            lines.append(child_node._build_tree_string(level + 1, child_id))

        return "\n".join(lines)

    def __str__(self) -> str:
        return self._build_tree_string()

    def __repr__(self) -> str:
        return f"TrieNode(children={len(self.children)}, is_end={self.is_end_of_word})"


class HashMaskLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        allowed_words: list[set[str]],
        tokenizer: PreTrainedTokenizer,
        prompt_len: int,
    ):
        """
        :param allowed_words: at each position, the set of allowed words.
        :param tokenizer: HF tokenizer to convert words to subword tokens.
        :param prompt: initial prompt.
        """
        self.tokenizer = tokenizer
        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must have a defined eos_token_id")
        self.word_tries = [TrieNode() for _ in allowed_words]
        self.words_nb = len(self.word_tries)
        self.prompt_len = prompt_len

        # we perform batched tokenization for performance reasons
        flat_words = []
        words_to_trie_idx = []
        for i, word_set in enumerate(allowed_words):
            for word in word_set:
                # prefix = " " if i > 0 and not word.startswith(" ") else ""
                prefix = " "
                flat_words.append(prefix + word)
                words_to_trie_idx.append(i)
        batched_tokenized = self.tokenizer(flat_words, add_special_tokens=False)
        batched_token_ids = batched_tokenized["input_ids"]

        for trie_idx, token_ids in zip(words_to_trie_idx, batched_token_ids):
            node = self.word_tries[trie_idx]

            # handle empty tokenizations
            if not token_ids:
                node.is_end_of_word = True
                continue

            for token_id in token_ids:
                if token_id not in node.children:
                    node.children[token_id] = TrieNode()
                node = node.children[token_id]
            node.is_end_of_word = True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.Tensor:
        """
        :param input_ids: (b, s)
        :param scores: (b, v)
        """
        # edge case: user forces 0 words, so we output EOS immediately
        if len(self.word_tries) == 0:
            mask = torch.full_like(scores, -float("inf"))
            mask[:, self.tokenizer.eos_token_id] = 0
            return scores + mask

        processed_scores = scores.clone()
        generated_ids = input_ids[:, self.prompt_len :]

        for batch_idx in range(input_ids.shape[0]):
            seq_generated = generated_ids[batch_idx].tolist()

            # a state represent a generation possibility
            # (current_word_idx, current_trie_node | None)
            #                                         ^
            #                                terminal DONE state
            current_states = {(0, self.word_tries[0])}

            for token in seq_generated:
                next_states = set()

                for word_idx, node in current_states:
                    # TODO: should not happen?
                    # if node is None:
                    #     if token == self.tokenizer.eos_token_id:
                    #         next_states.add((word_idx, None))
                    #     continue

                    if token in node.children:
                        next_node: TrieNode = node.children[token]

                        # first path: continue the current word
                        next_states.add((word_idx, next_node))

                        # second path: go to the next word if applicable
                        if next_node.is_end_of_word:
                            if word_idx + 1 < self.words_nb:
                                next_states.add(
                                    (word_idx + 1, self.word_tries[word_idx + 1])
                                )
                            else:
                                next_states.add((self.words_nb, None))

                current_states = next_states

                # there are no valid states left: the sequence does
                # not respect constraints
                if len(current_states) == 0:
                    break

            allowed_next_tokens = set()
            for _, node in current_states:
                if node is None:
                    allowed_next_tokens.add(self.tokenizer.eos_token_id)
                else:
                    allowed_next_tokens.update(node.children.keys())

            mask = torch.full_like(scores[batch_idx], float("-inf"))
            mask[list(allowed_next_tokens)] = 0
            processed_scores[batch_idx] += mask

        return processed_scores


def get_allowed_words(
    target_words: list[str], vocab: set[str], hash_len: int
) -> list[set[str]]:
    hash2words = defaultdict(set)
    word2hash = {word: hash_token(word, hash_len) for word in vocab}
    for word in word2hash.keys():
        hash2words[hash_token(word, hash_len)].add(word)
    return [hash2words[word2hash[word]] for word in target_words]


def constrained_inference(
    model,
    tokenizer,
    device: Literal["cuda", "cpu"],
    prompt: list[str],
    allowed_words: list[set[str]],
) -> list[str]:
    prompt_tokens = tokenizer(" ".join(prompt), return_tensors="pt").to(device)

    processor = HashMaskLogitsProcessor(
        allowed_words, tokenizer, len(prompt_tokens["input_ids"][0])
    )

    output_ids = model.generate(
        **prompt_tokens,
        # just a "reasonable" default. Normally,
        # HashMaskLogitsProcessor force the model to generate EXACTLY
        # len(allowed_words) tokens. But in case there is a rare bug,
        # we do not want to generate tokens forever.
        max_new_tokens=len(allowed_words) * 5,
        do_sample=False,
        top_p=None,
        top_k=None,
        logits_processor=LogitsProcessorList([processor]),
    )
    # output with prompt. We exclude the last token to remove EOS
    raw_output = tokenizer.decode(output_ids[0][:-1])
    # limit ourselves to generated tokens
    output = raw_output.split(" ")[len(prompt) :]
    # pad if we did not generate everything
    output = output + ["[UNK]"] * (len(allowed_words) - len(output))

    # TODO: force len
    return output


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds  # type: ignore
ex.observers.append(FileStorageObserver("runs"))


@ex.config
def config():
    model_name: str
    start_words_nb: int
    device: Literal["cuda", "cpu"]


@ex.automain
def main(
    _run: Run, model_name: str, start_words_nb: int, device: Literal["cuda", "cpu"]
):
    print_config(_run)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

    # TODO: hardcoded
    target_words = list(iter_book_chapters("./data/Frankenstein/F-1818"))[0]
    prompt: list[str] = "Complete this text from the novel Frankenstein :".split()  # type: ignore
    prompt += target_words[:start_words_nb]
    target_words = target_words[start_words_nb:]
    user_words = list(iter_book_chapters("./data/Frankenstein/F-1823"))[0]

    # TODO: this is a very important assumption.
    vocab = set()
    for editions in EDITION_SETS.values():
        for edition_path in editions.values():
            for chapter in iter_book_chapters(edition_path):
                for word in chapter:
                    vocab.add(word)
    _run.log_scalar("vocab_size", len(vocab))

    pipe_strategy = [
        make_plugin_retokenize(max_token_len=16, max_splits_nb=8),
        make_plugin_mlm("answerdotai/ModernBERT-base", window=32, device=device),
        make_plugin_case(),
        make_plugin_propagate(),
    ]

    progress = tqdm(list(range(16, 3, -1)))
    for hash_len in progress:
        progress.set_description(f"h={hash_len}")

        allowed_words = get_allowed_words(target_words, vocab, hash_len)
        pred_words = constrained_inference(
            model, tokenizer, device, prompt, allowed_words
        )
        _run.log_scalar(
            "model.errors_percent", errors_percent(target_words, pred_words), hash_len
        )

        target_hashed = hash_tokens(target_words, hash_len)
        aligned_words = align_tokens(
            target_hashed,
            user_words,
            hash_len=hash_len,
            alignment_plugins=pipe_strategy,
        )
        _run.log_scalar(
            "alignment.errors_percent",
            errors_percent(target_words, aligned_words),
            hash_len,
        )
