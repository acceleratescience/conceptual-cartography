import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging


# suppress the annoying tokenization length warning.
transformers_logging.set_verbosity(40)


class ContextEmbedder:
    def __init__(self, model_name: str):
        """An embedder for text contexts.

        Args:
            model_name (str): The name of the model to use. Must be a model from the HuggingFace Transformers library.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            strip_accents=False,  # If not needed
            clean_up_tokenization_spaces=False,  # If not needed
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        self.model.eval()

        # max input size
        self.max_length = self.model.config.max_position_embeddings

    def __call__(
        self,
        sentences: list[str],
        target_word: str,
        context_window: int,
        line_context_only: bool = False,
    ) -> Tuple[torch.Tensor, list[list[int]], list[int]]:
        """Create embeddings for target word occurrences in the provided sentences.

        Args:
            sentences (List[str]): List of sentences to process.
            target_word (str): The target word to extract the context for.
            context_window (int): The size of the context window to extract.
            line_context_only (bool): If True, only use the line where the target word appears as context.

        Returns:
            Tuple[torch.Tensor, List[List[int]], List[int]]: Returns embeddings tensor, contexts, and indices.
        """
        if 2 * context_window > self.max_length and not line_context_only:
            raise ValueError(
                f"The context window is too large for the model. The maximum context window is {self.max_length // 2}."
            )

        target_word = target_word.lower()
        pattern = r"\b" + re.escape(target_word) + r"\b"

        if line_context_only:
            # Process each line separately
            all_contexts = []
            all_indices = []

            for sentence in sentences:
                if not sentence.strip():
                    continue

                sentence_lower = sentence.lower()

                if not re.search(pattern, sentence_lower):
                    continue

                # Tokenize just this sentence
                encoding = self.tokenizer(
                    sentence, add_special_tokens=False, return_offsets_mapping=True
                )
                tokens = encoding["input_ids"]
                offset_mapping = encoding["offset_mapping"]

                text_occurrences = []
                for match in re.finditer(pattern, sentence_lower):
                    text_occurrences.append((match.start(), match.end()))

                token_occurrences = []
                for text_start, text_end in text_occurrences:
                    token_start = None
                    token_end = None

                    for idx, (start, end) in enumerate(offset_mapping):
                        if start == text_start:
                            token_start = idx
                        if end == text_end:
                            token_end = idx + 1
                            break

                    if token_start is not None and token_end is not None:
                        token_occurrences.append((token_start, token_end))

                for token_start, token_end in token_occurrences:
                    all_contexts.append(tokens)
                    all_indices.append(token_start)
        else:
            # Join all text and process as one
            text = " ".join(sentences).lower()

            # Tokenize the full text with offsets
            encoding = self.tokenizer(
                text, add_special_tokens=False, return_offsets_mapping=True
            )
            tokens = encoding["input_ids"]
            offset_mapping = encoding["offset_mapping"]

            # Find exact target word occurrences using regex with word boundaries
            text_occurrences = []
            for match in re.finditer(pattern, text):
                text_occurrences.append((match.start(), match.end()))

            # Map text occurrences to token indices
            token_occurrences = []
            for text_start, text_end in text_occurrences:
                token_start = None
                token_end = None

                for idx, (start, end) in enumerate(offset_mapping):
                    if start == text_start:
                        token_start = idx
                    if end == text_end:
                        token_end = idx + 1
                        break

                if token_start is not None and token_end is not None:
                    token_occurrences.append((token_start, token_end))

            # Extract contexts
            all_contexts = []
            all_indices = []

            for token_start, token_end in token_occurrences:
                # Calculate context boundaries
                context_start = max(0, token_start - context_window)
                context_end = min(len(tokens), token_end + context_window)

                # Extract context
                context = tokens[context_start:context_end]
                target_pos = token_start - context_start

                all_contexts.append(context)
                all_indices.append(target_pos)

        if len(all_contexts) == 0:
            return torch.zeros(1, self.model.config.hidden_size), [], []

        # Prepare all contexts at once
        padded_contexts = []
        attention_masks = []
        max_len = (
            max(len(context) for context in all_contexts) + 2
        )  # +2 for special tokens

        for context in all_contexts:
            # Add special tokens
            padded = (
                [self.tokenizer.cls_token_id] + context + [self.tokenizer.sep_token_id]
            )
            attention_mask = [1] * len(padded) + [0] * (max_len - len(padded))
            padded = padded + [self.tokenizer.pad_token_id] * (max_len - len(padded))

            padded_contexts.append(padded)
            attention_masks.append(attention_mask)

        # Convert to tensors
        inputs = torch.tensor(padded_contexts).to(self.device)
        attention_mask = torch.tensor(attention_masks).to(self.device)

        # Single forward pass
        with torch.no_grad():
            outputs = self.model(
                inputs, attention_mask=attention_mask, output_hidden_states=True
            ).last_hidden_state

            # Extract embeddings for each occurrence
            embeddings = []
            for i, context in enumerate(all_contexts):
                start_idx = all_indices[i] + 1  # +1 for [CLS] token
                # Get the embedding for the target word
                word_embedding = (
                    outputs[i, start_idx : start_idx + 1, :]
                    .mean(dim=0, keepdim=True)
                    .cpu()
                )
                embeddings.append(word_embedding)

            embeddings = torch.cat(embeddings, dim=0)

        return embeddings, all_contexts, all_indices

    def process_file(
        self,
        file_path: str,
        target_word: str,
        context_window: int,
        line_context_only: bool = False,
    ) -> Tuple[torch.Tensor, list[list[int]], list[int]]:
        """Process a text file and create embeddings for the target word occurrences.

        Args:
            file_path (str): The path to the text file to process.
            target_word (str): The target word to extract the context for.
            context_window (int): The size of the context window to extract.
            line_context_only (bool): If True, only use the line where the target word appears as context.

        Returns:
            Tuple[torch.Tensor, List[List[int]], List[int]]: Returns embeddings tensor, contexts, and indices.
        """
        sentences = self._read_file(file_path)
        return self.__call__(sentences, target_word, context_window, line_context_only)

    def process_directory(
        self,
        directory_path: str,
        target_word: str,
        context_window: int,
        line_context_only: bool = False,
    ):
        """Process all text files in a directory.

        Args:
            directory_path (str): The path to the directory containing text files.
            target_word (str): The target word to extract the context for.
            context_window (int): The size of the context window to extract.
            line_context_only (bool): If True, only use the line where the target word appears as context.

        Returns:
            Tuple[torch.Tensor, List[List[str]]]: Returns embeddings tensor and contexts.
        """
        files = glob.glob(f"{directory_path}/*.txt")

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.process_file, f, target_word, context_window, line_context_only
                )
                for f in tqdm(files, desc="Submitting files", unit="file")
            ]

            results = []
            for future in tqdm(futures, desc="Processing files", unit="file"):
                results.append(future.result())

        word_embeddings = []
        contexts = []
        for embeddings, context, _ in results:
            if len(context) > 0:
                word_embeddings.append(embeddings)
                contexts.extend(context)

        return torch.cat(word_embeddings, dim=0), contexts

    def _read_file(self, file_path: str) -> List[str]:
        """Read and preprocess text from a file.

        Args:
            file_path (str): The path to the text file.

        Returns:
            List[str]: List of preprocessed sentences from the file.
        """
        with open(file_path, "r") as file:
            # Skip first line and get remaining lines
            lines = file.read().split("\n")[1:]
            return lines
