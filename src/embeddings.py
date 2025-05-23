import glob
import os
import re
# Keep for potential pre-processing? Tried but it seems actually slower...
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

# suppress warnings, not sure why this is needed
transformers_logging.set_verbosity(40)


class ContextEmbedder:
    def __init__(self, model_name: str):
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            strip_accents=False,
            clean_up_tokenization_spaces=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        self.model.eval()
        self.max_length = self.model.config.max_position_embeddings
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.cls_token_id is None or self.sep_token_id is None or self.pad_token_id is None:
            raise ValueError(
                "Tokenizer must have CLS, SEP, and PAD tokens defined.")

    @staticmethod
    def _find_occurrences_in_sentence_static(args: Dict) -> List[Tuple[str, int]]:
        """_summary_

        Args:
            args (Dict): _description_

        Returns:
            List[Tuple[str, int]]: _description_
        """
        sentence_text: str = args["sentence_text"]
        pattern_str: str = args["pattern_str"]

        results = []
        if not sentence_text.strip():
            return results

        sentence_lower = sentence_text.lower()
        for match in re.finditer(pattern_str, sentence_lower):
            results.append((sentence_text, match.start()))
        return results

    def _prepare_model_batch(self, current_batch_token_ids: List[List[int]], current_batch_target_indices: List[int]):
        """_summary_

        Args:
            current_batch_token_ids (List[List[int]]): _description_
            current_batch_target_indices (List[int]): _description_

        Returns:
            _type_: _description_
        """
        batch_max_len = 0
        for tokens in current_batch_token_ids:
            # +2 for [CLS] and [SEP]
            batch_max_len = max(batch_max_len, len(tokens) + 2)

        batch_max_len = min(batch_max_len, self.max_length)

        padded_input_ids = []
        attention_masks = []
        adjusted_target_indices = []

        for i, tokens in enumerate(current_batch_token_ids):
            truncated_tokens = tokens[:batch_max_len - 2]

            input_ids_with_special = [self.cls_token_id] + \
                truncated_tokens + [self.sep_token_id]
            padding_len = batch_max_len - len(input_ids_with_special)

            padded_ids = input_ids_with_special + \
                [self.pad_token_id] * padding_len
            attn_mask = [1] * len(input_ids_with_special) + [0] * padding_len

            padded_input_ids.append(padded_ids)
            attention_masks.append(attn_mask)

            if current_batch_target_indices[i] < len(truncated_tokens):
                adjusted_target_indices.append(
                    current_batch_target_indices[i] + 1)
            else:
                adjusted_target_indices.append(-1)  # Placeholder for invalid

        return (
            torch.tensor(padded_input_ids, dtype=torch.long).to(self.device),
            torch.tensor(attention_masks, dtype=torch.long).to(self.device),
            adjusted_target_indices,
        )

    def __call__(
        self,
        sentences: list[str],
        target_word: str,
        context_window: int | None = None,
        model_batch_size: int = 64,
    ) -> Tuple[torch.Tensor, list[list[int]], list[int]]:
        """_summary_

        Args:
            sentences (list[str]): A list of sentences to search for the target word.
            target_word (str): The target word to find in the sentences.
            context_window (int | None, optional): Total size of context window. The number of
                tokens surrounding the target word will be context_window // 2. Defaults to None.
            model_batch_size (int, optional): Batch size to be fed into the model only. Does not
                batch the input to the tokenizer. Defaults to 64.

        Raises:
            ValueError: If context_window is larger than the model's max length.

        Returns:
            Tuple[torch.Tensor, list[list[int]], list[int]]: eturns the final_embeddings, 
                collected_valid_contexts, collected_valid_indices
        """

        

        target_word_lower = target_word.lower()
        pattern = r"\b" + re.escape(target_word_lower) + r"\b"

        contexts_for_model_input: List[List[int]] = []
        target_indices_in_context: List[int] = []

        # if context_window is None, we use the whole sentence
        if context_window is None:
            lines_with_target_info = []
            for sentence_idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                sentence_lower = sentence.lower()
                for match in re.finditer(pattern, sentence_lower):
                    lines_with_target_info.append((sentence, match.start()))

            if not lines_with_target_info:
                return torch.zeros(1, self.model.config.hidden_size), [], []

            tokenizer_batch_size = model_batch_size * 4

            for i in range(0, len(lines_with_target_info), tokenizer_batch_size):
                current_tokenizer_batch = lines_with_target_info[i:i +
                                                                tokenizer_batch_size]
                batch_texts = [item[0] for item in current_tokenizer_batch]
                batch_char_offsets = [item[1]
                                    for item in current_tokenizer_batch]

                encodings = self.tokenizer(
                    batch_texts,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    padding=False,
                    truncation=True,
                    max_length=self.max_length - 2
                )

                for j, text_idx_in_batch in enumerate(range(len(batch_texts))):
                    char_start_offset = batch_char_offsets[j]
                    token_ids = encodings.input_ids[text_idx_in_batch]
                    offset_mapping = encodings.offset_mapping[text_idx_in_batch]

                    target_token_start = -1
                    for token_idx, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
                        if tok_char_start == char_start_offset:
                            target_token_start = token_idx
                            break

                    if target_token_start != -1:
                        contexts_for_model_input.append(token_ids)
                        target_indices_in_context.append(target_token_start)

        else:
            if context_window is not None and 2 * context_window > self.max_length:
                raise ValueError(
                    f"Context window {context_window} too large for model max length {self.max_length}. Max context window: {self.max_length // 2}"
                )   
            full_text = " ".join(s for s in sentences if s.strip()).lower()
            if not full_text:
                return torch.zeros(1, self.model.config.hidden_size, dtype=self.model.dtype), [], []

            encoding_full_text = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True
            )
            all_tokens = encoding_full_text["input_ids"]
            all_offset_mapping = encoding_full_text["offset_mapping"]

            text_occurrences = []
            for match in re.finditer(pattern, full_text):
                text_occurrences.append((match.start(), match.end()))

            token_occurrence_spans = []
            # Find the token spans for each occurrence of the target word
            for text_start, text_end in text_occurrences:
                tok_start, tok_end = -1, -1
                for idx, (char_s, char_e) in enumerate(all_offset_mapping):
                    if char_s == text_start:
                        tok_start = idx
                    if char_e == text_end:
                        tok_end = idx + 1
                        break
                        
                if tok_start != -1 and tok_end != -1 and tok_start < tok_end:
                    token_occurrence_spans.append((tok_start, tok_end))
            for token_start, token_end_of_target in token_occurrence_spans:
                context_s = max(0, token_start - context_window)
                context_e = min(len(all_tokens),
                                token_end_of_target + context_window)

                actual_context_token_list = all_tokens[context_s:context_e]
                target_pos_in_this_window = token_start - context_s

                contexts_for_model_input.append(actual_context_token_list)
                target_indices_in_context.append(target_pos_in_this_window)

        if not contexts_for_model_input:
            return torch.zeros(1, self.model.config.hidden_size, dtype=self.model.dtype), [], []

        all_embeddings_list = []
        hidden_embeddings_list = []
        collected_valid_contexts = []
        collected_valid_indices = []

        for i in tqdm(range(0, len(contexts_for_model_input), model_batch_size), desc="Inferring batches"):
            batch_contexts = contexts_for_model_input[i:i+model_batch_size]
            batch_target_starts = target_indices_in_context[i:i +
                                                            model_batch_size]

            input_ids_tensor, attn_mask_tensor, adjusted_target_token_indices = \
                self._prepare_model_batch(batch_contexts, batch_target_starts)

            if input_ids_tensor.numel() == 0:
                continue  # Skip empty batches, obviously

            with torch.no_grad():
                outputs = self.model(
                    input_ids_tensor, attention_mask=attn_mask_tensor, output_hidden_states=True)
                
                hidden_states = outputs.hidden_states
                last_hidden_state = outputs.last_hidden_state

            for j, target_tok_idx in enumerate(adjusted_target_token_indices):

                if target_tok_idx != -1 and target_tok_idx < last_hidden_state.shape[1]:

                    final_layer_embedding = last_hidden_state[j,
                                            target_tok_idx:target_tok_idx+1, :].mean(dim=0)
                    all_embeddings_list.append(final_layer_embedding.unsqueeze(
                        0).cpu())

                    collected_valid_contexts.append(batch_contexts[j])
                    collected_valid_indices.append(batch_target_starts[j])

                    hidden_states_for_target = []
                    for state in hidden_states:

                        state_embedding = state[j, target_tok_idx:target_tok_idx+1, :].mean(dim=0)
                        hidden_states_for_target.append(state_embedding.unsqueeze(0).cpu())
                    
                    hidden_states_for_target = torch.cat(hidden_states_for_target, dim=0)
                    hidden_embeddings_list.append(hidden_states_for_target)

        if not all_embeddings_list:
            return torch.zeros(1, self.model.config.hidden_size, dtype=self.model.dtype), [], []

        final_embeddings = torch.cat(all_embeddings_list, dim=0)
        hidden_embeddings = torch.stack(hidden_embeddings_list)

        valid_sentences = [self.tokenizer.decode(context) for context in collected_valid_contexts]

        output = {
            'final_embeddings' : final_embeddings,
            'hidden_embeddings' : hidden_embeddings,
            'valid_contexts' : valid_sentences,
            'valid_indices' : collected_valid_indices
        }

        return output
