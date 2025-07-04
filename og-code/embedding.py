import logging
from typing import Union

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

from .utils import get_device

transformers_logging.set_verbosity(40)  # suppress the annoying tokenization length warning.


class BNCContextEmbedder:
    def __init__(self, model_name: str, checkpoint_path: str = None):
        """An embedder for the BNC corpus.

        Args:
            model_name (str): The name of the model to use. Must be a model from the HuggingFace Transformers library.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.device = get_device()

        self.model.to(self.device)
        self.model.eval()

        # max input size
        self.max_length = self.model.config.max_position_embeddings

    def __call__(self, file_path: str, target_word: str, context_window: int):
        """Create embeddings of a list of texts using a pretrained Model.

        Args:
            file_path (str): The path to the text file to embed.
            target_word (str): The target word to extract the context for.
            context_window (int): The size of the context window to extract. Should be less than the max length of the model, otherwise a warning will be raised.

        Returns:
            _type_: _description_
        """
        # check that context_window is not bigger than the max length the model can handle and if so, raise a warning
        if 2 * context_window > self.max_length:
            raise ValueError(
                f"The context window is too large for the model. The maximum context window is {self.max_length // 2}."
            )

        text, tokens, target_tokens = self._extract_tokens(file_path, target_word)
        occurrences = self._find_sequence_occurrences(tokens, target_tokens)
        contexts, indices = self._extract_context(
            occurrences, tokens, target_tokens, context_window
        )

        if len(contexts) == 0:
            return torch.zeros(1, self.model.config.hidden_size), [], []
        else:
            embeddings = []
            with torch.no_grad():
                for i, context in enumerate(contexts):
                    # add special tokens to context
                    context = [101] + context + [102]
                    inputs = torch.tensor(context).unsqueeze(0).to(self.device)
                    outputs = self.model(inputs, output_hidden_states=True).last_hidden_state
                    word_embedding = outputs[:, indices[i]+1, :].cpu()
                    embeddings.append(word_embedding)

                embeddings = torch.cat(embeddings, dim=0)

            return embeddings, contexts, indices

    def _find_sequence_occurrences(self, target_list: list, sequence: list) -> list:
        """Given a list of tokens, find the indices of the sequence in the list.

        Args:
            target_list (list): The list of tokens to search through.
            sequence (list): The sequence of tokens to search for.

        Returns:
            list: A list of indices where the sequence occurs in the target list.
        """
        occurrences = []
        sequence_length = len(sequence)
        for i in range(len(target_list) - sequence_length):
            if target_list[i : i + sequence_length] == sequence:
                # check to see if the previous or subsequent token begins with a #
                if i > 0 and i < len(target_list) - sequence_length:
                    if self.tokenizer.decode(target_list[i + 1]).startswith("##"):
                        continue

                occurrences.append(i)

        return occurrences

    def _extract_tokens(self, file_path: str, target_word: str) -> Union[str, list, list]:
        """Extract the tokens from a text file and encode them using the tokenizer.

        Args:
            file_path (str): The path to the text file to embed.
            target_word (str): The target word to extract the context for.

        Returns:
            Union[str, list, list]: The text, the tokenized text, and the tokenized target word.
        """
        with open(file_path, "r") as file:
            text = file.read()
            # remove the first line
            text = text.split("\n")[1:]
            # make everything lower case
            text = [line.lower() for line in text]
            text = " ".join(text)

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        target_input_ids = self.tokenizer.encode(target_word, add_special_tokens=False)

        return text, input_ids, target_input_ids

    def _extract_context(
        self, occurences: list, input_ids: list, target_input_ids: list, context_window: int = 30
    ) -> Union[list, list]:
        """Extract the context for each occurence of the target word.

        Args:
            occurences (list): The list of indices where the target word occurs.
            input_ids (list): The list of tokenized input ids.
            target_input_ids (list): The list of tokenized target word ids.
            context_window (int, optional): Size of the surrounding tokens to extract. Defaults to 30.

        Returns:
            Union[list, list]: A list of the contexts and a list of the indices of the target word within the context.
        """
        contexts = []
        indices = []
        for occurence in occurences:
            start = max(0, occurence - context_window + 1)
            end = occurence + context_window + 1
            indices.append(occurence - start)
            context = input_ids[start:end]
            contexts.append(context)
            # get index of target word within context

        return contexts, indices

    def fine_tune(
        self, data_path: str, epochs: int, learning_rate: float, batch_size: int, save_path: str
    ):
        """Fine tunes the model on the BNC corpus.

        Args:
            data_path (str): The path to the BNC corpus.
            epochs (int): The number of epochs to train for.
            learning_rate (float): The learning rate for the optimizer.
            batch_size (int): The batch size for the training data.
            save_path (str): The path to save the model to.
        """
        # Load the BNC corpus
        with open(data_path, "r") as f:
            bnc_corpus = f.read()

        tokenized_corpus = self.tokenizer(
            bnc_corpus, return_tensors="pt", padding=True, truncation=True
        )
        dataloader = torch.utils.data.DataLoader(
            tokenized_corpus, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)

                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss

                loss.backward()
                optimizer.step()

        # Save the model
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
