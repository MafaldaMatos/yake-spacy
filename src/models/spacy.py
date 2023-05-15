from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Tuple

import spacy
from spacy.training import Example
from tieval.evaluate import span_identification

from src.data.spacy import SpacyDataLoader
from src.utils import generate_id

logger = logging.getLogger(__name__)


@dataclass
class TransitionParserConfig:
    """
    Configuration for spacy.TransitionBasedParser.v2
    https://spacy.io/api/architectures#TransitionBasedParser

    hidden_width: 8, 16, 32, 64, 128, 256, ...
    maxout_pieces: 1, 2, 3
    use_upper: bool
    """
    hidden_width: int = 64
    maxout_pieces: int = 2
    use_upper: bool = True


@dataclass
class EmbedConfig:
    """
    Configuration for spacy.HashEmbedCNN.v2
    https://spacy.io/api/architectures#HashEmbedCNN

    width: 96, 128 or 300
    depth: between 2 and 8
    embed_size: between 2000 and 10000
    window_size: 1 or 2
    maxout_pieces: between 1 and 3
    subword_features: True for alphabetic languages
    """
    width: int = 96
    depth: int = 4
    embed_size: int = 2000
    window_size: int = 1
    maxout_pieces: int = 3
    subword_features: bool = True


class SpacyModel:
    def __init__(
            self,
            language: str,
            parser_config: TransitionParserConfig = None,
            embed_config: EmbedConfig = None,
    ):
        """
        Creat spacy Named Entity Recognition model.
        To check the supported languages check spacy
        `Language Support <https://spacy.io/usage/models#languages>`_

        language: any language supported in spacy. For instance "en", "pt", or "de".
        """

        self.nlp = spacy.blank(language)

        if parser_config is None:  # default parameters
            parser_config = TransitionParserConfig()

        if embed_config is None:  # default parameters
            embed_config = EmbedConfig()

        config = {
            "moves": None,
            "update_with_oracle_cut_size": 100,
            "model": {
                "@architectures": "spacy.TransitionBasedParser.v2",
                "state_type": "ner",
                "extra_state_tokens": False,
                "hidden_width": parser_config.hidden_width,
                "maxout_pieces": parser_config.maxout_pieces,
                "use_upper": parser_config.use_upper,
                "tok2vec": {
                    "@architectures": "spacy.HashEmbedCNN.v2",
                    "pretrained_vectors": None,
                    "width": embed_config.width,
                    "depth": embed_config.depth,
                    "embed_size": embed_config.embed_size,
                    "window_size": embed_config.window_size,
                    "maxout_pieces": embed_config.maxout_pieces,
                    "subword_features": embed_config.subword_features,
                }
            },
            "incorrect_spans_key": "incorrect_spans",
        }

        self.nlp.add_pipe("ner", config=config)

        # add TIMEX tag to ner model
        ner = self.nlp.get_pipe("ner")
        ner.add_label("TIMEX")

    def predict(self, texts: List[str]) -> List:
        """Make predictions over a set of documents."""
        result = []
        for text in texts:
            prediction = self.nlp(text)
            timexs = [(entity.start_char, entity.end_char) for entity in prediction.ents]
            result.append(timexs)
        return result

    def evaluate(self, texts: List[str], annotations: List[List[Tuple[int, int]]]):
        """Evaluate model effectiveness."""
        predictions = self.predict(texts)
        metrics = span_identification(annotations, predictions)
        return metrics

    def _train_step(
            self,
            data: SpacyDataLoader,
            dropout: float = 0,
    ) -> float:
        """One training epoch over the training set."""

        losses = {}
        data.shuffle()
        for batch in data:

            example = []
            for text, annotation in batch:
                doc = self.nlp.make_doc(text)
                example += [Example.from_dict(doc, annotation)]

            self.nlp.update(example, drop=dropout, losses=losses)

        return losses["ner"] / len(data)

    def fit(
            self,
            train_data: SpacyDataLoader,
            validation_data: SpacyDataLoader,
            n_epochs: int,
            dropout: float,
            model_path: Path
    ) -> dict:
        """Complete training loop for the spacy ner model."""

        train_id = generate_id()
        logger.info(f"Started train loop with id {train_id}")

        metrics = {
            "train": {
                "loss": [],
                "f1": []
            },
            "validation": {
                "f1": []
            }
        }

        self.nlp.begin_training()

        logger.info("      |Loss \t|F1")
        logger.info("Epoch |Train\t|Train\t|Validation")
        best_val_f1 = 0.
        for epoch in range(n_epochs):

            train_loss = self._train_step(train_data, dropout=dropout)
            metrics["train"]["loss"] += [train_loss]

            train_metrics = self.evaluate(train_data.texts, train_data.annotations)
            train_f1 = train_metrics["micro"]["f1"]
            metrics["train"]["loss"] += [train_f1]

            val_metrics = self.evaluate(validation_data.texts, validation_data.annotations)
            val_f1 = val_metrics["micro"]["f1"]
            metrics["validation"]["f1"] += [val_f1]

            logger.info(
                f"{epoch + 1:5} |"
                f"{train_loss:.3f}\t|"
                f"{train_f1:.4f}\t|"
                f"{val_f1:.4f}"
            )

            # early stopping
            if n_epochs >= 3 and max(metrics["validation"]["f1"][-3:]) < best_val_f1:
                break

            # store model that presents the best f1 score on the validation set
            if val_f1 > best_val_f1:
                logger.debug("Stored weights.")
                self.save(Path(model_path / train_id))
                best_val_f1 = val_f1

        return metrics

    def load(self, path: Path) -> None:
        """Load models from disk."""
        self.nlp = spacy.load(path)

    def save(self, path: Path) -> None:
        """Store model in disk."""
        if not path.is_dir():
            path.mkdir(parents=True)
        self.nlp.to_disk(path)
