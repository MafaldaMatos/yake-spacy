import logging
import os
from pathlib import Path
from typing import List, Tuple

from sparknlp.base import (
    DocumentAssembler,
    Pipeline,
    LightPipeline
)
from sparknlp.annotator import (
    NerDLApproach,
    NerDLModel,
    NerConverter,
    WordEmbeddingsModel,
    Tokenizer,
    SentenceDetector,
)
from tieval.evaluate import span_identification

from src.data.sparknlp import SparkNLPDataLoader
from src.utils import parse_sparknlp_log, generate_id

logger = logging.getLogger(__name__)


class SparkNLPModel:
    def __init__(
            self,
            spark,
            embeddings_path: Path
    ):
        self.spark = spark
        self.ner_model = None
        self.embeddings_path = embeddings_path
        self.ner_path = None
        self.prediction_model = None

    def predict(self, texts: List[str]) -> List:
        """Make predictions over a set of documents."""
        result = []
        for text in texts:
            prediction = self.prediction_model.fullAnnotate([text])
            timexs = [(annot.begin, annot.end + 1) for annot in prediction[0]["ner_chunk"]]
            result.append(timexs)
        return result

    def evaluate(self, texts: List[str], annotations: List[List[Tuple[int, int]]]):
        """Evaluate model effectiveness."""
        predictions = self.predict(texts)
        metrics = span_identification(annotations, predictions)
        return metrics

    def fit(
            self,
            train_data: SparkNLPDataLoader,
            n_epochs: int,
            batch_size: int,
            learning_rate: float,
            dropout: float,
            model_path: Path,
            logs_path: Path
    ) -> dict:
        """Complete training loop for the spacy ner model."""

        train_id = generate_id()

        logger.info(f"Started train loop with id {train_id}")
        logger.info("Building training pipeline.")
        training_pipeline = build_training_pipeline(
            embeddings_path=self.embeddings_path,
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout=dropout,
            logs_path=logs_path
        )

        logger.info("Initialize training.")
        training_model = training_pipeline.fit(train_data.dataframe)
        logger.info("Train done.")

        logger.info("Storing NER weights.")
        self.ner_model = training_model.stages[1]
        self.ner_path = model_path / train_id
        self.save(self.ner_path)

        self.prediction_model = build_prediction_model(
            embeddings_path=self.embeddings_path,
            ner_path=self.ner_path,
            spark=self.spark
        )

        logger.info("Retrieving metrics from sparknlp log file. \n"
                    "Notice that this metrics are computed on the BIO "
                    "annotation and not on the tag identification level.")
        log_file_path = os.listdir(logs_path)[0]  # assumes that the folder only contains one file.
        metrics = parse_sparknlp_log(logs_path / log_file_path)
        os.remove(logs_path / log_file_path)

        return metrics

    def save(self, path: Path) -> None:
        """Store NER model in disk."""
        self.ner_model.write().overwrite().save(str(path))

    def load(self, path: Path) -> None:
        """Load NER model from disk."""
        self.prediction_model = build_prediction_model(
            embeddings_path=self.embeddings_path,
            ner_path=path,
            spark=self.spark
        )


def build_training_pipeline(
        n_epochs: int,
        batch_size: int,
        learning_rate: float,
        dropout: float,
        embeddings_path: Path,
        logs_path: Path,
):
    """Build a trainable sparknlp pipeline."""
    embeddings = WordEmbeddingsModel.load(str(embeddings_path)) \
        .setInputCols("sentence", "token") \
        .setOutputCol("embeddings")

    ner = NerDLApproach() \
        .setInputCols("sentence", "token", "embeddings") \
        .setLabelColumn("label") \
        .setOutputCol("ner") \
        .setMaxEpochs(n_epochs) \
        .setLr(learning_rate) \
        .setBatchSize(batch_size) \
        .setDropout(dropout) \
        .setVerbose(1) \
        .setValidationSplit(0.2) \
        .setEvaluationLogExtended(True) \
        .setEnableOutputLogs(True) \
        .setOutputLogsPath(str(logs_path)) \
        .setIncludeConfidence(True) \
        .setEnableMemoryOptimizer(True) \
        .setUseBestModel(True) \
        .setRandomSeed(73)

    return Pipeline(stages=[
        embeddings,
        ner,
    ])


def build_prediction_model(
        embeddings_path: Path,
        ner_path: Path,
        spark
):
    """Build a sparknlp pipeline with specific weights."""
    doc_assemble = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = SentenceDetector() \
        .setInputCols("document") \
        .setOutputCol("sentence")

    tokenizer = Tokenizer() \
        .setInputCols("sentence") \
        .setOutputCol("token")

    embeddings = WordEmbeddingsModel.load(str(embeddings_path)) \
        .setInputCols("sentence", "token") \
        .setOutputCol("embeddings")

    ner = NerDLModel.load(str(ner_path)) \
        .setInputCols("sentence", "token", "embeddings") \
        .setOutputCol("ner")

    ner_converter = NerConverter() \
        .setInputCols("sentence", "token", "ner") \
        .setOutputCol("ner_chunk")

    pipeline = Pipeline(stages=[
        doc_assemble,
        sentence_detector,
        tokenizer,
        embeddings,
        ner,
        ner_converter
    ])

    empty_df = spark.createDataFrame([[""]]).toDF("text")
    return LightPipeline(pipeline.fit(empty_df))
