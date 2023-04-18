import logging
from pathlib import Path

from pyspark.sql.session import SparkSession
from pyspark.sql.functions import rand
from sparknlp.base import DocumentAssembler, Pipeline, LightPipeline
from sparknlp.annotator import SentenceDetector, Tokenizer
from sparknlp.training import CoNLL

logger = logging.getLogger(__name__)


class SparkNLPDataLoader:

    def __init__(self, path: Path, spark: SparkSession):
        dataframe = CoNLL().readDataset(spark, str(path))
        dataframe = dataframe.orderBy(rand(seed=73))
        self.dataframe = dataframe.repartition(500)


def _build_annotation_pipeline() -> Pipeline:
    doc_assemble = DocumentAssembler() \
        .setInputCol("text") \
        .setOutputCol("document")

    sentence_detector = SentenceDetector() \
        .setInputCols("document") \
        .setOutputCol("sentence")

    tokenizer = Tokenizer() \
        .setInputCols("sentence") \
        .setOutputCol("token")

    return Pipeline(stages=[
        doc_assemble,
        sentence_detector,
        tokenizer,
    ])


def build_annotation_model(spark: SparkSession) -> LightPipeline:
    annot_pipeline = _build_annotation_pipeline()
    empty_df = spark.createDataFrame([[""]]).toDF("text")
    annot_model = LightPipeline(annot_pipeline.fit(empty_df))
    return annot_model
