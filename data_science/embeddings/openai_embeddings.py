from uuid import uuid4
import pandas as pd
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

class OpenAIEmbedder():
    def __init__(self, model_name="text-embedding-3-small"):
        self.batch_data_udf = self.make_batch_data_udf(model_name)
        
    def make_batch_data_udf(self, model_name: str):
        @pandas_udf(
            ArrayType(
                StructType([
                    StructField("custom_id", StringType(), True),
                    StructField("method", StringType(), True),
                    StructField("url", StringType(), True),
                    StructField(
                        "body",
                        StructType([
                            StructField("model", StringType(), True),
                            StructField("input", StringType(), True)
                        ]),
                        True
                    )
                ])
            )
        )
        def batch_data(texts: pd.Series) -> pd.Series:
            batch = []
            for row in texts:
                if isinstance(row, str):
                    request = {
                        "custom_id": str(uuid4()),
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "model": model_name,
                            "input": row
                        }
                    }
                    batch.append(request)
            return pd.Series(batch)
        return batch_data