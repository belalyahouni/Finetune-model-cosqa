import mteb
import logging
from sentence_transformers import SentenceTransformer
from mteb import MTEB

logger = logging.getLogger(__name__)

model_name = 'intfloat/e5-base-v2'
model = SentenceTransformer(model_name)
tasks = mteb.get_tasks(
    tasks=[
        "AppsRetrieval",
        "CodeFeedbackMT",
        "CodeFeedbackST",
        "CodeTransOceanContest",
        "CodeTransOceanDL",
        "CosQA",
        "SyntheticText2SQL",
        "StackOverflowQA",
        "COIRCodeSearchNetRetrieval",
        "CodeSearchNetCCRetrieval",
    ]
)
evaluation = MTEB(tasks=tasks)
results = evaluation.run(
    model=model,
    overwrite_results=True
)
print(result)
