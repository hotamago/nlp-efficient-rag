import time

import torch
from transformers import AutoModel

t = time.time()
model = AutoModel.from_pretrained("naver/provence-reranker-debertav3-v1", trust_remote_code=True)
if torch.cuda.is_available():
    model = model.to("cuda")
print("load s:", round(time.time() - t, 1))
question = "How tall is the Eiffel Tower?"
ctx = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel. The tower is 330 metres tall, about the "
    "same height as an 81-storey building. Paris is also famous for its cafes and museums."
)
t = time.time()
out = model.process(question, ctx)
print("process s:", round(time.time() - t, 2))
print(out)
