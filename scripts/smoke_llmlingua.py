import time

import torch
from llmlingua import PromptCompressor

ctx = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
    "It is named after the engineer Gustave Eiffel, whose company designed and built the tower "
    "from 1887 to 1889. Locally nicknamed La dame de fer, it was constructed as the centerpiece "
    "of the 1889 World's Fair. The tower is 330 metres tall, about the same height as an "
    "81-storey building, and the tallest structure in Paris."
)
print("cuda:", torch.cuda.is_available())
t = time.time()
pc = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)
print("load s:", round(time.time() - t, 1))
t = time.time()
out = pc.compress_prompt([ctx], rate=0.5, force_tokens=["\n", "?"])
print("compress s:", round(time.time() - t, 2))
print({k: v for k, v in out.items() if k != "compressed_prompt_list"})
