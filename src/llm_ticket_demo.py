import os
import lmstudio as lms

model = lms.llm(os.environ["LM_STUDIO_MODEL"])
prompt = "Jawab JSON dengan kunci summary, priority, reason, missing_info. Tiket: Pembayaran gagal dan saldo terpotong."
result = model.respond(prompt)
print(result.content)
