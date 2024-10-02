from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="aapoliakova/cls_level_bsf")
res = classifier("I love you")
print(res[0]['label'])