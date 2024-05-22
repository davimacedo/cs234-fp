from transformers import pipeline

def gpt2():
    pipe = pipeline(
        "text-generation",
        model="gpt2-xl",
        device_map="auto"
    )

    out = pipe("Q: What is the best in Soccer, Brazil or Argentina?\nA:")

    print(out)
