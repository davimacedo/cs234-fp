from transformers import pipeline

def gpt2():
    pipe = pipeline(
        "text-generation",
        model="gpt2-xl",
        device_map="auto",
        max_length=512
    )

    out = pipe("Q: What is the best in Soccer, Brazil or Argentina?\nA: \xa0The Best is the Brazilian side. \xa0They have everything, with fantastic players like Fredy, Carlos Alberto, Falcao, and Neymar!\nQ: Isn't Argentina better?\nA: ")

    print(out)
