from transformers import pipeline

def gp2():
    pipe = pipeline('text-generation', model='gpt2')

    out = pipe("Hi!")

    print(out[0])
