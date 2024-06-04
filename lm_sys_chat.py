def main(args):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--small", action="store_true", help="if true, use gpt2, else, use gpt2-xl")
    parser.add_argument("-t", "--tokenize", action="store_true", help="tokenize texts")

    args = parser.parse_args()

    main(args)
