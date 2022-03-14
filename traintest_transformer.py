from datasets import load_dataset

def main():
    # load RU-EN dataset
    data = load_dataset("wmt19","ru-en")
    print(f'{data["train"][0]=}')

if __name__=="__main__":
    main()
