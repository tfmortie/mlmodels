import matplotlib.pyplot as plt
from data import AnimalDataset

def main():
    data = AnimalDataset("/Users/thomas/Github/mlmodels/data/Animals") 
    img, lbl = data[1001]

if __name__ == "__main__":
    main()
