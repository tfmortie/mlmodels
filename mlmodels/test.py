import matplotlib.pyplot as plt
from data import AnimalDataset
from nf import NF

def main():
    data = AnimalDataset("/home/data/tfmortier/Github/mlmodels/data/Animals") 
    test_model = NF(4)
    print(f'{len(data)=}')

if __name__ == "__main__":
    main()
