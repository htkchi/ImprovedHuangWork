import numpy as np

class DataLoader:
    def __init__(self, problem_idx, shuffle=True):
        problem_names = [
            "set_cover",
            "traveling_salesman",
            "general_mip",
            "knapsack",
            "facility_location"
        ]
        name = problem_names[problem_idx]
        self.name = name
        self.features = np.array([])
        data_path = f'./training_bags/{name}.npz'   # Set the correct data path here
        training_data = np.load(data_path)
        self.features = training_data["features"]
        self.labels = training_data["labels"]

        #training_data = np.load('./cut_features/' + name + '/' + name + "_pretrain.npz")
        #self.features = training_data['arr_0']
        #self.labels = training_data['arr_1']
        if shuffle:
            shuffle_idx = np.arange(self.features.shape[0])
            np.random.shuffle(shuffle_idx)
            self.features = self.features[shuffle_idx,:]
            self.labels = self.labels[shuffle_idx]

if __name__ == "__main__":
    dataLoader = DataLoader(problem_idx=0, shuffle=True)
    print(dataLoader.features.shape)
    print(dataLoader.labels.shape)
