from src.Parser import Parser


class FileManager:
    def __init__(self):
        self.data = None
        self.train_data = []
        self.test_data = []

    def load_file(self, file_name):
        with open(file_name) as file:
            self.data = file.readlines()
        self.data = [x.strip() for x in self.data]
        parser = Parser()
        self.data = parser.parse_iris_data(self.data)
        x = 0
        for i in range(len(self.data)):
            normalize_data = list(map(normalize,self.data[i][0:len(self.data[i])-1]))
            normalize_data.append(self.data[i][-1])
            self.data[i] = normalize_data
            if x % 2 == 0:
                self.test_data.append(self.data[i])
            else:
                self.train_data.append(self.data[i])
            x += 1

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


def normalize(data):
    return (((data - 0)*(1-0))/(8-0))+0


