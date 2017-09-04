class Parser:
    def parse_iris_data(self, data):
        parse_data = []
        for line in data:
            inputs = line.split(",")
            for i in range(len(inputs) - 1):
                inputs[i] = float(inputs[i])
            inputs[len(inputs) - 1] = self.classify_iris_class(inputs[len(inputs) - 1])
            parse_data.append(inputs)
        return parse_data

    def classify_iris_class(self, name):
        return {
            'Iris-setosa': [1, 0, 0],
            'Iris-versicolor': [0, 1, 0],
            'Iris-virginica': [0, 0, 1]
        }[name]
