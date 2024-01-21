"""
    def load_bottlenecks(self):
        list = []
        for dir in os.listdir(self.bottleneckDir):
            for file in os.listdir(self.bottleneckDir + '/' + dir):
                full_name = self.bottleneckDir + '/' + dir + '/' + file
                self.image_files.append(full_name)
                with open(full_name, 'rt') as f:
                    list.append([float(y) for y in f.readlines()[0].split(",")])
        self.bottlenecks = np.array(list)
        self.bottlenecks = np.transpose(self.bottlenecks)
"""