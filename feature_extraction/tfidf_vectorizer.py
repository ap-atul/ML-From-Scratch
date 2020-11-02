import math


def tokenize(document, separator):
    return document.split(separator)


class TfidfVectorizer:
    def __init__(self, separator=" ", max_features=10):
        self.separator = separator
        self.maxFeatures = max_features
        self.noOfDocuments = 0
        self.corpus = dict()
        self.tf = dict()
        self.idf = dict()

    def fit(self, texts):
        self.noOfDocuments = len(texts)

        # finding the frequency of each word
        for index, text in enumerate(texts):

            textNormalize = dict()
            words = tokenize(text, self.separator)

            # calculating the frequency
            for word in words:
                textNormalize[word] = textNormalize.get(word, 0.0) + 1

            # normalizing / calculating the probability for each text
            length = float(len(words))
            for value in textNormalize:
                textNormalize[value] = (textNormalize.get(value) / length)

            # adding the tf for each doc
            self.corpus[index] = textNormalize.keys()
            self.tf[index] = textNormalize

    def transform(self, texts):
        docFrequency = dict()

        for text in texts:
            words = tokenize(text, self.separator)

            for word in words:
                indexList = list()
                for index, wordList in self.corpus.items():
                    if word in wordList:
                        indexList.append(index)

                docFrequency[word] = indexList

        for word, items in docFrequency.items():
            self.idf[word] = math.log10(self.noOfDocuments / len(items))

        finalVal = dict()
        for index, wordDict in self.tf.items():
            final = dict()
            for word, tf in wordDict.items():
                final[word] = tf * self.idf.get(word, 0.0)

            finalVal[index] = final

        return self.addPadding([list(value.values()) for _, value in finalVal.items()])

    def addPadding(self, array):
        sparse = list()
        for value in array:
            if len(value) < self.maxFeatures:
                value.extend([0.] * (self.maxFeatures - len(value)))
            elif len(value) > self.maxFeatures:
                value = value[0:self.maxFeatures]

            sparse.append(value)

        return sparse


# tfidvectorizer = TfidfVectorizer()
# data = ["This is a a sample", "This is another example example another example",
#         "Hi how are you", "I am fina what about you"]
# tfidvectorizer.fit(data)
# data = tfidvectorizer.transform(data)
# import numpy as np
#
# print(np.asarray(data))
