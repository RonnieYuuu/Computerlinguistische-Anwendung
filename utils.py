import random
from nltk import word_tokenize

def dot(dictA, dictB):
    vec_product = 0
    for i in range(0, len(dictA)):
        if list(dictA.keys())[i] in dictB.keys():
            vec_product += dictA[list(dictA.keys())[i]] * dictB[list(dictA.keys())[i]]
        else:
            vec_product = vec_product
    return vec_product # TODO: Ex. 2: return vector product between features vectors represented by dictA and dictB.

def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]

class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = dict()
        # TODO: Ex. 3: create a dictionary that contains for each feature in the list the count how often it occurs.
        for feature in feature_list:
            if feature in feature_counts.keys():
                feature_counts[feature] += 1
            else:
                feature_counts[feature] = 1
        return cls(feature_counts, label)

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)

class Dataset:
    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])


    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in most instances)."""
        feature_instance_count = {}

        for inst in self.instance_list:
            for feature in inst.feature_counts.keys():
                if feature in feature_instance_count:
                    feature_instance_count[feature] += 1
                else:
                    feature_instance_count[feature] = 1
        sorted_features = sorted(feature_instance_count.items(), key=lambda item: item[1], reverse=True)
        top_n = [feature for feature, count in sorted_features[:n]]
        return set(top_n) # TODO: Ex. 4: Return set of n features that occur in most instances.

    def set_feature_set(self, feature_set):
        """
        This restrics the self.feature_set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the self.feature set."""
        # TODO: Ex. 5: Filter features according to feature set.
        for inst in self.instance_list:
            inst.feature_counts = {f: count for f, count in inst.feature_counts.items() if f in feature_set}
            self.feature_set = feature_set


    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent label for all instances in the dataset. """
        most_frequent_feature = list(self.get_topn_features(1))[0]
        correct_predictions = 0

        for inst in self.instance_list:
            if most_frequent_feature in inst.feature_counts:
                correct_predictions += 1

        total_instances = len(self.instance_list)

        return correct_predictions / total_instances # TODO: Ex. 6: Return accuracy of always predicting most frequent label in data set.


    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
