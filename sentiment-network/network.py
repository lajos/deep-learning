#!/usr/bin/env python3

import time
import sys
import numpy as np
import pickle
from collections import Counter

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1, min_count=20, polarity_cutoff=0.05):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """
        # Assign a seed to our random number generator to ensure we get
        # reproducable results during development
        np.random.seed(1)

        self.min_count = min_count
        self.polarity_cutoff = polarity_cutoff

        # process the reviews and their associated labels so that everything
        # is ready for training
        self.pre_process_data(reviews, labels)

        # Build the network to have the number of hidden nodes and the learning rate that
        # were passed into this initializer. Make the same number of input nodes as
        # there are vocabulary words and create a single output node.
        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)


    def pre_process_data(self, reviews, labels):

        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for review, label in zip(reviews, labels):
            words = review.split(' ')
            for w in words:
                if label=='POSITIVE':
                    positive_counts[w] += 1
                else:
                    negative_counts[w] += 1
                total_counts[w] +=1

        pos_neg_ratios = Counter()

        # Calculate the ratios of positive and negative uses of the most common words
        #       Consider words to be "common" if they've been used at least 50 times

        for word,n in total_counts.most_common():
            if n > 50:
                pos_neg_ratios[word] = positive_counts[word] / float(negative_counts[word]+1)

        for word,ratio in pos_neg_ratios.most_common():
            pos_neg_ratios[word] = np.log(pos_neg_ratios[word])

        review_vocab = set()

        # try:
        #     with open('review_vocab.dat','rb') as f:
        #         review_vocab = pickle.load(f)
        #     print('.using review_vocab.dat')
        # except:
        #     for review in reviews:
        #         for word in review.split(' '):
        #             review_vocab.add(word)
        #     with open('review_vocab.dat','wb') as f:
        #         pickle.dump(review_vocab, f)
        #     print('.saved review_vocab.dat')

        # this used to add all words
        # for review in reviews:
        #     for word in review.split(' '):
        #         review_vocab.add(word)

        for word,ratio in pos_neg_ratios.most_common():
            if abs(ratio)>self.polarity_cutoff and total_counts[word]>self.min_count:
                review_vocab.add(word)

        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = ['NEGATIVE','POSITIVE']

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i,word in enumerate(review_vocab):
            self.word2index[word]=i

        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i,label in enumerate(self.label_vocab):
            self.label2index[label]=i

        print(self.review_vocab[:10])
        print(len(self.review_vocab))
        # print(self.label_vocab)
        # print(self.label2index)



    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights

        # initialize self.weights_0_1 as a matrix of zeros. These are the weights between
        #       the input layer and the hidden layer.
        # self.weights_0_1 = np.random.normal(size=(self.input_nodes,self.hidden_nodes))
        self.weights_0_1 = np.zeros(shape=(self.input_nodes,self.hidden_nodes))

        # initialize self.weights_1_2 as a matrix of random values.
        #       These are the weights between the hidden layer and the output layer.
        # self.weights_1_2 = np.zeros(shape=(self.hidden_nodes, self.output_nodes))
        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))
        # self.weights_1_2 = np.random.normal(0.0, 1, (self.hidden_nodes, self.output_nodes))

        # Create the input layer, a two-dimensional matrix with shape
        #       1 x input_nodes, with all values initialized to zero
        # self.layer_0 = np.zeros((1,input_nodes))

        self.layer_1 = np.zeros((1,hidden_nodes))


    def update_input_layer(self,review):
        self.layer_0 *= 0
        for w in review.split(' '):
            self.layer_0[0][self.word2index[w]] = 1

    def get_target_for_label(self,label):
        return self.label2index[label]

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self,output):
        return output*(1-output)


    def train(self, training_reviews_raw, training_labels):
        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        # make sure out we have a matching number of reviews and labels
        assert(len(training_reviews) == len(training_labels))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            # Get the next review and its correct label
            review = training_reviews[i]
            label = training_labels[i]

            #self.update_input_layer(review)
            target = self.get_target_for_label(label)

            # forward pass

            #layer_1_in = np.dot(self.layer_0, self.weights_0_1)

            self.layer_1*=0
            for w in review:
                self.layer_1 += self.weights_0_1[w]

            layer_1_in = self.layer_1
            layer_1_out = layer_1_in		# don't use activation function

            layer_2_in = np.dot(layer_1_out, self.weights_1_2)
            layer_2_out = self.sigmoid(layer_2_in)

            output = layer_2_out

            error = target - output

            error_2 = error
            error_term_2 = error_2 * self.sigmoid_output_2_derivative(layer_2_out)


            error_1 = np.dot(error_term_2, self.weights_1_2.T)
            error_term_1 = error_1		# no activation function for layer_1 (derivative 1*)

            # backprop

            # print('weights')
            # print(self.weights_0_1.shape)
            # print(self.weights_1_2.shape)

            # print('error')
            # print(error_1.shape)
            # print(error_2.shape)

            # print('error term')
            # print(error_term_1.shape)
            # print(error_term_2.shape)

            # print('layer outputs')
            # print(self.layer_0.shape)
            # print(layer_1_out.shape)
            # print(layer_2_out.shape)

            # d_weights_0_1 = np.dot(self.layer_0.T, error_term_1)
            d_weights_1_2 = np.dot(layer_1_out.T, error_term_2)

            # self.weights_0_1 += self.learning_rate * d_weights_0_1
            self.weights_1_2 += self.learning_rate * d_weights_1_2

            for index in review:
                self.weights_0_1[index] += error_term_1[0] * self.learning_rate # update input-to-hidden weights with gradient descent step

            if abs(error)<0.5:
                correct_so_far += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")


    def test(self, testing_reviews, testing_labels):
        """
        Attempts to predict the labels for the given testing_reviews,
        and uses the test_labels to calculate the accuracy of those predictions.
        """

        # keep track of how many correct predictions we make
        correct = 0

        # we'll time how many predictions per second we make
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1

            # For debug purposes, print out our prediction accuracy and speed
            # throughout the prediction process.

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

    def run(self, review):
        """
        Returns a POSITIVE or NEGATIVE prediction for the given review.
        """
        # Run a forward pass through the network, like in the "train" function.

        ## New for Project 5: Removed call to update_input_layer function
        #                     because layer_0 is no longer used

        # Hidden layer
        ## New for Project 5: Identify the indices used in the review and then add
        #                     just those weights to layer_1
        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]

        # Output layer
        ## New for Project 5: changed to use self.layer_1 instead of local layer_1
        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # return NEGATIVE for other values
        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"

def main():
    with open('reviews.txt','r') as f:
        reviews = list(map(lambda x:x[:-1],f.readlines()))

    with open('labels.txt','r') as f:
        labels = list(map(lambda x:x[:-1].upper(),f.readlines()))

#	mlp = SentimentNetwork(reviews, labels, hidden_nodes=10, learning_rate=0.1)

# 	mlp = SentimentNetwork(reviews,labels, learning_rate=0.01)
# 	#mlp.train(reviews[:10],labels[:10])
# 	mlp.train(reviews[:1000],labels[:1000])
# #	mlp.train(reviews[:-1000],labels[:-1000])
# 	#mlp.test(reviews[-1000:],labels[-1000:])
# 	mlp.test(reviews[-100:],labels[-100:])

    # mlp = SentimentNetwork(reviews[:-1000],labels[:-1000], learning_rate=0.001)
    # mlp.train(reviews[:-1000],labels[:-1000])

    mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=30,polarity_cutoff=0.7,learning_rate=0.01)
    mlp.train(reviews[:-1000],labels[:-1000])
    mlp.train(reviews[:-1000],labels[:-1000])
    mlp.train(reviews[:-1000],labels[:-1000])
    mlp.train(reviews[:-1000],labels[:-1000])
    mlp.test(reviews[-1000:],labels[-1000:])

if __name__=='__main__':
    main()


