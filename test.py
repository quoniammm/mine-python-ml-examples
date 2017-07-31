import time
import sys
import numpy as np

class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):
        np.random.seed(1)

        self.pre_process_data(reviews, labels)

        self.init_network(len(self.review_vocab), hidden_nodes, 1, learning_rate)

    
    def pre_process_data(self, reviews, labels):

        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                review_vocab.add(word)
        
        self.review_vocab = list(review_vocab)

        label_vocab = set()
        for label in labels:
            label_vocab.add(label)

        self.label_vocab = list(label_vocab)

        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        self.word2index = {}
        for i, word in  enumerate(self.review_vocab):
            self.word2index[word] = i

        self.label2index = []
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.weights_0_1 = np.zeros((self.input_nodes, self.hidden_nodes))
        self.weights_1_2 = np.zeros((self.hidden_nodes, self.output_nodes))

        self.layer_0 = np.zeros((1, self.input_nodes))

    def update_input_layer(self, review):

        self.layer_0 *= 0

        for word in review.split(" "):
            
            if(word in self.word2index.keys()):
                self.layer_0[0][self.word2index[word]]

    def get_target_for_label(self, label):
        if(lablel == 'POSITIVE'):
            return 1
        else:
            return 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self, output):
        return output * (1 -　output)

    def train(self, training_reviews, training_labels):

        assert(len(training_labels) == len(training_reviews))

        correct_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):

            review = training_reviews[i]
            label = training_labels[i]

            self.update_input_layer(review)

            layer_1 = self.layer_0.dot(self.weights_0_1)

            layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))

            layer_2_error = layer_2 - self.get_target_for_label(label) # Output layer error is the difference between desired target and actual output.
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)

            # Backpropagated error
            layer_1_error = layer_2_delta.dot(self.weights_1_2.T) # errors propagated to the hidden layer
            layer_1_delta = layer_1_error # hidden layer gradients - no nonlinearity so it's the same as the error

            # Update the weights
            self.weights_1_2 -= layer_1.T.dot(layer_2_delta) * self.learning_rate # update hidden-to-output weights with gradient descent step
            self.weights_0_1 -= self.layer_0.T.dot(layer_1_delta) * self.learning_rate # update input-to-hidden weights with gradient descent step

            # Keep track of correct predictions.
            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
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
        
        # Input Layer
        self.update_input_layer(review.lower())

        # Hidden layer
        layer_1 = self.layer_0.dot(self.weights_0_1)

        # Output layer
        layer_2 = self.sigmoid(layer_1.dot(self.weights_1_2))
        
        # Return POSITIVE for values above greater-than-or-equal-to 0.5 in the output layer;
        # return NEGATIVE for other values
        if(layer_2[0] >= 0.5):
            return "POSITIVE"
        else:
            return "NEGATIVE"



        