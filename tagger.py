import os
import io
import sys
import math

class Tagger:
    def __init__(self):
        """ Initialize class variables here """
        self.word_tags = []
        self.tags = ()
        self.tags_count = {}
        self.initial_tag_prob = {}
        self.transition_count = []
        self.transition_prob = []
        self.final_transition_prob = []
        self.emission_prob = []
        self.tokens = ()
        self.tokens_idx_lookup = {}
        self.tags_idx_lookup = {}
        self.viterbi = []
        self.backpointer = []
        self.likely_tags = []


    def load_corpus(self, path):
        """
        Returns all sentences as a sequence of (word, tag) pairs found in all
        files from as directory
        `path`.

        Inputs:
            path (str): name of directory
        Outputs:
            word_tags: 2d-list that represent sentences in the corpus. Each
            sentence is then represented as a list of tuples (word, tag)
        """
        if not os.path.isdir(path):
            sys.exit("Input path is not a directory")

        for filename in os.listdir(path):
            # Iterates over files in directory
            """ YOUR CODE HERE """
            f = open(os.path.join(path, filename), 'r')
            for line in f.readlines():
                token_pos = line.strip().split()
                word_tag = []
                for tp in range(len(token_pos)):
                    t = token_pos[tp].split("/")
                    t[0] = t[0].lower()
                    word_tag.append(tuple(t))
                if len(word_tag) == 0:
                    continue
                self.word_tags.append(word_tag)

        return self.word_tags


    def initialize_probabilities(self, sentences):
        """
        Initializes the initial, transition and emission probabilities into
        class variables

        Inputs:
            sentences: a 2d-list of sentences, usually the output of
            load_corpus
        Outputs:
            None
        """
        if type(sentences) != list:
            sys.exit("Incorrect input to method")
        if len(sentences) != 0:
            self.word_tags = sentences
        """ 1. Compute Initial Tag Probabilities """
        tags = []
        tok = []
        for i in range(len(self.word_tags)):
            for j in range(len(self.word_tags[i])):
                tags.append(self.word_tags[i][j][1])
                tok.append(self.word_tags[i][j][0])

        self.tags = tuple(sorted(set(tags)))
        self.tokens = tuple(set(tok))
        for i in range(len(self.tokens)):
            self.tokens_idx_lookup[self.tokens[i]] = i
        for i in range(len(self.tags)):
            self.tags_idx_lookup[self.tags[i]] = i
        for i in range(len(self.tags)):
            count = 0
            for j in range(len(self.word_tags)):
                if self.tags[i] == self.word_tags[j][0][1]:
                    count+=1
            self.initial_tag_prob[self.tags[i]] = count/len(self.word_tags)
        """ 2. Compute Transition Probabilities """
        self.transition_prob = [[0]*len(self.tags) for i in range(len(self.tags))]
        for i in range(len(self.tags)):
            self.tags_count[self.tags[i]] = 0
        for i in range(len(self.word_tags)):
            for j in range(len(self.word_tags[i])):
                self.tags_count[self.word_tags[i][j][1]] += 1
        for i in range(len(self.word_tags)):
            for j in range(1, len(self.word_tags[i])):
                r = self.tags_idx_lookup[self.word_tags[i][j-1][1]]
                c = self.tags_idx_lookup[self.word_tags[i][j][1]]
                self.transition_prob[r][c] +=1
        for i in range(len(self.transition_prob)):
            for j in range(len(self.transition_prob)):
                self.transition_prob[i][j] = self.transition_prob[i][j]/self.tags_count[self.tags[i]]


        self.final_transition_prob = [0]*len(self.tags)

        for i in range(len(self.word_tags)):
            last_tag = self.word_tags[i][len(self.word_tags[i])-1][1]
            self.final_transition_prob[self.tags_idx_lookup[last_tag]] += 1
        for i in range(len(self.final_transition_prob)):
            self.final_transition_prob[i] = self.final_transition_prob[i]/len(self.word_tags)


        """ 3. Compute Emission Probabilities """
        self.emission_prob = [[0]*len(self.tokens) for i in range(len(self.tags))]
        for i in range(len(self.word_tags)):
            for j in range(len(self.word_tags[i])):
                token_tag_idx = self.word_tags[i][j]
                self.emission_prob[self.tags_idx_lookup[token_tag_idx[1]]][self.tokens_idx_lookup[token_tag_idx[0]]] +=1

        for i in range(len(self.emission_prob)):
            for j in range(len(self.emission_prob[i])):
                self.emission_prob[i][j] = self.emission_prob[i][j]/(self.tags_count[self.tags[i]])

        #Add one smoothing
        for i in range(len(self.emission_prob)):
            for j in range(len(self.emission_prob)):
                self.emission_prob[i][j] += 1
        return


    def viterbi_decode(self, sentence):
        """
        Implementation of the Viterbi algorithm

        Inputs:
            sentence (str): a sentence with N tokens, be those words or
            punctuation, in a given language
        Outputs:
            likely_tags (list[str]): a list of N tags that most likely match
            the words in the input sentence. The i'th tag corresponds to
            the i'th word.
        """

        if type(sentence) != str:
            sys.exit("Incorrect input to method")

        """ Tokenize sentence """
        sentence_tokens = sentence.lower().split()

        """ Implement the Viterbi algorithm """
        self.viterbi = [[0]*len(sentence_tokens) for i in range(len(self.tags)+1)]
        self.backpointer = [[0] * len(sentence_tokens) for i in range(len(self.tags) + 1)]

        for i in range(len(self.tags)):
            try:
                self.viterbi[i][0] = self.initial_tag_prob[self.tags[i]] * self.emission_prob[i][self.tokens_idx_lookup[sentence_tokens[0]]]
            except:
                self.viterbi[i][0] = 0
            self.backpointer[i][0] = 0

        for i in range(1, len(sentence_tokens)):
            for j in range(len(self.tags)):
                temp = []
                for k in range(len(self.tags)):
                    try:
                        temp.append(self.viterbi[k][i-1]*self.transition_prob[k][j]*self.emission_prob[j][self.tokens_idx_lookup[sentence_tokens[i]]])
                    except:
                        temp.append(self.viterbi[k][i-1]*self.transition_prob[k][j]*0.00001)
                comp = list(enumerate(temp))
                #print(comp)
                argmax, maximum = max(comp, key=lambda x: x[1])
                self.viterbi[j][i] = maximum
                self.backpointer[j][i] = argmax
        tmp = []
        for i in range(len(self.tags)):
            tmp.append(self.viterbi[i][len(sentence_tokens)-1]*self.final_transition_prob[self.tags_idx_lookup[self.tags[i]]])
        argmax2, maximum2 = max(list(enumerate(tmp)), key=lambda x:x[1])
        self.viterbi[len(self.tags)][len(sentence_tokens)-1] = maximum2
        self.backpointer[len(self.tags)][len(sentence_tokens)-1] = argmax2

        self.likely_tags = []
        for i in range(1, len(sentence_tokens)):
            count = 0
            for j in range(len(self.backpointer)-1):
                if self.backpointer[j][i] != 0:
                    self.likely_tags.append(self.tags[self.backpointer[j][i]])
                    break
                count += 1
            if count == len(self.tags):
                self.likely_tags.append(self.tags[0])

        self.likely_tags.append(self.tags[self.backpointer[len(self.tags)][len(sentence_tokens)-1]])
        print(self.likely_tags)
        return self.likely_tags





#path = os.path.join(os.getcwd(),"train\modified_brown")
#p = sys.argv[1]
path = sys.argv[1]
tagger = Tagger()
tagger.load_corpus(path)
tagger.initialize_probabilities(tagger.word_tags)
tagger.viterbi_decode("the planet jupiter and its moons are in effect a mini solar system .")
