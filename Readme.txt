Directory Structure:
|--tagger.py
|--train
	|--modified_brown
		|--ca01
		.
		.
		.
		


How to run the code:
If code is executed directly, please follow the below steps
	1. Make sure that the training files are extracted and present in the location /train/modified_brown
	2. Paste the test sentence inside the tagger.viterbi_decode function in Line 233 - tagger.viterbi_decode("")
	3. Run tagger.py. Command: python3 tagger.py /home/011/a/ax/axm200011/nlp_hw2/train/modified_brown
	
If the code is imported as package, please follow the steps below:
1. Comment the following lines:	
	Line 229: path = os.path.join(os.getcwd(),"train\modified_brown") 
	Line 230: tagger = Tagger()
	Line 231: tagger.load_corpus(path)
	Line 232: tagger.initialize_probabilities(tagger.word_tags)
	Line 233: tagger.viterbi_decode("computers process programs accurately .")
2. Import the class and use the appropriate funtions to run POS tagger.

