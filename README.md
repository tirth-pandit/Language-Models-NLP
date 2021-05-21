# Language Models 
### 1]   4-gram Language Models 
	4-gram Language models on both given corpus. Used 2 types of Smoothing Techniques 
	
		1) Kneser - Ney 
		2) Witten - bell

	Running The Program :
		$ python language_model.py  -smoothing  -corpus_path  -perplexity
		
	    a) smoothing :
			k - kniser ney
			w - witten bell

	    b) corpus_path :
			path to the text file of corpus  

		c) perplexity:
			yes - if want to print perplexity 
			no - if do not want to print preplexity
		 

	Example :
		$ python language_model.py k ../Health_English.txt yes
<br>

### 2] Neural Language Model 
		LSTM Base Neural Model 
