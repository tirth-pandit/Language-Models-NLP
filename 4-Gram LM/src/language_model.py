import re
import random
import numpy as np
import sys


class tokenization:

	def __init__(self , filename):

		toks , c = self.preproceesing( filename )
		self.tokens = toks 
		self.count = c 


	def tokenize(self , file):
		count = 0

		f = open(file)
		lines = f.readlines()

		tokens = []

		for i in lines:
			temp ,c = self.get_tokens(i)

			count += c

			tokens.append(temp)

		return tokens , count 


	def get_tokens(self ,statement):
	    fs = self.remove_space_from_quatation(statement)
	    ans = re.findall(r'[\'\w]+', fs)
	    
	    final_tokens = []
	    
	    for i in ans:
	        final_tokens.append( i.lower())
	        
	    
	    c = len(set(final_tokens))

	    return final_tokens ,c

	def remove_space_from_quatation(self ,string):
	    temp = string.split()
	    
	    final_str = ''
	    
	    for i in temp:
	        
	        if i=='’':
	            t = final_str[:len(final_str)-1] + "'"
	            final_str = t
	        else:
	            final_str += i
	            final_str += ' '
	            
	    return final_str


	def add_padding(self ,tokens):
	    start_token = '<s>'
	    end_token = '<e>'

	    final_tokens = []

	    for i in tokens:
	        temp = [start_token] + i + [end_token]
	        final_tokens.append(temp)

	    return final_tokens



	def preproceesing( self ,filename ):
	    tokens , total_tokens = self.tokenize(filename)
	    
	    padded_tokens = self.add_padding(tokens)
	    
	    return padded_tokens ,total_tokens



class grams:

	def __init__(self , tokens ):
		self.n_grams = [ None ,self.unigram(tokens) , self.bigram(tokens) , self.trigram(tokens) , self.fourgram(tokens) ]
		self.freq = self.calc_freq( self.n_grams )
	
	def unigram(self ,tokens):
	    grams = []
	    
	    for stmt in tokens:
	        for i in stmt:
	            temp = tuple([i])
	            grams.append(temp)
	    
	    return grams

	def bigram(self ,tokens):
	    
	    gram = []
	    
	    for line in tokens:
	        for i in range( len(line) - 1):
	            temp = tuple( [line[i] ,line[i+1]])
	            gram.append( temp)
	            
	    return gram 


	def trigram(self ,tokens):
	    
	    gram = []
	    
	    for line in tokens:
	        for i in range( len(line) - 2):
	            temp = tuple( [line[i] ,line[i+1],line[i+2]])
	            gram.append(temp)
	            
	    return gram


	def fourgram(self ,tokens):
	    gram = []
	    
	    for line in tokens:
	        for i in range( len(line) - 3):
	            temp = tuple( [line[i] ,line[i+1] , line[i+2] ,line[i+3] ])
	            gram.append(temp)
	    
	    return gram

	def calc_freq( self , n_grams ):
	    freq = [ None , {} ,{} ,{} ,{} ]
	    
	    for n in range(1,5):
	        gram = n_grams[n]
	        
	        for i in gram:
	            if i not in freq[n]:
	                freq[n][i] = 1
	            else:
	                freq[n][i] += 1
	                
	    return freq


class kn:

	def __init__(self , stmt ,freq ,total_tokens):
		
		probability = 1
		for i in stmt:
			cur_word = i[3]
			prev_words = list(i[0:3])

			temp = self.kn_recur(cur_word, prev_words ,True ,freq ,total_tokens)
			print(temp)
			
			if temp == 0 :
				continue

			probability *= temp

		self.prob = probability	

	def kn_recur( self , w , prev , top , freq , total_tokens):

		if len(prev) == 0:

			if top == True:
				count = 0

				if w in freq[1]:
					count = freq[1][word]
				else:
					count = 1

				return ( count / total_tokens )

			else:
				num = 0

				for i in freq[2]:
					if i[1] == w:
						num+=1

				return num / len( freq[2] )

		d = 0.7
		words = prev + [w]

		all_gram = len(words)
		prev_gram = len(prev)

		key = tuple(words)
		num1 = 0

		if top == True:
			if key in freq[all_gram]:
				num1 = max( freq[all_gram][key] - d , 0 )
		else:
			temp = words[1:]

			for i in freq[all_gram]:
				t1 = list(i)
				t2 = t1[ 1:len(t1)]
				t3 = tuple(t2)

				if t3 == temp:
					num1 += 1

		key = tuple(prev)
		deno = 0

		if top == True:
			if key in freq[prev_gram]:
				deno = freq[prev_gram][key]
			else:
				deno = 1
		else:
			deno = len(freq[all_gram])

		num2 = 0
		for i in freq[all_gram]:
			t1 = list(i)
			t2 = t1[0:len(t1)-1]
			t3 =  tuple(t2)

			if t3 == tuple(prev):
				num2 += 1

		if num1==0 and num2==0:
			ans = random.uniform(0.1, 0.2) + self.kn_recur(w ,prev[1:] ,False ,freq ,total_tokens)
			return ans
		
		if num2 == 0:
			return ( num1 / deno )

		return (num1/deno) + d*(num2/deno)*self.kn_recur(w , prev[1:] ,False ,freq ,total_tokens)


class wb:

	def __init__(self ,stmt ,freq) :

		probability = 1
		for i in stmt:
        	
			temp = self.witten_bell(i ,freq)
			print(temp)
			
			if temp == 0:
				continue

			probability *= temp

		self.prob = probability	

	def val( self,gram ,freq):
		ans = 0

		if len(gram) == 0 :
			for i in freq[1].values():
				ans += i
		elif gram in freq[ len(gram)].keys():
			ans = freq[len(gram)][gram]
		else:
			ans = 0
		return ans 

	def lambdas( self,gram ,freq):
		count = 0

		for i in freq[ len(gram)+1 ].keys():
			if i[:-1] == gram:
				count += 1

		deno = self.val(gram ,freq)

		deno = deno + count

		if deno != 0:
			lam = count/float(deno)
			return lam

		return random.uniform(0.1, 0.2)


	def witten_bell( self,words,freq):
	    
	    if len(words) == 1 :
	        gram = tuple([words])
	        num = self.val(gram ,freq)
	        
	        deno = 0
	        for i in freq[1].values():
	            deno += i
	        
	        return num/float(deno)
	    
	    gram = tuple(words)
	    
	    num = self.val( gram ,freq)
	    deno = self.val( gram[:-1] ,freq)
	    
	    if deno != 0 :
	        
	        ans1 = ( 1-self.lambdas(gram[:-1] , freq) )
	        ans2 =  self.lambdas(gram[:-1] ,freq ) * self.witten_bell(gram[1:] ,freq)
	        
	        ans = ans1*(num/deno) + ans2
	        
	        return ans 
	    
	    else:
	        ans = random.uniform(0.1, 0.2)  +(self.lambdas(gram[:-1] ,freq) * self.witten_bell(gram[1:] ,freq))
	        return ans


def test_preprocess(stmt):

	fs = remove_space_from_quatation(stmt)
	ans = re.findall(r'[\'\w]+', fs)

	final_tokens = []

	for i in ans:
		final_tokens.append( i.lower())

	c = len(set(final_tokens))

	final_tokens = add_padding(final_tokens)
	return final_tokens


def remove_space_from_quatation(string):
	temp = string.split()
	final_str = ''

	for i in temp:
		if i=='’':
			t = final_str[:len(final_str)-1] + "'"
			final_str = t
		else:
			final_str += i
			final_str += ' '

	return final_str

def add_padding(tokens):
    start_token = '<s>'
    end_token = '<e>'
    

    return [start_token] + tokens + [end_token]


def fourgram(tokens):
    gram = []
    
    for line in tokens:
        for i in range( len(line) - 3):
            temp = tuple( [line[i] ,line[i+1] , line[i+2] ,line[i+3] ])
            gram.append(temp)
    
    return gram

if __name__ == "__main__":

	smoothing = sys.argv[1]
	corpus = sys.argv[2]

	test = input('Input sentence : ')
	token = tokenization(corpus)

	gram_class = grams(token.tokens)

	final_tokens = test_preprocess(test)

	stmt_gram = fourgram([final_tokens])
	print(stmt_gram)
	
	prob = None

	if smoothing == 'k':
		kn_class = kn(stmt_gram , gram_class.freq ,token.count)
		prob = kn_class.prob
	else:
		wb_class = wb(stmt_gram , gram_class.freq)
		prob = wb_class.prob

	print(prob)
