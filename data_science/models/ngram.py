import nltk

class NGramModel:
    def __init__(self, n=2, left_pad_symbol="<s>", right_pad_symbol="</s>"):
        self.n = n
        self.model = None
        self.ngram_fn = lambda x : nltk.ngrams(
            x,
            n=self.n,
            pad_left=True,
            pad_right=True,
            left_pad_symbol=left_pad_symbol,
            right_pad_symbol=right_pad_symbol
        )

    def train(self, corpus):
        from nltk.probability import ConditionalFreqDist, MLEProbDist
    
        cfdist = ConditionalFreqDist()
    
        for doc in corpus:
            ngrams = self.ngram_fn(doc)
            # Convert ngrams to list of tuples for processing
            
            for ngram in ngrams:
                context = ngram[:-1]  # all but last word
                word = ngram[-1]      # last word
                cfdist[context][word] += 1
        
        self.model = MLEProbDist(cfdist)

    def generate(self, context):
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        return self.model.generate(context)

    def score(self, text):
        if not self.model:
            raise ValueError("Model has not been trained yet.")
        return self.model.prob(text)