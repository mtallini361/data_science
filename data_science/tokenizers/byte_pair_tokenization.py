from collections import Counter

def byte_pair_encoding(corpus, num_merges):
    # Initialize the vocabulary with unique characters
    vocab = set(''.join(corpus))
    
    # Convert corpus into list of tokens (as lists of characters)
    corpus_tokens = [list(word) for word in corpus]

    for _ in range(num_merges):
        # Count pairs of adjacent tokens across all corpus entries
        pairs = Counter()
        for tokens in corpus_tokens:
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1

        if not pairs:
            break  # No more pairs to merge

        # Find the most frequent pair
        most_common_pair = max(pairs, key=pairs.get)
        tL, tR = most_common_pair
        tNEW = tL + tR

        # Update vocabulary
        vocab.add(tNEW)

        # Replace occurrences of the pair with the new token
        new_corpus_tokens = []
        for tokens in corpus_tokens:
            new_tokens = []
            skip = False
            i = 0
            while i < len(tokens):
                if skip:
                    skip = False
                    i += 1
                    continue
                if i < len(tokens) - 1 and tokens[i] == tL and tokens[i + 1] == tR:
                    new_tokens.append(tNEW)
                    skip = True  # skip next token because it's part of merged pair
                else:
                    new_tokens.append(tokens[i])
                i += 1
            new_corpus_tokens.append(new_tokens)
        corpus_tokens = new_corpus_tokens

    return vocab