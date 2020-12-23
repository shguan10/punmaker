# Pun Maker
Given a phrase, the pun maker will mutate it to produce a close-sounding phrase with different words. 

After translating the phrase into IPA (International Phonetic Alphabet), it will concatenate all chars into one giant string. Then it randomly partitions the string into separate strings. Next it uses fuzzy string matching to retrieve a close-sounding real word from the American dictionary for each partition.

## Specifics for the fuzzy string matching
The encoding will basically follow the same one in [this paper](https://arxiv.org/pdf/1803.02893.pdf). Given a query string, the algorithm will retrieve the word in the dictionary with the highest cosine similarity of their encodings.

### Generating the dataset
1. It will assume the strings are already in IPA, not the Latin alphabet.
2. For some word w, randomly sample a word k with edit distance <= 5 to w. Also randomly sample words with edit distance > 5 to w. These sampled words will be the set of candidate words for the word w.

test