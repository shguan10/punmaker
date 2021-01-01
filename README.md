# Pun Maker
Given a phrase, the pun maker will mutate it to produce a close-sounding phrase with different words. 

After translating the phrase into IPA (International Phonetic Alphabet), it will concatenate all chars into one giant string. Then it randomly partitions the string into separate strings. Next it uses fuzzy string matching to retrieve a close-sounding real word from the American dictionary for each partition.

## Dependencies
1. [eng-to-ipa-to-eng](https://github.com/shguan10/eng_to_ipa_to_eng)
2. edlib
3. numpy

## Specifics for the fuzzy string matching
The encoding will basically follow the same one in [this paper](https://arxiv.org/pdf/1803.02893.pdf). Given a query string, the algorithm will retrieve the word in the dictionary with the highest cosine similarity of their encodings.

Another approach is to treat each word as a bag of ngrams, and use the tf-idf vector as the word's encoding.

### Generating the dataset
1. It will assume the strings are already in IPA, not the Latin alphabet.
2. For some word w, randomly sample a word k with edit distance <= 5 to w. Also randomly sample words with edit distance > 5 to w. These sampled words will be the set of candidate words for the word w.

#### TODO
1. the tf-idf encoder works decently well. the gru encoding does not work. need to figure out why

