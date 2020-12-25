# Pun Maker
Given a phrase, the pun maker will mutate it to produce a close-sounding phrase with different words. 

After translating the phrase into IPA (International Phonetic Alphabet), it will concatenate all chars into one giant string. Then it randomly partitions the string into separate strings. Next it uses fuzzy string matching to retrieve a close-sounding real word from the American dictionary for each partition.

## Dependencies
1. [eng-to-ipa-to-eng](https://github.com/shguan10/eng_to_ipa_to_eng)
2. edlib
3. numpy

## Specifics for the fuzzy string matching
The encoding will basically follow the same one in [this paper](https://arxiv.org/pdf/1803.02893.pdf). Given a query string, the algorithm will retrieve the word in the dictionary with the highest cosine similarity of their encodings.

### Generating the dataset
1. It will assume the strings are already in IPA, not the Latin alphabet.
2. For some word w, randomly sample a word k with edit distance <= 5 to w. Also randomly sample words with edit distance > 5 to w. These sampled words will be the set of candidate words for the word w.

#### TODO
1. Write the f and g encoders
2. prep the dataset
    - randomly sample 10,000 words. these will be our 9000 datapoints
    - for each datapoint, radonly sample 1 word with close edit distance, and 4 other words with far edit distance
    - split into train/val/test of 8:1:1
3. figure out how to quickly calculate edit distance
4. write function to quickly sample word with close edit distance