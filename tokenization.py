import sys
import pprint
# Tokenization :(

# Tokenization is at the heart of much weirdess of LLMs. Do not brush it off.
# Why can't LLM do super simple string processing tasks like reversing a string? Tokenization.
# Why is LLM worse at non-English languages (e.g. Japanase) Tokenization.
# Why is LLM bad at simple arithmetic? Tokenization.
# Why did GPT-2 have more than necessary trouble coding in Python? Tokenization.
# Why did my LLM abruptly halt when it sees the string '<|endoftext|>'? Tokenization.
# Why is this weird warning i get about a "trailing whitespace"? Tokenization.
# Why the LLM break if i ask it about "SolidGoldMagikarp"? Tokenization.
# Why sould i prefer to use YAML over JSON with LLMs? Tokenization.
# Why is LLM not actually end-to-end language modeling? Tokenization.
# What is the real root of suffering? Tokenization.

# Good Tokenization web app: https://tiktokenizer.vercerl.app

# Example string:

    # Tokenization is at the heart of much weirdess of LLMs. Do not brush it off.

    # 127 + 677 = 804
    # 1275 + 6773 = 8041
    
    # Egg.
    # I have a Egg.
    # egg.
    # EGG.

    # for i in range(1, 101):
    #       if i % 3 == 0 and i % 5 == 0:
    #           print("FizzBuzz")
    #       elif i % 3 == 0:
    #           print("Fizz")
    #       elif i % 5 == 0:
    #           print("Buzz")
    #       else:
    #           print(i)

# salu2 = "안녕하세요 👋 (Hello in korean!) ḫ a"
# print(f"안 es {ord("안")}")
# print("lista",[ord(x) for x in salu2])
# print('utf-8',salu2.encode("utf-8"))
# print('***'*10)
# print('list : utf-8',list(salu2.encode("utf-8")))
# print('***'*10)
# print('utf-16',salu2.encode("utf-16"))
# print('***'*10)
# print('list : utf-16',list(salu2.encode("utf-16")))
# print('***'*10)
# print('utf-32',salu2.encode("utf-32"))
# print('***'*10)
# print('list : utf-32',list(salu2.encode("utf-32")))

# text from https://www.reedbeta.com/blog/programmers-intro-to-unicode/
text = 'Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.'
tokens = text.encode('utf-8')
imprimir = sys.stdout
print('Bytes: ',tokens)
tokens = list(map(int, tokens)) # convert to a list of integers in range 0...255 for convenience
print('-----------')
print(text)
print('length:', len(text))
print('-----------')
print(tokens)
print('length:', len(tokens))

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

stats = get_stats(tokens)
# pprint.pprint(stats, indent= 4 ,width=500)
# pprint.pprint((sorted(((v,k) for k, v in stats.items()), reverse = True)))
print('Máximo: ',max(stats, key = stats.get))
top_pair = max(stats, key = stats.get)

def merge(ids, pair, idx):
    # in the list of ints (ids), replace all consecutive occurences of pair with the new token idx
    newids = []
    i = 0
    while i < len(ids):
        # if we are not at the very last position AND the pair matches, replace it
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    
    return newids

print(merge([5,6,6,7,9,1], (6,7), 99))

# print('length tokens:', len(tokens))
# tokens2 = merge(tokens, top_pair, 256)
# print(tokens2)
# print('length tokens2: ', len(tokens2))

vocab_size = 276
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) --> int
for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key = stats.get)
    idx = 256 + i
    print(f'merging {pair} into a new token {idx}')
    ids = merge(ids, pair, idx)
    merges[pair] = idx 


print('tokens length:', len(tokens))
print('ids length: ', len(ids))
print(f'compression ratio: {len(tokens) / len(ids):.2f}X')

# Note, the Tokenizer is a completely separate, independent module from the LLM. It has
# its own training dataset of text (which could be different from that of the LLM), on
# which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. It then
# translates back and forth between raw text and sequences of tokens. The LLM later only
# ever sees the tokens and never directly deals with any text.

# DECODING
# Given a sequence of integers in the range [0, vocab_size], what is the text?
vocab = {idx: bytes([idx]) for idx in range(256)}
# print('Vocab before:', vocab)
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

pprint.pprint(vocab)
def decode(ids):
    # given ids (list of integers), return Python 
    tokens = b''.join(vocab[idx] for idx in ids)
    token = tokens.decode('utf-8', errors='replace')
    return token

print('Decode: ',decode([130]))

def encode(text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode('utf-8'))
    while len(text) >= 2: # to ensure that with get more than 1 characters
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float('inf')))
        if pair not in merges:
            break # nothing else can be merged 

        idx = merges[pair]           
        tokens = merge(tokens, pair, idx)
    return tokens

print(decode(encode('hello world!')))

print(decode(encode(text)))
text2 = decode(encode(text))        
print('Text es igual a text2?!: ',text == text2)