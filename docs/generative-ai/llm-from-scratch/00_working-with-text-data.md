---
title: 'Working with Text Data'

parent: LLM From Scratch

nav_order: 1

tags:
  - CLIP
  - Transformers
  - Multimodal Model
  - Computer Vision
  - Machine Learnig
---

# Working with Text Data
{: .no_toc }

This blog post covers the steps involved in preparing text for training large language models, including:

- Splitting the text into word and subword tokens.
- Using byte pair encoding (BPE) for more advanced tokenization.
- Sampling training examples using a sliding window approach.
- Converting tokens into vectors to be fed into the large language model.

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>


## Tokenization

You can find the input raw text [here](https://en.wikisource.org/wiki/Brother_Leo).


```python
import os

with open('/input/Leo.txt', 'r', encoding="utf-8") as file:
  content = file.read()

print(f"Total number of characters present is {len(content)}")
```
```
Total number of characters present is 18036
```

```python
print(content[:150])
```
```
IT was a sunny morning, and I was on my way to Torcello. Venice lay behind us a dazzling line, with towers of gold against the blue lagoon. All at onc
```


```python
import re
text = "Hello, world. This is a sample text."
result = re.split(r'(\s)', text)
print(result)
```
```
['Hello,', ' ', 'world.', ' ', 'This', ' ', 'is', ' ', 'a', ' ', 'sample', ' ', 'text.']
```

This result is a list of individual words, whitespaces and punctuation characters.

Now, let's modify the regular expression that splits on whitespaces, commas and periods.


```python
result = re.split(r'([,.]|\s)', text)
print(result)
```

```
['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ' ', 'is', ' ', 'a', ' ', 'sample', ' ', 'text', '.', '']
```

```python
result = [item for item in result if item.strip()]
print(result)
```

```
['Hello', ',', 'world', '.', 'This', 'is', 'a', 'sample', 'text', '.']
```


```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
```

```
['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```

In the above example, the sample text gets splitted into 10 different tokens. Now, we will apply this tokenizer to our text file content.


```python
preprocessed_text =  re.split(r'([,.:;?_!"()\']|--|\s)', content)
preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
print(len(preprocessed_text))
```

```
4024
```

So,we have in total 4690 tokens in our corpus. Let's print first 30 tokens from this list.

```python
print(preprocessed_text[:30])
```

```
['IT',
'was',
'a',
'sunny',
'morning',
',',
'and',
'I',
'was',
'on',
'my',
'way',
'to',
'Torcello',
'.',
'Venice',
'lay',
'behind',
'us',
'a',
'dazzling',
'line',
',',
'with',
'towers',
'of',
'gold',
'against',
'the',
'blue']
```


We can clearly see from the output that we don't have any whitespaces and special characters as a token in this list. we successfully onverted the raw text into individual tokens.

## Converting tokens to token IDs

Let's create a list of unique tokens and sort them alphabetically to identify vocabulary size.


```python
all_unique_words = sorted(set(preprocessed_text))
vocab_size = len(all_unique_words)
print(vocab_size)
```

```
988
```



```python
vocab = {token: integer for integer, token in enumerate(all_unique_words)}

for i, item in enumerate(vocab.items()):
  print(item)
  if i >= 50:
    break
```

```
('!', 0)
('"', 1)
("'", 2)
(',', 3)
('.', 4)
(':', 5)
(';', 6)
('?', 7)
('A', 8)
('After', 9)
('Ah', 10)
('All', 11)
('Altinum', 12)
('And', 13)
('As', 14)
('At', 15)
('Besides', 16)
('Brother', 17)
('Burano', 18)
('But', 19)
('Deserto', 20)
('English', 21)
('Enter', 22)
('Esau', 23)
('Even', 24)
('Excellency', 25)
('Far', 26)
('First', 27)
('Francesco', 28)
('Francis', 29)
('French', 30)
('God', 31)
('He', 32)
('Here', 33)
('His', 34)
('I', 35)
('IT', 36)
('If', 37)
('Indeed', 38)
('It', 39)
('Leo', 40)
('Lorenzo', 41)
('May', 42)
('Meanwhile', 43)
('No', 44)
('Now', 45)
('Once', 46)
('One', 47)
('Only', 48)
('Our', 49)
('Perhaps', 50)
```


## Implementing simple Text Tokenizer


```python
class TokenizerV1:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i:s for s, i in vocab.items()}

  def encode(self, text):
    preprocessed_text =  re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
    ids = [self.str_to_int[token] for token in preprocessed_text]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[id] for id in ids])
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text
```


```python
tokenizer = TokenizerV1(vocab)
text = """Yes, it is not for himself that he is searching,"
           said the superior."""
ids = tokenizer.encode(text)
print(ids)
```

```
[76, 3, 466, 462, 576, 353, 419, 849, 404, 462, 732, 3, 1, 721, 850, 826, 4]
```

```python
print(tokenizer.decode(ids))
```

```
Yes, it is not for himself that he is searching," said the superior.
```


Now, we apply tokenizer to the sample text which is not present in vocabulary.


```python
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
```

```
---------------------------------------------------------------------------

KeyError                                  Traceback (most recent call last)

<ipython-input-18-65fd82dd7ab1> in <cell line: 2>()
        1 text = "Hello, do you like tea?"
----> 2 print(tokenizer.encode(text))

<ipython-input-15-b387c8ff68db> in encode(self, text)
        7     preprocessed_text =  re.split(r'([,.:;?_!"()\']|--|\s)', text)
        8     preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
----> 9     ids = [self.str_to_int[token] for token in preprocessed_text]
        10     return ids
        11 

<ipython-input-15-b387c8ff68db> in <listcomp>(.0)
        7     preprocessed_text =  re.split(r'([,.:;?_!"()\']|--|\s)', text)
        8     preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
----> 9     ids = [self.str_to_int[token] for token in preprocessed_text]
        10     return ids
        11 

KeyError: 'Hello'
```


This error suggests that we need large and diverse training sets to extend the vocab when working on LLMs.

## Adding Special Context Tokens

Now, we will add two special tokens here:

* `<|unk|>`
* `<|endoftext|>`

```python
all_tokens = sorted(set(preprocessed_text))
all_tokens.extend(["<|unk|>", "<|endoftext|>"])
```


```python
vocab = {token:integer for integer, token in enumerate(all_tokens)}
print(len(vocab))
```
```
990
```


```python
for i, item in enumerate(list(vocab.items())[-5:]):
  print(item)
```

```
('yours—at', 985)
('youth', 986)
('you—one', 987)
('<|unk|>', 988)
('<|endoftext|>', 989)
```


```python
class TokenizerV2:
  def __init__(self, vocab):
    self.str_to_int = vocab
    self.int_to_str = {i:s for s, i in vocab.items()}

  def encode(self, text):
    preprocessed_text =  re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]
    preprocessed_text = [token if token in self.str_to_int else  "<|unk|>" for token in preprocessed_text]
    ids = [self.str_to_int[token] for token in preprocessed_text]
    return ids

  def decode(self, ids):
    text = " ".join([self.int_to_str[id] for id in ids])
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text
```


```python
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)
```

```
Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
```


```python
tokenizer = TokenizerV2(vocab)
ids = tokenizer.encode(text)
print(ids)
```

```
[988, 3, 262, 979, 506, 988, 7, 989, 988, 850, 988, 988, 585, 850, 988, 4]
```

```python
tokens = tokenizer.decode(ids)
print(tokens)
```

```
<|unk|>, do you like <|unk|>? <|endoftext|> <|unk|> the <|unk|> <|unk|> of the <|unk|>.
```

## Byte Pair Encoding


```python
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)
```

```
[15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 1659, 617, 34680, 27271, 13]
```


```python
tokens = tokenizer.decode(ids)
print(tokens)
```

```
Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace.
```


```python
sample_text = "Akwirw ier"
sample_ids = tokenizer.encode(sample_text)
print(sample_ids)
```

```
[33901, 86, 343, 86, 220, 959]
```

## Data Sampling with Sliding Window


```python
with open('/input/Leo.txt', 'r', encoding='utf-8') as file:
  raw_text = file.read()

print(raw_text)
```

```
'IT was a sunny morning, and I was on my way to Torcello. Venice lay behind us a dazzling line, with towers of gold against the blue lagoon. All at once a breeze sprang up from the sea; the small, feathery islands seemed to shake and quiver, and, like leaves driven before a gale, those flocks of colored butterflies, the fishing-boats, ran in before the storm. Far away to our left stood the ancient tower of Altinum, with the island of Burano a bright pink beneath the towering clouds. To our right, and much nearer, was a small cypress-covered islet. One large umbrella-pine hung close to the sea, and behind it rose the tower of the convent church. The two gondoliers consulted together in hoarse cries and decided to make for it.\n\n"It is San Francesco del Deserto," the elder explained to me. "It belongs to the little brown brothers, who take no money and are very kind. One would hardly believe these ones had any religion, they are such a simple people, and they live on fish and the vegetables they grow in their garden."\n\nWe fought the crooked little waves in silence after that; only the high prow rebelled openly against its sudden twistings and turnings. The arrowy-shaped gondola is not a structure made for the rough jostling of waves, and the gondoliers put forth all their strength and skill to reach the tiny haven under the convent wall. As we did so, the black bars of cloud rushed down upon us in a perfect deluge of rain, and we ran speechless and half drowned across the tossed field of grass and forget-me-nots to the convent door. A shivering beggar sprang up from nowhere and insisted on ringing the bell for us.\n\nThe door opened, and I saw before me a young brown brother with the merriest eyes I have ever seen. They were unshadowed, like a child\'s, dancing and eager, and yet there was a strange gentleness and patience about him, too, as if there was no hurry even about his eagerness.\n\nHe was very poorly dressed and looked thin. I think he was charmed to see us, though a little shy, like a hospitable country hostess anxious to give pleasure, but afraid that she has not much to offer citizens of a larger world.\n\n"What a tempest!" he exclaimed. "You have come at a good hour. Enter, enter, Signore! And your men, will they not come in?"\n\nWe found ourselves in a very small rose-red cloister; in the middle of it was an old well under the open sky, but above us was a sheltering roof spanned by slender arches. The young monk hesitated for a moment, smiling from me to the two gondoliers. I think it occurred to him that we should like different entertainment, for he said at last:\n\n"You men would perhaps like to sit in the porter\'s lodge for a while? Our Brother Lorenzo is there; he is our chief fisherman, with a great knowledge of the lagoons; and he could light a fire for you to dry yourselves by—Signori. And you, if I mistake not, are English, are you not, Signore? It is probable that you would like to see our chapel. It is not much. We are very proud of it, but that, you know, is because it was founded by our blessed father, Saint Francis. He believed in poverty, and we also believe in it, but it does not give much for people to see. That is a misfortune, to come all this way and to see nothing." Brother Leo looked at me a little wistfully. I think he feared that I should be disappointed. Then he passed before me with swift, eager feet toward the little chapel.\n\nIt was a very little chapel and quite bare; behind the altar some monks were chanting an office. It was clean, and there were no pictures or images, only, as I knelt there, I felt as if the little island in its desert of waters had indeed secreted some vast treasure, and as if the chapel, empty as it had seemed at first, was full of invisible possessions. As for Brother Leo, he had stood beside me nervously for a moment; but on seeing that I was prepared to kneel, he started, like a bird set free, toward the altar steps, where his lithe young impetuosity sank into sudden peace. He knelt there so still, so rapt, so incased in his listening silence, that he might have been part of the stone pavement. Yet his earthly senses were alive, for the moment I rose he was at my side again, as patient and courteous as ever, though I felt as if his inner ear were listening still to some unheard melody.\n\nWe stood again in the pink cloister. "There is little to see," he repeated. "We are poverelli; it has been like this for seven hundred years." He smiled as if that age-long, simple service of poverty were a light matter, an excuse, perhaps, in the eyes of the citizen of a larger world for their having nothing to show. Only the citizen, as he looked at Brother Leo, had a sudden doubt as to the size of the world outside. Was it as large, half as large, even, as the eager young heart beside him which had chosen poverty as a bride?\n\nThe rain fell monotonously against the stones of the tiny cloister.\n\n"What a tempest!" said Brother Leo, smiling contentedly at the sky. "You must come in and see our father. I sent word by the porter of your arrival, and I am sure he will receive you; that will be a pleasure for him, for he is of the great world, too. A very learned man, our father; he knows the French and the English tongue. Once he went to Rome; also he has been several times to Venice. He has been a great traveler."\n\n"And you," I asked—"have you also traveled?"\n\nBrother Leo shook his head.\n\n"I have sometimes looked at Venice," he said, "across the water, and once I went to Burano with the marketing brother; otherwise, no, I have not traveled. But being a guest-brother, you see, I meet often with those who have, like your Excellency, for instance, and that is a great education."\n\nWe reached the door of the monastery, and I felt sorry when another brother opened to us, and Brother Leo, with the most cordial of farewell smiles, turned back across the cloister to the chapel door. "Even if he does not hurry, he will still find prayer there," said a quiet voice beside me.\n\nI turned to look at the speaker. He was a tall old man with white hair and eyes like small blue flowers, very bright and innocent, with the same look of almost superb contentment in them that I had seen in Brother Leo\'s eyes.\n\n"But what will you have?" he added with a twinkle. "The young are always afraid of losing time; it is, perhaps, because they have so much. But enter, Signore! If you will be so kind as to excuse the refectory, it will give me much pleasure to bring you a little refreshment. You will pardon that we have not much to offer?"\n\nThe father—for I found out afterward that he was the superior himself—brought me bread and wine, made in the convent, and waited on me with his own hands. Then he sat down on a narrow bench opposite to watch me smoke. I offered him one of my cigarettes, but he shook his head, smiling.\n\n"I used to smoke once," he said. "I was very particular about my tobacco. I think it was similar to yours—at least the aroma, which I enjoy very much, reminds me of it. It is curious, is it not, the pleasure we derive from remembering what we once had? But perhaps it is not altogether a pleasure unless one is glad that one has not got it now. Here one is free from things. I sometimes fear one may be a little indulgent about one\'s liberty. Space, solitude, and love—it is all very intoxicating."\n\nThere was nothing in the refectory except the two narrow benches on which we sat, and a long trestled board which formed the table; the walls were white-washed and bare, the floor was stone. I found out later that the brothers ate and drank nothing except bread and wine and their own vegetables in season, a little macaroni sometimes in winter, and in summer figs out of their own garden. They slept on bare boards, with one thin blanket winter and summer alike. The fish they caught they sold at Burano or gave to the poor. There was no doubt that they enjoyed very great freedom from "things."\n\nIt was a strange experience to meet a man who never had heard of a flying-machine and who could not understand why it was important to save time by using the telephone or the wireless-telegraphy system; but despite the fact that the father seemed very little impressed by our modern urgencies, I never have met a more intelligent listener or one who seized more quickly on all that was essential in an explanation.\n\n"You must not think we do nothing at all, we lazy ones who follow old paths," he said in answer to one of my questions. "There are only eight of us brothers, and there is the garden, fishing, cleaning, and praying. We are sent for, too, from Burano to go and talk a little with the people there, or from some island on the lagoons which perhaps no priest can reach in the winter. It is easy for us, with our little boat and no cares."\n\n"But Brother Leo told me he had been to Burano only once," I said. "That seems strange when you are so near."\n\n"Yes, he went only once," said the father, and for a moment or two he was silent, and I found his blue eyes on mine, as if he were weighing me.\n\n"Brother Leo," said the superior at last, "is our youngest. He is very young, younger perhaps than his years; but we have brought him up altogether, you see. His parents died of cholera within a few days of each other. As there were no relatives, we took him, and when he was seventeen he decided to join our order. He has always been happy with us, but one cannot say that he has seen much of the world." He paused again, and once more I felt his blue eyes searching mine. "Who knows?" he said finally. "Perhaps you were sent here to help me. I have prayed for two years on the subject, and that seems very likely. The storm is increasing, and you will not be able to return until to-morrow. This evening, if you will allow me, we will speak more on this matter. Meanwhile I will show you our spare room. Brother Lorenzo will see that you are made as comfortable as we can manage. It is a great privilege for us to have this opportunity; believe me, we are not ungrateful."\n\nIt would have been of no use to try to explain to him that it was for us to feel gratitude. It was apparent that none of the brothers had ever learned that important lesson of the worldly respectable—that duty is what other people ought to do. They were so busy thinking of their own obligations as to overlook entirely the obligations of others. It was not that they did not think of others. I think they thought only of one another, but they thought without a shadow of judgment, with that bright, spontaneous love of little children, too interested to point a moral. Indeed, they seemed to me very like a family of happy children listening to a fairy-story and knowing that the tale is true.\n\nAfter supper the superior took me to his office. The rain had ceased, but the wind howled and shrieked across the lagoons, and I could hear the waves breaking heavily against the island. There was a candle on the desk, and the tiny, shadowy cell looked like a picture by Rembrandt.\n\n"The rain has ceased now," the father said quietly, "and to-morrow the waves will have gone down, and you, Signore, will have left us. It is in your power to do us all a great favor. I have thought much whether I shall ask it of you, and even now I hesitate; but Scripture nowhere tells us that the kingdom of heaven was taken by precaution, nor do I imagine that in this world things come oftenest to those who refrain from asking.\n\n"All of us," he continued, "have come here after seeing something of the outside world; some of us even had great possessions. Leo alone knows nothing of it, and has possessed nothing, nor did he ever wish to; he has been willing that nothing should be his own, not a flower in the garden, not anything but his prayers, and even these I think he has oftenest shared. But the visit to Burano put an idea in his head. It is, perhaps you know, a factory town where they make lace, and the people live there with good wages, many of them, but also much poverty. There is a poverty which is a grace, but there is also a poverty which is a great misery, and this Leo never had seen before. He did not know that poverty could be a pain. It filled him with a great horror, and in his heart there was a certain rebellion. It seemed to him that in a world with so much money no one should suffer for the lack of it.\n\n"It was useless for me to point out to him that in a world where there is so much health God has permitted sickness; where there is so much beauty, ugliness; where there is so much holiness, sin. It is not that there is any lack in the gifts of God; all are there, and in abundance, but He has left their distribution to the soul of man. It is easy for me to believe this. I have known what money can buy and what it cannot buy; but Brother Leo, who never has owned a penny, how should he know anything of the ways of pennies?\n\n"I saw that he could not be contented with my answer; and then this other idea came to him—the idea that is, I think, the blessed hope of youth that this thing being wrong, he, Leo, must protest against it, must resist it! Surely, if money can do wonders, we who set ourselves to work the will of God should have more control of this wonder-working power? He fretted against his rule. He did not permit himself to believe that our blessed father, Saint Francis, was wrong, but it was a hardship for him to refuse alms from our kindly visitors. He thought the beggars\' rags would be made whole by gold; he wanted to give them more than bread, he wanted, poverino! to buy happiness for the whole world."\n\nThe father paused, and his dark, thought-lined face lighted up with a sudden, beautiful smile till every feature seemed as young as his eyes.\n\n"I do not think the human being ever has lived who has not thought that he ought to have happiness," he said. "We begin at once to get ready for heaven; but heaven is a long way off. We make haste slowly. It takes us all our lives, and perhaps purgatory, to get to the bottom of our own hearts. That is the last place in which we look for heaven, but I think it is the first in which we shall find it."\n\n"But it seems to me extraordinary that, if Brother Leo has this thing so much on his mind, he should look so happy," I exclaimed. "That is the first thing I noticed about him."\n\n"Yes, it is not for himself that he is searching," said the superior. "If it were, I should not wish him to go out into the world, because I should not expect him to find anything there. His heart is utterly at rest; but though he is personally happy, this thing troubles him. His prayers are eating into his soul like flame, and in time this fire of pity and sorrow will become a serious menace to his peace. Besides, I see in Leo a great power of sympathy and understanding. He has in him the gift of ruling other souls. He is very young to rule his own soul, and yet he rules it. When I die, it is probable that he will be called to take my place, and for that it is necessary he should have seen clearly that our rule is right. At present he accepts it in obedience, but he must have more than obedience in order to teach it to others; he must have a personal light.\n\n"This, then, is the favor I have to ask of you, Signore. I should like to have you take Brother Leo to Venice to-morrow, and, if you have the time at your disposal, I should like you to show him the towers, the churches, the palaces, and the poor who are still so poor. I wish him to see how people spend money, both the good and the bad. I wish him to see the world. Perhaps then it will come to him as it came to me—that money is neither a curse nor a blessing in itself, but only one of God\'s mysteries, like the dust in a sunbeam."\n\n"I will take him very gladly; but will one day be enough?" I answered.\n\nThe superior arose and smiled again.\n\n"Ah, we slow worms of earth," he said, "are quick about some things! You have learned to save time by flying-machines; we, too, have certain methods of flight. Brother Leo learns all his lessons that way. I hardly see him start before he arrives. You must not think I am so myself. No, no. I am an old man who has lived a long life learning nothing, but I have seen Leo grow like a flower in a tropic night. I thank you, my friend, for this great favor. I think God will reward you."\n\nBrother Lorenzo took me to my bedroom; he was a talkative old man, very anxious for my comfort. He told me that there was an office in the chapel at two o\'clock, and one at five to begin the day, but he hoped that I should sleep through them.\n\n"They are all very well for us," he explained, "but for a stranger, what cold, what disturbance, and what a difficulty to arrange the right thoughts in the head during chapel! Even for me it is a great temptation. I find my mind running on coffee in the morning, a thing we have only on great feast-days. I may say that I have fought this thought for seven years, but though a small devil, perhaps, it is a very strong one. Now, if you should hear our bell in the night, as a favor pray that I may not think about coffee. Such an imperfection! I say to myself, the sin of Esau! But he, you know, had some excuse; he had been hunting. Now, I ask you—one has not much chance of that on this little island; one has only one\'s sins to hunt, and, alas! they don\'t run away as fast as one could wish! I am afraid they are tame, these ones. May your Excellency sleep like the blessed saints, only a trifle longer!"\n\nI did sleep a trifle longer; indeed, I was quite unable to assist Brother Lorenzo to resist his coffee devil during chapel-time. I did not wake till my tiny cell was flooded with sunshine and full of the sound of St. Francis\'s birds. Through my window I could see the fishing-boats pass by. First came one with a pair of lemon-yellow sails, like floating primroses; then a boat as scarlet as a dancing flame, and half a dozen others painted some with jokes and some with incidents in the lives of patron saints, all gliding out over the blue lagoon to meet the golden day.'
```



```python
encoded_text = tokenizer.encode(raw_text)
print(len(encoded_text))
```

```
4279
```

It has 4279 tokens in total in the training set. Now, we remove first 50 tokens from the dataset.


```python
encoded_sample = encoded_text[50:]
```

It's time to create the input-target pairs. Let's look into one example first,


```python
context_window = 4
x = encoded_sample[:context_window]
y = encoded_sample[1:context_window+1]
print(x)
print("    ",y)
```

```
[88, 14807, 3947, 284]
    [14807, 3947, 284, 13279]
```

```python
for i in range(1, context_window+1):
  input = encoded_sample[:i]
  target = encoded_sample[i]
  print(input, "-->", target)
```

```
[88] --> 14807
[88, 14807] --> 3947
[88, 14807, 3947] --> 284
[88, 14807, 3947, 284] --> 13279
```

Let's repeat the above process for getting input-target pairs, but with the actual tokens in the text not the tokenIDs.


```python
for i in range(1, context_window+1):
  input = encoded_sample[:i]
  target = encoded_sample[i]
  print(tokenizer.decode(input), "-->", tokenizer.decode([target]))
```

```
y -->  islands
y islands -->  seemed
y islands seemed -->  to
y islands seemed to -->  shake
```

we've now created input-target pairs.

## Dataset for Batched Input and Targets


```python
import torch
from torch.utils.data import Dataset, DataLoader
```

```python
class GPTDatasetV1(Dataset):
  def __init__(self, text, tokenizer, max_length, stride):
    self.input_ids = list()
    self.target_ids = list()

    token_ids = tokenizer.encode(text)

    for i in range(0, len(token_ids) - max_length, stride):
      input = token_ids[i:i+max_length]
      target = token_ids[i+1:i+max_length+1]
      self.input_ids.append(torch.tensor(input))
      self.target_ids.append(torch.tensor(target))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, index):
    return self.input_ids[index], self.target_ids[index]
```

## DataLoader to generate batches with input-target pairs


```python
# define DataLoader object
def create_dataloader_v1(
    text,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0
):

  tokenizer = tiktoken.get_encoding('gpt2')
  dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
  dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    drop_last=drop_last,
    num_workers=num_workers
)

  return dataloader
```

Let's test the dataloader with batch size of 1 with context size 4.


```python
with open('/input/Leo.txt', 'r', encoding='utf-8') as file:
  raw_text = file.read()
```

**1st example**


```python
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=1,
    max_length=4,
    stride=1,
    shuffle=False
)
```

```python
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
```

```
[tensor([[ 2043,   373,   257, 27737]]), tensor([[  373,   257, 27737,  3329]])]
```


```python
second_batch = next(data_iter)
print(second_batch)
```

```
[tensor([[  373,   257, 27737,  3329]]), tensor([[  257, 27737,  3329,    11]])]
```

**2nd example**


```python
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)
```

```python
data_iter = iter(dataloader)
input, target = next(data_iter)

print(input)
print(target)
```

```
tensor([[ 2043,   373,   257, 27737],
        [ 3329,    11,   290,   314],
        [  373,   319,   616,   835],
        [  284,  4022,  3846,    78],
        [   13, 29702,  3830,  2157],
        [  514,   257, 41535,  1627],
        [   11,   351, 18028,   286],
        [ 3869,  1028,   262,  4171]])
tensor([[  373,   257, 27737,  3329],
        [   11,   290,   314,   373],
        [  319,   616,   835,   284],
        [ 4022,  3846,    78,    13],
        [29702,  3830,  2157,   514],
        [  257, 41535,  1627,    11],
        [  351, 18028,   286,  3869],
        [ 1028,   262,  4171, 19470]])
```


## Create Token Embeddings

Let'e see how we can convert token IDs to embeddings through an example:


```python
input_ids = torch.tensor([2, 4, 1, 5])
```

Also, let's suppose that we have vocabulary of size 6 and we want to create embeddings of size 3.

```python
vocab_size = 6
output_dim = 3

torch.manual_seed(144)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
```

```
Parameter containing:
tensor([[ 2.0498, -0.5850,  0.0478],
        [ 0.9948,  0.8840,  0.0773],
        [ 3.2101, -1.1649, -0.5699],
        [ 1.3446,  1.2875,  0.9301],
        [ 0.5089,  0.4857, -0.9258],
        [ 1.8692,  0.9056,  0.5658]], requires_grad=True)
```

The weight matrix of embedding layer contains small random values. And these values will be initialized during LLM training as part of the LLM optimization.


```python
print(embedding_layer(input_ids))
```
```
tensor([[ 3.2101, -1.1649, -0.5699],
        [ 0.5089,  0.4857, -0.9258],
        [ 0.9948,  0.8840,  0.0773],
        [ 1.8692,  0.9056,  0.5658]], grad_fn=<EmbeddingBackward0>)
```

Now, we have successfully created mbedding vectors from token IDs. Next, we will add samll modification to this embeddings for encoding positional information within text.

## Encoding Word Positions

Now, let's create embeddings with vocab size 50,257 and output embedding dimesions is 256.


```python
vocab_size = 50257
output_dim = 256

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
```

Let's initiate dataloader first,


```python
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=8,
    max_length=4,
    stride=4,
    shuffle=False
)
```


```python
data_iter = iter(dataloader)
input, target = next(data_iter)

print(input)
print(input.shape)
```

```
tensor([[ 2043,   373,   257, 27737],
        [ 3329,    11,   290,   314],
        [  373,   319,   616,   835],
        [  284,  4022,  3846,    78],
        [   13, 29702,  3830,  2157],
        [  514,   257, 41535,  1627],
        [   11,   351, 18028,   286],
        [ 3869,  1028,   262,  4171]])
torch.Size([8, 4])
```


Now, we will create embeddings


```python
token_embeddings = embedding_layer(input)
print(token_embeddings.shape)
```

```
torch.Size([8, 4, 256])
```

This 8 x 4 x 256 dimensional tensor shows that each token ID is embedded as 256-dimensional vector.


```python
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
```

```
torch.Size([4, 256])
```


```python
print(pos_embeddings[1])
```

```
tensor([-1.0864e+00, -6.2188e-02, -1.0294e+00, -2.2932e-01, -6.0874e-01,
        5.3524e-01,  1.3238e+00,  2.3330e+00, -9.9768e-02, -4.7199e-01,
        -5.3248e-01,  1.8350e+00,  2.8382e-01,  1.0543e+00, -1.7998e+00,
        1.4655e+00, -1.9528e+00,  2.1799e-01,  1.3202e+00, -4.7460e-01,
        3.8520e-01, -8.1749e-02,  5.1069e-01,  1.0609e+00,  2.9112e-02,
        -4.0899e-02, -5.1593e-01,  2.6452e-01, -9.2384e-01, -1.0146e+00,
        -5.9922e-01,  2.3189e-01,  5.8988e-01,  1.5490e-01,  1.1972e+00,
        3.7747e-01, -1.1821e+00,  1.5121e+00, -2.6745e-01,  7.3872e-01,
        -5.1275e-01, -3.7004e-01,  5.3351e-01,  7.8175e-01,  1.2124e+00,
        -2.5448e+00,  9.7309e-01,  5.9424e-01, -1.5780e-01,  6.6926e-01,
        1.8240e+00, -1.5038e+00,  9.2822e-01,  1.1650e+00,  1.4926e+00,
        1.3172e-01,  1.1997e+00, -4.4992e-01,  6.4799e-01, -1.0770e+00,
        -1.9323e+00, -4.1374e-01, -5.7242e-01,  3.8571e-03,  5.9323e-01,
        1.3487e+00,  9.8320e-01,  2.1459e+00,  1.0223e+00, -2.6239e-01,
        -9.0979e-01, -6.6782e-01, -1.5164e+00,  2.2336e+00, -6.0366e-02,
        -2.5484e+00, -9.8538e-01,  9.0997e-01,  1.8729e+00,  1.5174e+00,
        -5.3172e-01,  2.0688e-01, -1.3384e+00,  3.3057e-01,  1.1832e+00,
        -1.3464e+00,  4.3966e-03,  3.0783e-01, -1.3532e+00, -1.2229e+00,
        -1.5297e+00, -9.4590e-01, -1.8807e+00,  2.6920e-01,  8.4349e-01,
        -1.6708e+00, -1.3432e-01,  1.2948e+00,  5.2372e-01,  4.3310e-01,
        1.4048e+00,  3.5675e-01, -1.4196e+00, -1.4237e-01,  4.4419e-01,
        -1.5778e+00,  7.4118e-01, -7.4923e-01, -1.4543e-02,  1.0029e+00,
        -5.4395e-01,  1.8963e-01,  4.4493e-01, -1.9422e+00,  4.7572e-01,
        -5.5757e-01,  4.2290e-01, -1.3519e+00,  7.2638e-01, -1.2369e+00,
        7.3580e-02,  6.1420e-01,  1.0867e+00,  1.2623e+00,  1.7236e-01,
        -1.3619e+00,  4.1051e-01, -9.4217e-01,  3.9643e-01, -1.2299e+00,
        8.3577e-01, -1.0962e+00,  1.9803e+00, -1.0517e+00, -1.6224e+00,
        4.7761e-01, -2.1575e-01, -2.8793e-01,  1.2426e+00,  4.1573e-01,
        2.9522e-01,  8.2924e-01,  7.2013e-01, -3.7844e-01, -6.7713e-01,
        -1.3009e-01,  1.3781e+00,  1.2638e+00,  1.0405e+00, -6.9116e-01,
        -1.1137e+00, -1.6517e-01, -1.5704e+00,  4.8140e-01, -7.0258e-01,
        4.8240e-01, -8.6068e-02, -1.0679e-01, -1.3662e+00,  5.6207e-01,
        -6.2838e-01, -2.7161e+00, -6.8913e-01, -1.2084e+00,  1.4747e+00,
        -6.4045e-01,  6.2532e-02,  1.3075e+00, -1.1566e+00,  1.2228e+00,
        3.0038e-01,  8.4032e-01,  1.3070e+00, -1.7483e-01,  1.9360e-02,
        -1.0322e+00,  6.9732e-01, -9.1352e-01,  2.4516e-03, -4.3687e-01,
        -9.5746e-01,  4.7267e-01,  1.2204e-01, -1.7901e-01, -6.9567e-01,
        7.3348e-01, -5.2585e-01,  4.5684e-01,  1.7876e-01, -1.9677e-01,
        -7.0036e-01, -8.2225e-01,  4.8610e-01,  2.9099e-01, -6.6831e-01,
        7.8089e-01,  5.8560e-01,  3.8179e-02, -5.7200e-01,  2.3666e-01,
        2.0995e-01,  1.5591e-01,  1.2787e+00, -8.4235e-02,  7.4501e-01,
        -9.0400e-01,  2.2010e+00, -1.2678e-01, -1.2541e+00,  2.1067e+00,
        -9.2460e-01, -1.9878e+00,  4.4463e-01,  7.7941e-01, -1.1651e-01,
        -2.3809e-02,  5.8151e-01, -1.3654e+00,  9.5289e-01,  1.3414e-01,
        -1.4263e+00, -7.7353e-01,  2.1153e-01,  1.0364e+00,  2.3102e-01,
        -1.0749e+00, -1.5740e+00,  7.7539e-02, -1.1177e+00, -5.8328e-01,
        1.1142e+00, -2.1941e-01, -1.4421e+00,  1.7749e+00,  5.0158e-01,
        -1.7264e+00, -1.5059e+00, -1.2828e+00, -4.6148e-01, -6.4830e-01,
        -9.4239e-01,  1.1113e-01, -1.3676e+00, -1.2901e+00, -6.3204e-01,
        7.6443e-01,  1.7050e+00, -8.1775e-01, -1.3036e-01, -6.3764e-01,
        -2.8533e-02,  8.7735e-01,  6.4208e-01, -1.7845e+00, -5.4175e-01,
        -1.3801e+00], grad_fn=<SelectBackward0>)
```

```python
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
```
```
torch.Size([8, 4, 256])
```
