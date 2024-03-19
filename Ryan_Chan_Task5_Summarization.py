import requests
import re
from bs4 import BeautifulSoup

# Scrape Text from URL
def get_page(url):
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')
    return soup

def collect_text(soup):
    text = f'url: {url}\n\n'
    para_text = soup.find_all('p')
    print(f"paragraphs text = \n {para_text}")
    for para in para_text:
        text += f"{para.text}\n\n"
    return text

url = "https://medium.com/@subashgandyer/papa-what-is-a-neural-network-c5e5cc427c7"

text = collect_text(get_page(url))
print(text)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from gensim.summarization import summarize as gensim_summarize
from summa.summarizer import summarize as summa_summarize
from summa import keywords

# Convert scraped text to plaintext parser
parser = PlaintextParser.from_string(text, Tokenizer("english"))
doc = parser.document
print(doc)

# Sumy
# 1. LSA
lsa_summarizer = LsaSummarizer()
# 2. Luhn
luhn_summarizer = LuhnSummarizer()
# 3. LexRank
lexrank_summarizer = LexRankSummarizer()
# 4. TextRank
textrank_summarizer = TextRankSummarizer()

lsa_summary = lsa_summarizer(doc, 5)
luhn_summary = luhn_summarizer(doc, 5)
lexrank_summary = lexrank_summarizer(doc, 5)
textrank_summary = textrank_summarizer(doc, 5)

# Gensim
gensim_summary = gensim_summarize(text, word_count=200, ratio=0.1)

# Summa
summa_summary = summa_summarize(text, ratio=0.1)

# Print summaries
print("LSA Summary:")
for sentence in lsa_summary:
    print(sentence)

print("\nLuhn Summary:")
for sentence in luhn_summary:
    print(sentence)

print("\nLexRank Summary:")
for sentence in lexrank_summary:
    print(sentence)

print("\nTextRank Summary:")
for sentence in textrank_summary:
    print(sentence)

print("\nGensim Summary:")
print(gensim_summary)

print("\nSumma Summary:")
print(summa_summary)

# Put all summaries into a single string
all_summaries = (
    "LSA Summary:\n" +
    '\n'.join(str(sentence) for sentence in lsa_summary) + "\n\n" +
    "Luhn Summary:\n" +
    '\n'.join(str(sentence) for sentence in luhn_summary) + "\n\n" +
    "LexRank Summary:\n" +
    '\n'.join(str(sentence) for sentence in lexrank_summary) + "\n\n" +
    "TextRank Summary:\n" +
    '\n'.join(str(sentence) for sentence in textrank_summary) + "\n\n" +
    "Gensim Summary:\n" +
    gensim_summary + "\n\n" +
    "Summa Summary:\n" +
    summa_summary
)

# Write all summaries to a single text file
with open("Ryan_Chan_Task5_Summarization.txt", "w") as f:
    f.write(all_summaries)
