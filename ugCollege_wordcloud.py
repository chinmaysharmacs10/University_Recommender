import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS   # pillow also

df = pd.read_csv('admission_data_cleaned.csv')

# Make a variable in which all the UG College names are added
words = " ".join(df['ugCollege'])

# Create the word cloud
wc = WordCloud(background_color="white", random_state=1,stopwords=STOPWORDS, max_words = 2000, width =1500, height = 1500)
wc.generate(words)

# Plot the word cloud
plt.figure(figsize=[10,10])
plt.imshow(wc,interpolation="bilinear")
plt.axis('off')
plt.show()