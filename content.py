# # %%
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# # %%
# # %%
# data = pd.read_csv('data\movies_metadata.csv', low_memory=False)
# data = data.head(20000)
# # %%
# data['overview'] = data['overview'].fillna('')

# vectorizer = TfidfVectorizer(stop_words='english')
# x = vectorizer.fit_transform(data['overview'])
# tfidf_dict = vectorizer.get_feature_names()

# data_array = x.toarray()
# df = pd.DataFrame(data_array, columns=tfidf_dict)

# indices = pd.Series(data.index, index=data['title']).drop_duplicates()

# # %%

# df.to_csv('df.csv')
# # %%



print("Hello")