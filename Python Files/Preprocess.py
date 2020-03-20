def Preprocess(documents, remove_stopwords, removePuncs, useStemming, useLemma):

    for l in range(len(documents)):
      label = documents[l][1]
      tmpReview = []
      for w in documents[l][0]:
        newWord = w
        if remove_stopwords and (w in stopwords_en):
          continue
        if removePuncs and (w in punctuations):
          continue

        if useStemming:
          newWord = lancaster.stem(newWord)

        if useLemma:
          newWord = wordnet_lemmatizer.lemmatize(newWord)
        tmpReview.append(newWord)
      documents[l] = (tmpReview, label)
      documents[l] = (' '.join(tmpReview), label)

    return documents
