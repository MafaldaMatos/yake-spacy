# NLP Spacy Model with Yake Weak Labeling

The goal of this project is to train a Spacy model with weak labelled annotations made by Yake.
In order to do this, a dataset was selected to label - the Portuguese corpus from project Professor HeidelTime, labelling was done, and lastly, training of a base Portuguese Spacy model. 
Some bugs were found on Yake's keyword extractor, which led to faulty annotations. The unaligned ones were removed from the final dataset.
Using 1000 samples, Yake's keyword extractor with n=3 and 10 epochs of training, the model obtained a f1 accuracy of 24\%.
Due to the problems encountered, obtaining a high accuracy model could still be possible.
For future work, it would be important to - fix Yake's bugs and try again, use other languages, and create more Portuguese narrative data sets.
