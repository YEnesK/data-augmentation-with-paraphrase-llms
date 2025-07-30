# data-augmentation-with-paraphrase-llms
Implementation LLM-based text-augmentation pipeline that enlarges IMDB and AG News datasets with PEGASUS/T5 paraphrasers, embeds all texts with all-MiniLM-L6-v2, trains an MLP classifier, and reports accuracy gains of up to ~10 pp thanks to augmented train / test ensembles.
