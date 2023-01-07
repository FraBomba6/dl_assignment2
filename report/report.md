---
title: "Deep Learning Assignment"
author: [Francesco Bombassei De Bona, Andrea Cantarutti]
date: "2022-01-08"
keywords: [NLP, DeepLearning]
---

# Introduction

The following report describes the strategies adopted and the results obtained with the aim of defining a Natural Language Processing model. The latter was specifically trained to solve cloze tests with a limited set of available candidates. 

# Project Structure

The root folder of the project repository contains the following files:

- `requirements.txt`, which lists the libraries that need to be installed
- `assignment_2.pdf`, which is the original assignment document
- `README.md`, which contains a brief description of the project
- `config.sh`, which allows to rapidly configure any debian machine for training
- `data/`
  + `train.jsonl`, which contains the train dataset
  + `test.jsonl`, which contains the test dataset
- `src/`
  + `utils.py`, which referes to a module with all the utility functions that were implemented
  + `maskedlm.py`, which refers to the BertForMaskedLM implementation strategy
  + `multiplechoice.py`, which referes to the BertForMultipleChoice implementation strategy
- `report/`, which contains all the files needed to generate the current report

# Task Description 

The task requires to develop a model that predicts the solution of a cloze test with a set of two candidate answers, in which one of them is correct. The results obtained, then, need to be analyzed in order to evaluate the performance and the overall accuracy of the model.

# Models and results

## Dataset

The data consists of two **ljson** files which correspond to the train and test datasets. These were first downloaded to the `data/` folder. A custom function was, then, defined in the `src/utils.py` module in order to load each line inside a **pandas DataFrame**.

Another function was, then, defined in order to apply a random split to a TensorDataset. This was done in order to extract a **validation dataset** from the **training dataset**. More specifically, a random 20% of the training set was used for validation purposes.

## BertForMultipleChoice

### Input Handling and Tokenization

The first solution implemented involved the usage of the **BertForMultipleChoice** model. The latter is built on top of **BERT** and allows to finetune it on question answering with a limited set of options.

Starting from the explanation found in the **HuggingFace documentation**, several strategies were adopted in order to define the **input structure**. For this purpose, a custom tokenize function (named `tokenize_for_multiple_choice(sentence_list, options_list, target_list)`) was defined in the `utils.py` module.

The first strategy involved the composition of the base sentence with all the available options. The second one, instead, required to split the sentence at its gap and, then, combine the first part, the `[SEP]` token, the candidate option and the remaining part. The last one involved the combination of the incomplete sentence with the candidate option, separated by a `[SEP]` token. 

Given the following case:

- **Sentence**: `Emily looked up and saw Patricia racing by overhead, as _ was under the ramp .`
- **Options**: `[Emily, Patricia]`
- **Answer**: `Emily`

The three structures obtained would, then, be:

| Strategy | Output                                                                                                                        |
|----------|-------------------------------------------------------------------------------------------------------------------------------|
| 1        | `[CLS] Emily looked up and saw Patricia racing by overhead, as Emiliy was under the ramp . [SEP]`, `[CLS] Emily looked up and saw Patricia racing by overhead, as Patricia was under the ramp . [SEP]` |
| 2        | `[CLS] Emily looked up and saw Patricia racing by overhead, as [SEP] Emily was under the ramp [SEP]`, `[CLS] Emily looked up and saw Patricia racing by overhead, as [SEP] Patricia was under the ramp [SEP]` |
| 3        |  `[CLS] Emily looked up and saw Patricia racing by overhead, as _ was under the ramp [SEP] Emiliy [SEP]`, `[CLS] Emily looked up and saw Patricia racing by overhead, as _ was under the ramp [SEP] Patricia [SEP]` |


Moreover, sentences were tokenized using a **BertTokenizer** with a **max length** of 64 (chosen in order to allow each sentence not to be truncated) and **padding to max length** applied. The list of input ids, the list of attention masks and the return targets one were, finally, stacked in order to allow batch processing by the model.

### Results

## BertForMaskedLM

# Conclusions
