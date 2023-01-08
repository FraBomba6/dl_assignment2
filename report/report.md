---
title: "Deep Learning Assignment"
author: [Francesco Bombassei De Bona, Andrea Cantarutti]
date: "2022-01-08"
keywords: [NLP, DeepLearning]
header-includes:
  - \usepackage{graphicx}
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

| Strategy | Output                                                                                                                                                                                                             |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1        | `[CLS] Emily looked up and saw Patricia racing by overhead, as Emiliy was under the ramp . [SEP]`, `[CLS] Emily looked up and saw Patricia racing by overhead, as Patricia was under the ramp . [SEP]`             |
| 2        | `[CLS] Emily looked up and saw Patricia racing by overhead, as [SEP] Emily was under the ramp [SEP]`, `[CLS] Emily looked up and saw Patricia racing by overhead, as [SEP] Patricia was under the ramp [SEP]`      |
| 3        | `[CLS] Emily looked up and saw Patricia racing by overhead, as _ was under the ramp [SEP] Emiliy [SEP]`, `[CLS] Emily looked up and saw Patricia racing by overhead, as _ was under the ramp [SEP] Patricia [SEP]` |

Moreover, sentences were tokenized using a **BertTokenizer** with a **max length** of 64 (chosen in order to allow each sentence not to be truncated) and **padding to max length** applied. The list of input ids, the list of attention masks and the return targets one were, finally, stacked in order to allow batch processing by the model.

### Results

The training module was implemented in `src/multiplechoice.py`. The training properties employed were defined as follows:

| Property      | Value  |
| ------------- | ------ |
| Epochs        | 2      |
| Batch Size    | 32     |
| Optimizer     | AdamW  |
| Learning Rate | 5e-3   |
| Scheduler     | linear |

Regardless of the strategy used for the input structuring, a slight and therefore not significant decrease in the loss function's output was observed, together with no improvement in terms of **prediction accuracy**. Moreover, it was determined that the model **converges** after the training. In fact, the latter causes all the predictions to be the same in terms of option index. This behaviour still remains unclear and causes BertForMultipleChoice not to be a suitable option for the given task.

The following screenshot shows the output of the evaluation on the training, validation and testing dataset **before the two epochs training**:

|
|

\begin{figure}[H]
\centering
\includegraphics[width=400px]{img/MultipleChoice_before.png}
\end{figure}

\newpage

The following screenshot shows, instead, the output of the evaluation on the training, validation and testing dataset **after the training of each epoch**:

\begin{figure}[H]
\centering
\includegraphics[width=340px]{img/MultipleChoice_after.png}
\end{figure}

\newpage

## BertForMaskedLM

The second solution attempted involved the usage of the **BertForMaskedLM** model. The latter is built on top of **BERT** and allows to finetune it on the prediction of words which were intentionally hidden from a sentence by looking at its context.

Again, a custom tokenize function (named `tokenize_for_mlm(sentence_list, options_list, target_list)`) was defined in the `utils.py` module, in order to tokenize and structure the input as a Masked Language Modelling task.

On these basis, the `_` character (which represents the missing word in the dataset) was replaced with the apposite `[MASK]` token and, then, both the correct and the masked sentence were tokenized using a **BertTokenizer** with, again, **max length** of 64 and **padding to max length** applied. The correct sentence was used in order to produce a target label list, while the masked one was used as input for the model. 

The two candidates were also tokenized with **max length** of 6 and **padding to max length** applied (in this case, no special tokens were added by the tokenizer). This was done in order to ensure the presence of the missing words in the model's vocabulary.

The list of input ids, the list of attentions masks, the target labels list and the option tokens list (together with the correct option index) were, then, all stacked to allow batch processing by the model.

### Results

The training module was implemented in `src/maskedlm.py`. The training properties employed were defined as follows:

| Property      | Values           |
| ------------- | ---------------- |
| Epochs        | 2                |
| Batch Size    | 16, 32, 64       |
| Optimizer     | AdamW            |
| Learning Rate | 5e-3, 5e-4, 5e-5 |
| Scheduler     | linear           |

Even in this case, a non-significant decrease in the loss function's output and no improvement in terms of accuracy were observed, even by changing hyperparameters such as **batch size** and **learning rate**. After the training phase, the same kind of convergence found in the previous model caused, again, all the predictions to drift to the same index.

A subsequent attempt involved the complete substitution of the option candidates with atomic tokens (for example, _a_ instead of _Emily_ and _b_ instead of _Patricia_) in order to avoid the presence of multiple tokens corresponding to a single word, but no improvements were observed.

The following screenshot shows the output of the evaluation on the training, validation and testing dataset **before the two epochs training**:

|
|

\begin{figure}[H]
\centering
\includegraphics[width=400px]{img/Masked_before.png}
\end{figure}

\newpage

The following screenshot shows, instead, the output of the evaluation on the training, validation and testing dataset **after the training of each epoch**:

\begin{figure}[H]
\centering
\includegraphics[width=280px]{img/Masked_after.png}
\end{figure}

\newpage

## XLNetForMultipleChoice

Other attempts involved the usage of the **XLNetForMultipleChoice** transformer model, which, in some cases, provides a few improvements compared to BERT. Even in this case, though, the same performances of the previously attemptes models were observed.

# Conclusions



