# Overview

This project explores the creativity of Large Language Models (LLMs) by challenging them to invent artificial languages while following predefined word orders: Subject-Object-Verb (SOV) and Subject-Verb-Object (SVO). By analyzing the generated languages, we aim to understand the impact of word order on linguistic structure and similarity to existing languages. The study compares the results from an open-source model, Llama2-7B, and a non-open-source model, ChatGPT, to assess their linguistic capabilities and limitations.

# Features

This project implements a Python-based system that facilitates the management of conversations with LLMs, the extraction of generated artificial languages, and various analyses such as language detection and tokenization patterns. The system allows for multiple conversation sessions, storing them in .json format for easy retrieval. Each model is tasked with generating 100 texts in its respective artificial language, which are then processed and analyzed. Additionally, the project generates visualizations in the form of pie charts to display language detection results and performs tokenization analysis for Llama2-generated texts.

# Technical Details

The study utilizes two different LLMs: Llama2-7B, which was accessed via Hugging Face, and ChatGPT, which is not open-source. The implementation is done entirely in Python, leveraging multiple libraries to handle various tasks. The transformers library is crucial for accessing Llama2’s Autotokenizer, while torch and bitsandbytes enable efficient processing of the models. Data handling is done using json, and matplotlib is used to generate pie charts for visual analysis. The project also employs langdetect for identifying the detected language of the generated texts, collections for data processing, and os for managing files.

# Results

The study found that Llama2-7B generated a language called Nuvolish, following the SOV word order, while ChatGPT created Tsaléran, adhering to the SVO structure. Interestingly, despite enforcing specific word orders, the generated languages were not necessarily most similar to other real-world languages that follow the same structure. For example, Nuvolish showed similarities to languages such as Croatian, Italian, Finnish, and English, even though they primarily use an SVO structure. Similarly, Tsaléran was most similar to Indonesian, Swedish, Somali, and Turkish, with only some of these languages actually following the expected SVO pattern. This suggests that the predefined word order had less of an impact on linguistic similarity than initially anticipated.

Moreover, working with Llama2 revealed key insights into its tokenization process. By modifying its system prompt, we were able to guide the model's linguistic behavior more effectively. The project also implemented tokenization analysis, which provided insights into how Llama2 assigns tokens to words and characters within its artificial language.
