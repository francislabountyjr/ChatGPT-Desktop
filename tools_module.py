import requests
import torch
import copy
import urllib.parse
from typing import Any, List
from duckduckgo_search import ddg, ddg_answers, ddg_images, ddg_videos, ddg_news, ddg_maps, ddg_translate, ddg_suggestions
from transformers import (
    AutoTokenizer,
    AutoModel,
)

from utils import mean_pooling


class WebSearch():
    """
    Class to handle web search
    """
    
    def __init__(self) -> None:
        """
        Initialize web search
        """
        self.cache = {}

    def search(self, keywords: Any, region: str = "us-en", safesearch: str = "Off", time: Any | None = 'y', max_results: Any | None = 20, page: int = 1, output: Any | None = None, download: bool = False, cache: bool = False) -> list:
        """
        Search

        Parameters:
            keywords (Any): keywords
            region (str): region
            safesearch (str): safesearch
            time (Any | None): time (one of: 'd', 'w', 'm', 'y')
            max_results (Any | None): max results
            page (int): page
            output (Any | None): output
            download (bool): download
            cache (bool): If True, cache results

        Returns:
            list: results
        """
        if cache and 'search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download) in self.cache:
            return self.cache['search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download)]
        response = ddg(keywords=keywords, region=region, safesearch=safesearch, time=time, max_results=max_results, page=page, output=output, download=download)
        if cache:
            self.cache['search' + str(keywords) + str(region) + str(safesearch) + str(time) + str(max_results) + str(page) + str(output) + str(download)] = response
        return response


class WolframAlpha():
    """
    Class to handle wolfram alpha api requests
    """
    def __init__(self, app_id: str) -> None:
        """
        Initialize wolfram alpha

        Parameters:
            app_id (str): app id
        """
        self.app_id = app_id
        self.cache = {}
        self.endpoints = {
            'short_answer': 'https://api.wolframalpha.com/v1/result?i=',
        }

    def get_short_answer(self, query: str, cache: bool = False) -> str:
        """
        Get short answer result

        Parameters:
            query (str): query
            cache (bool): If True, cache results

        Returns:
            str: result
        """
        if cache and 'get_short_answer' + str(query) in self.cache:
            return self.cache['get_short_answer' + str(query)]
        response = requests.get(self.endpoints['short_answer'] + urllib.parse.quote(query) + '&appid=' + self.app_id)
        response = {
            'query': query,
            'response': response.text,
        }
        if cache:
            self.cache['get_short_answer' + str(query)] = response
        return response
    

class Retriever ():
    """
    Adapted from - https://github.com/conceptofmind/toolformer/blob/main/tools.py
    retrieval
    Uses Carptriever to retrieve sentences before the current context.
    input_sentences - List[String], sentences to retrieve from
    input_text - String, the input text (e.g. The dog's name is)
    k - The number of sentences to retrieve
    n - The number of sentences to add after the retrieved sentences
    output - A list of strings, each string is the retrieved sentence, and the n sentences after.
    """
    def __init__(self, device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
        """
        Initialize retriever
        
        Parameters:
            device (str): device
        """
        self.model = AutoModel.from_pretrained(
            "CarperAI/carptriever-1", add_pooling_layer=False
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("CarperAI/carptriever-1")

    def retrieval(self, input_sentences: List[str], input_text: str, k: int, n: int = 1) -> List[str]:
        """
        Retrieve sentences
        
        Parameters:
            input_sentences (List[str]): input sentences
            input_text (str): input text
            k (int): number of sentences to retrieve (sorted by score)
            n (int): number of following sentences to add to the retrieved sentences

        Returns:
            List[str]: retrieved sentences
        """
        if k > len(input_sentences):
            # I'd error but LMs do stupid stuff sometimes
            return input_sentences
        input_sentences = copy.deepcopy(input_sentences)
        input_sentences.append(input_text)
        output_list = []
        for sentence in input_sentences:
            inputs = self.tokenizer(
                sentence, padding=True, truncation=True, return_tensors="pt"
            )
            # print(inputs)
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = mean_pooling(outputs[0], inputs["attention_mask"])
            output_list.append(embeddings)
        query_embedding, sentence_embeddings = output_list[-1], torch.concat(
            output_list[:-1], 0
        )
        # print(len(sentence_embeddings), sentence_embeddings[0].shape)
        scores = (query_embedding @ sentence_embeddings.transpose(0, 1)).cpu().tolist()
        # print(scores)
        sentence_score_pairs = sorted(
            zip(range(len(input_sentences)), scores[0]), reverse=True, key=lambda x: x[1]
        )
        # print(sentence_score_pairs)
        output_list = []
        for sentence_pair in sentence_score_pairs[:k]:
            sentence_index = sentence_pair[0]
            sentence = input_sentences[sentence_index]
            for j in range(n):
                if sentence_index + j + 1 < len(input_sentences):
                    sentence += " " + input_sentences[sentence_index + j + 1]
            output_list.append(sentence)
        output_list
        return output_list
