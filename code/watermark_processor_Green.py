import hashlib
import random

import hashlib

import hashlib
import torch
from transformers import LogitsProcessor
from tokenizers import Tokenizer
import random
import copy
import torch
import math
from math import sqrt
from normalizers import normalization_strategy_lookup
from torch import Tensor

from nltk.util import ngrams
import collections

import numpy as np
import scipy.stats

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Title2Seed:
    """
    A class to encode words and sentences into numerical representations and 
    generate a probabilistic key using hashing and modular arithmetic.
    """
    
    @staticmethod
    def encode_word(word):
        """
        Encode a word by converting each letter to its alphabetical index (1-based).
        For example, 'JAB' -> '1012'.
        
        Args:
            word (str): The word to encode.
        
        Returns:
            str: The encoded representation of the word.
        """
        return ''.join(str(ord(char.upper()) - ord('A') + 1) for char in word if char.isalpha())
    
    @staticmethod
    def encode_sentence(sentence):
        """
        Encode a sentence by encoding each word and combining the results.
        
        Args:
            sentence (str): The sentence to encode.
        
        Returns:
            str: The concatenated encoded representation of all words in the sentence.
        """
        if not isinstance(sentence, str):
            raise TypeError("Expected 'sentence' to be of type 'str'")
    
        words = sentence.split()
        encoded_words = [Title2Seed.encode_word(word) for word in words]
        return ''.join(encoded_words)
    
    @staticmethod
    def probabilistic_conversion(encoded_sentence):
        """
        Use a probabilistic method (hashing and modular arithmetic) to convert the
        encoded sentence into a unique key.
        
        Args:
            encoded_sentence (str): The encoded sentence string.
        
        Returns:
            int: A unique key generated from the encoded sentence.
        """
        # Hash the encoded sentence using SHA256 for uniqueness
        hashed_value = hashlib.sha256(encoded_sentence.encode()).hexdigest()
        
        # Convert the hash to an integer
        int_value = int(hashed_value, 16)
        
        # Apply a modulo operation with a large prime number to ensure the key fits within a specific range
        prime_modulo = 15485863  # A large prime number
        key = int_value % prime_modulo
        
        return key
    
    @staticmethod
    def generate_seed(sentence):
        """
        Generate a unique random seed for the given sentence.
        
        Args:
            sentence (str): The sentence to generate a seed for.
        
        Returns:
            int: A unique seed generated from the sentence.
        """
        encoded_sentence = Title2Seed.encode_sentence(sentence)
        unique_key = Title2Seed.probabilistic_conversion(encoded_sentence)
        return unique_key


# # Example usage
# if __name__ == "__main__":
#     sentence = ["Can Large Language Models Unlock Novel Scientific Research Ideas","Can Large Language Models Unlock Novel Scientific Research Ideas"]
#     for sent in sentence:
#         seed = generate_seed(sent)
#         print(f"The unique random seed for the sentence '{sent}' is: {seed}")

class WatermarkBase(Title2Seed):
    def __init__(
            self,
            vocab: list[int] = None,
            gamma: float = 0.5,
            wm_mode = "combination",
            seeding_scheme: str = "simple_1",  # mostly unused/always default
            select_green_tokens: bool = True,
            title:str =None,
            args=None
    ):
        self.vocab = vocab
        self.wm_mode = wm_mode
        self.seeding_scheme = seeding_scheme
        self.select_green_tokens = select_green_tokens
        self.vocab_size = len(vocab)
        self.gamma = args.gamma
        self.rng = None
        super().__init__()

    def _get_greenlist_ids(self, title: str) -> list[int]:
        """
        Generate greenlist and redlist IDs based on a title and vocabulary parameters.

        Args:
            title (str): Title used to seed the random number generator.

        Returns:
            tuple[list[int], list[int]]: Greenlist and redlist token IDs.
        """
        # Use a dedicated torch Generator to avoid conflicts
        gen = torch.Generator(device='cpu')
        title_seed = Title2Seed.generate_seed(title)
        gen.manual_seed(title_seed)

        # Generate a random permutation
        vocab_permutation = torch.randperm(self.vocab_size, generator=gen).tolist()

        # Split into greenlist and redlist
        greenlist_size = int(self.vocab_size * self.gamma)
        greenlist_ids = vocab_permutation[:greenlist_size]
        redlist_ids = vocab_permutation[greenlist_size:]

        return greenlist_ids, redlist_ids
    
    def _get_orange_ids(paper: str, paperlist: list[int], redlist: list[int]) -> list[int]:
        """
        Generate an orange token list by taking all tokens from the paper text.

        Args:
            paper (str): Content of the research paper.
            tokenizer: Tokenizer instance to tokenize the paper.
            redlist (list[int]): List of redlist token IDs.

        Returns:
            list[int]: A list of orange token IDs (all tokens from the paper that are in the redlist).
        """
        # Tokenize the paper content

        # Intersect paper tokens with the redlist
        orange_tokens = list(set(paperlist) & set(redlist))

        if not orange_tokens:
            print("Warning: No overlap between paper tokens and redlist. Returning an empty orange list.")
        return orange_tokens
    
    

class WatermarkLogitsProcessor_with_preferance(WatermarkBase, LogitsProcessor):
    def __init__(self, title: str,paperlist: list[int], **kwargs):
        self.title = title
        self.paperlist = paperlist
        self.delta: float = 2.0
        self.theta: float = 2.0
        self.decrease_delta: bool = True,
        self.idx_t = 0
        super().__init__(**kwargs)
    
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
            green_tokens_mask = torch.zeros_like(scores)
            for b_idx in range(len(greenlist_token_ids)):
                green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
            final_mask = green_tokens_mask.bool()
            return final_mask


    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float, decrease_delta: bool) -> torch.Tensor:
        if decrease_delta:
            greenlist_bias = greenlist_bias * (1 / (1 + 0.001 * self.idx_t))
            # greenlist_bias=4.84*(math.e)**(-1*0.001*self.idx_t)
        scores[greenlist_mask] = scores[greenlist_mask] + self.delta #greenlist_bias
        # print(greenlist_bias,self.idx_t)
        return scores
    

    
    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor) ->torch.FloatTensor:
        if self.rng is None:
            self.rng = torch.Generator(device=device)
        greenlist_token_ids,  redlist_token_ids = self._get_greenlist_ids(self.title)
        greenlist_token_ids = list(set(greenlist_token_ids + self.paperlist))
        green_tokens_mask = self._calc_greenlist_mask(scores, [greenlist_token_ids]) 

        #scores_withnomask = copy.deepcopy(scores)
        scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta,decrease_delta=self.decrease_delta)
        return scores
         
       
class WatermarkDetector_with_preferance(WatermarkBase):
    def __init__(
            self,
            *args,
            device: torch.device = None,
            tokenizer: Tokenizer = None,
            z_threshold: float = 4.0,
            # normalizers: list[str] = ["unicode"],
            normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
            ignore_repeated_bigrams: bool = False,
            title: str = None,
            paperlist: None,
            # userid,
            **kwargs

    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.title = title
        self.paperlist = paperlist
        if self.title == None:
            raise ValueError('Title empty during decoding')

        self.z_threshold = z_threshold

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))
    
    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z
    
    def _compute_z_score_changed(self, observed_count, T, new_gamma):
        # count refers to number of green tokens, T is total number of tokens
        
        expected_count = new_gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
        self,
        input_ids: Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_p_value: bool = True,
    ):
        mark = ""

        ignore_repeated_bigrams = True #change
        if ignore_repeated_bigrams:  # false
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask == False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]],
                                      device=self.device)  # expects a 1-d prefix tensor on the randperm device

                greenlist_ids,_ = self._get_greenlist_ids(self.title)
                greenlist_ids = list(set(greenlist_ids + self.paperlist))

                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())

            score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        # print(green_token_count / num_tokens_scored)
        new_gamma = len(greenlist_ids)/self.vocab_size
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score_changed(green_token_count, num_tokens_scored, new_gamma)))
 
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score_changed(green_token_count, num_tokens_scored,new_gamma)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        sim_score=green_token_count / num_tokens_scored
        gr_sim_score=np.array(sim_score)
        return score_dict, gr_sim_score,None, mark

    def detect(
            self,
            text: str = None,
            tokenized_text: list[int] = None,
            return_prediction: bool = True,
            return_scores: bool = True,
            z_threshold: float = None,
            **kwargs,
    ) -> dict:
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections
        
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")
        

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
                self.device)
            
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]
            
        output_dict = {}
        # print("in _tokenized:", tokenized_text.shape)
        score_dict, gr_score,_, mark = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)

        return output_dict, gr_score, mark


            # print(tokenized_text)
            # exit(0)



if __name__ == "__main__":
    obj = WatermarkBase(vocab=[1,2,3])
    print(obj._get_greenlist_ids("hi"))
    





        
