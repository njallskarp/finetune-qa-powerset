from tokenizers import Encoding
from transformers.convert_slow_tokenizer import Tokenizer

from definitions.declarations import AnswerKey, QaData


def __get_all_domain_answers(domain_data: dict[str, list[QaData]]) -> list[AnswerKey]: 
    data = []

    train_data = domain_data["train"]
    test_data = domain_data["test"]

    data.extend(train_data)
    data.extend(test_data)

    return [qa["answer_info"] for qa in data]

def __add_token_positions(encodings: Encoding, tokenizer: Tokenizer, answers: list[AnswerKey]):
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    
    start_pos, end_pos = [], []
    
    for i, answer in enumerate(answers):
        start = encodings.char_to_token(i, answer['answer_start'])
        end   = encodings.char_to_token(i, answer['answer_end'])
        
        if start is None:
            start = tokenizer.model_max_length
        if end is None:
            end = encodings.char_to_token(i, answer['answer_end'] - 1)
        if end  is None:
            end = tokenizer.model_max_length
            
        start_pos.append(start)
        end_pos.append(end)
        
    encodings.update({'start_positions': start_pos, 'end_positions': end_pos})


def __correct_span_errors(dataset: dict[str, list[QaData]]) -> None:
    """
    Code reused from open source colab notebook and adapted:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    data = []
    train_data = dataset["train"]
    test_data = dataset["test"]
    data.extend(train_data)
    data.extend(test_data)

    for info in data:
        answer = info["answer_info"]
        text = info["paragraph"]

        real_answer = answer["text"]
        start_idx = answer["answer_start"]
        # Get the real end index
        end_idx = start_idx + len(real_answer)
        # Deal with the problem of 1 or 2 more characters
        if text[start_idx:end_idx] == real_answer:
            answer["answer_end"] = end_idx
        # When the real answer is more by one character
        elif text[start_idx - 1 : end_idx - 1] == real_answer:
            answer["answer_start"] = start_idx - 1
            answer["answer_end"] = end_idx - 1
        # When the real answer is more by two characters
        elif text[start_idx - 2 : end_idx - 2] == real_answer:
            answer["answer_start"] = start_idx - 2
            answer["answer_end"] = end_idx - 2
