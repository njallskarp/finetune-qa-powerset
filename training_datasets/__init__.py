
from definitions.declarations import QaData


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

        real_answer = answer['text']
        start_idx = answer['answer_start']
        # Get the real end index
        end_idx = start_idx + len(real_answer)
        # Deal with the problem of 1 or 2 more characters 
        if text[start_idx:end_idx] == real_answer:
            answer['answer_end'] = end_idx
        # When the real answer is more by one character
        elif text[start_idx-1:end_idx-1] == real_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1  
        # When the real answer is more by two characters  
        elif text[start_idx-2:end_idx-2] == real_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2
