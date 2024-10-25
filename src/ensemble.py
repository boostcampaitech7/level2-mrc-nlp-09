import os
import json

"""
This script performs soft voting on multiple nbest prediction files generated from 
a question-answering model. It aggregates the predictions, calculates the mean 
probability for each answer candidate, and selects the answer with the highest 
mean probability using a soft voting strategy.

Process:
1. Reads multiple JSON files from a given folder, each containing nbest predictions.
2. Concatenates the answer candidates from all the files for each question ID.
3. Counts the probability for each unique answer candidate from the combined files.
4. For each question, selects the answer candidate with the highest average probability 
    as the final prediction.
5. Writes the final predictions into a JSON file.

The final output will be a JSON file (`soft_voting_predictions.json`) containing the 
selected answers for each question.

Variables:
    folder_path (str): The path to the folder containing nbest prediction JSON files.
    nbest_files_concat_list (list): A list of concatenated nbest files data.
    answer_candidates_dict (dict): Dictionary to store unique answer candidates 
                                    and their associated probabilities for each question.
    final_ans_dict (dict): Dictionary that holds the final selected answer 
                            for each question after soft voting.

Returns:
    A JSON file `soft_voting_predictions.json` with the final answers.
"""

def main():
    folder_path = "/data/ephemeral/home/donghun/src/nbest_files"
    nbest_files = os.listdir(folder_path)
    nbest_files_concat_list = []
    for file in nbest_files:
        with open(os.path.join(folder_path, file), 'r') as contents:
            data = json.load(contents)
            nbest_files_concat_list.append(data)
    ids = nbest_files_concat_list[0].keys()
    answer_candidates_dict = {}
    for i in ids:
        answer = {}
        answer_candidate_text_01 = []
        num_nbest_files = len(nbest_files_concat_list)
        for f in range(num_nbest_files):
            answer_candidate_text_02 = []
            num_answer_candidates =len(nbest_files_concat_list[f][i])
            for k in range(num_answer_candidates):
                answer_candidate_text_02.append(nbest_files_concat_list[f][i][k]['text'])
            answer_candidate_text_01 += answer_candidate_text_02
        for answer_text in list(set(answer_candidate_text_01)):
            answer[answer_text] = []
        answer_candidates_dict[i] = answer
    for i in ids :
        num_nbest_files = len(nbest_files_concat_list)
        for f in range(num_nbest_files) :
            num_answer_candidates = len(nbest_files_concat_list[f][i])
            for k in range(num_answer_candidates) :
                text = nbest_files_concat_list[f][i][k]['text']
                if text in answer_candidates_dict[i].keys() :
                    answer_candidates_dict[i][text] += [nbest_files_concat_list[f][i][k]['probability']]
    final_ans_dict = {}
    for i in ids:
        voting_dict = {}
        text_list = []
        mean_list = []
        ans_dict = answer_candidates_dict[i]
        for text in ans_dict.keys():
            value_list = ans_dict[text]
            mean = sum(value_list) / len(value_list)
            text_list.append(text)
            mean_list.append(mean)
        voting_dict['text'] = text_list
        voting_dict['mean'] = mean_list
        max_value_index = mean_list.index(max(mean_list))
        final_ans_dict[i] = text_list[max_value_index]
    with open('soft_voting_predictions.json', 'w') as f:
        json.dump(final_ans_dict, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()