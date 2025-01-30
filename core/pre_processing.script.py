import argparse

import pandas as pd
import spacy

from core.dataset_sampler import BggDatasetRandomBalancedSampler
from core.pre_processing import PreProcessingService, DatasetGeneration

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--corpus_file', type=str, default="../data/corpus.csv", help="Path to the corpus file."
    )
    args_parser.add_argument(
        '--game_information_file', type=str,
        default="../resources/2024-08-18.csv", help="Path to the game information file (CSV)."
    )
    args_parser.add_argument(
        '--target_path', type=str,
        default="../output/dataset/default-pre-processed", help="Path to store the processed datasets."
    )
    args_parser.add_argument(
        '--random_states', type=str,
        default='2, 8, 97', help="Comma-separated list of random states."
    )

    args = args_parser.parse_args()
    print(f"Running with args: {args}")

    corpus_file = args.corpus_file
    random_states = list(map(int, args.random_states.split(',')))

    game_names = pd.read_csv(args.game_information_file)['Name']
    ambiguous_names = []  # To ignore title Games. There might be some?
    # Well, just numbers for a game name is not really ideal for us so we don't consider them game titles.
    game_names = game_names[~game_names.str.isdecimal()]
    game_names = game_names[~game_names.isin(ambiguous_names)]

    game_names = pd.concat([game_names, pd.Series(["Quick", "Catan"])], ignore_index=True)
    print(f"We have a total of: {len(game_names)} different game titles.")

    # Game names extraction:
    nlp = spacy.load("en_core_web_sm")  # We use small as we don't need anything over the top.
    game_names = game_names.swifter.apply(lambda x: nlp(x)).tolist()

    pipeline = PreProcessingService.default_pipeline(game_names, args.target_path)
    # pipeline = PreProcessingService.full_pipeline(game_names, args.target_path)
    # Pre-processing main function call.
    combinations: [DatasetGeneration] = [
        # DatasetGeneration(pipeline, 50000, BggDatasetRandomBalancedSampler(10000, corpus_file, random_states[0])),
        # DatasetGeneration(pipeline, 100000, BggDatasetRandomBalancedSampler(20000, corpus_file, random_states[1])),
        # DatasetGeneration(pipeline, 200000, BggDatasetRandomBalancedSampler(40000, corpus_file, random_states[2])),
        DatasetGeneration(pipeline, 80000, BggDatasetRandomBalancedSampler(10000, corpus_file, random_states[0])),
        DatasetGeneration(pipeline, 200000, BggDatasetRandomBalancedSampler(40000, corpus_file, random_states[1])),
    ]

    print('We will generate a total of:', len(combinations), ' datasets')
    for combination in combinations:
        pipeline, target_size, sampler = combination
        name = f"{int(target_size / 1000)}k"
        print("Generated dataset will be stored in file of prefix: " + name)
        file = pipeline.pre_process_corpus(combination.target_size, sampler, name)
        print(f"Generated dataset in file: {file}")
