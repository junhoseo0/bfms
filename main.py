import tyro

from bfms.data import ExoRLDataset


def main(dataset_dir: str, env: str, collection_method: str):
    exorl_dataset = ExoRLDataset(dataset_dir, env, collection_method)
    num_transitions = exorl_dataset._storage["observation"].shape[0]
    print(f"Dataset {env}/{collection_method} loaded with {num_transitions:,} transitions.")


if __name__ == "__main__":
    tyro.cli(main)
