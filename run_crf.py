from src.crf import MLPCRFClassifier, mlpcrf_loss, train_crf
from src.data import batch_dataset, load_data, split_data, time_series_split
from src.utils import ensure_reproducibility, print_to_file


def prepare_dataset():
    df = load_data()
    dataset = split_data(
        df,
        without_cols=[
            "Timestamp",
            "Light1",
            "Light2",
            "Light3",
            "Light4",
        ],
        scale="normalize",
    )
    seq_dataset = batch_dataset(dataset, sequence_length=25)
    return seq_dataset


def run_training(dataset, name):
    mlpcrf_model = MLPCRFClassifier([64], 4, 12)
    train_crf(
        model=mlpcrf_model,
        loss_fn=mlpcrf_loss,
        dataset=dataset,
        n_epochs=50,
        lr=0.001,
        batch_size=16,
        use_mask=False,
        patience=50,
        minimum=150,
        name=name,
        save_figs=True,
    )


def main():
    dataset = prepare_dataset()
    ensure_reproducibility()

    name = "second_one"

    with print_to_file(name):
        run_training(dataset, name)


if __name__ == "__main__":
    main()
