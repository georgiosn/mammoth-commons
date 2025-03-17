from mammoth import testing
from catalogue.dataset_loaders.uci_csv import data_uci
from catalogue.model_loaders.no_model import no_model
from catalogue.metrics.interactive_augmentation_report import interactive_augmentation_report


def test_interactive_augmentation_report():
    with testing.Env(no_model, data_uci, interactive_augmentation_report) as env:
        dataset_name = "credit"
        target = "Y"
        dataset = env.data_uci(dataset_name=dataset_name, target=target)
        model = env.no_model()
        sensitive = ["X2", "X3"]
        html_result = env.interactive_augmentation_report(dataset, model, sensitive=sensitive)
        html_result.show()


if __name__ == "__main__":
    test_interactive_augmentation_report()
