from cdmodel.model.sequential_manifest_v1 import SequentialConversationModel
from cdmodel.model.windowed import WindowedConversationModel


def get_model(
    dataset_config: dict,
    model_config: dict,
    training_config: dict,
    feature_names: list[str],
    SequentialClass=SequentialConversationModel,
):
    if model_config["type"] == "windowed":
        return WindowedConversationModel(
            window_size=model_config["window_size"],
            training_window_mode=model_config["training_window_mode"],
            lr=training_config["lr"],
            feature_names=feature_names,
            **model_config["args"],
        )
    elif model_config["type"] == "sequential":
        return SequentialClass(
            lr=training_config["lr"],
            features=feature_names,
            zero_pad=dataset_config["zero_pad"],
            **model_config["args"],
        )
    else:
        raise Exception(f"Unknown model type {model_config['type']}")
