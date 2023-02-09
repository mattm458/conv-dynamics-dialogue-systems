from model.sequential import SequentialConversationModel
from model.windowed import WindowedConversationModel


def get_model(model_config, training_config, feature_names):
    if model_config["type"] == "windowed":
        return WindowedConversationModel(
            window_size=model_config["window_size"],
            training_window_mode=model_config["training_window_mode"],
            lr=training_config["lr"],
            feature_names=feature_names,
            **model_config["args"],
        )
    elif model_config["type"] == "sequential":
        return SequentialConversationModel(
            lr=training_config["lr"],
            feature_names=feature_names,
            **model_config["args"],
        )
    else:
        raise Exception(f"Unknown model type {model_config['type']}")
