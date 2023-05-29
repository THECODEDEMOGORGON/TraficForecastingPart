import argparse
import json
from typing import Any, Dict

import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.preprocessing import LabelEncoder
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE, MAE, SMAPE, MultivariateNormalDistributionLoss

def load_data(file_path: str, data_col: str, id_col: str, target_column: str) -> pd.DataFrame:
    """
    Загружает данные из файла и возвращает DataFrame.

    Args:
        file_path (str): Путь к файлу данных.
        data_col (str): Имя столбца с датами.
        id_col (str): Имя столбца с идентификатором.
        target_column (str): Имя целевого столбца.

    Returns:
        pd.DataFrame: Загруженные данные в виде DataFrame.
    """
    df = pd.read_csv(file_path, parse_dates=True)[[data_col, id_col, target_column]]
    df[id_col] = df[id_col].astype(str)
    return df


def process_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Обрабатывает данные и создает наборы данных и загрузчики.

    Args:
        df (pd.DataFrame): Входные данные в виде DataFrame.
        config (dict): Конфигурационные параметры.

    Returns:
        tuple: Кортеж, содержащий набор данных, загрузчик для обучения и загрузчик для валидации.
    """
    dates_transformer = LabelEncoder()
    df['time_idx'] = dates_transformer.fit_transform(df[config["date_column"]])
    df['time_idx'] += 1

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=config["target_column"],
        categorical_encoders={config["id_column"]: NaNLabelEncoder().fit(df[config["id_column"]])},
        group_ids=[config["id_column"]],
        static_categoricals=[config["id_column"]],
        time_varying_unknown_reals=[config["target_column"]],
        max_encoder_length=config["context_length"],
        max_prediction_length=config["prediction_length"],
        allow_missing_timesteps=False
    )

    train_dataloader = dataset.to_dataloader(
        train=True,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        drop_last=True,
        batch_sampler="synchronized"
    )

    val_dataloader = dataset.to_dataloader(
        train=False,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        drop_last=True,
        batch_sampler="synchronized"
    )

    return dataset, train_dataloader, val_dataloader


def fit_model(train_dataloader: torch.utils.data.dataloader.DataLoader,
              val_dataloader: torch.utils.data.dataloader.DataLoader,
              subset_data_loader: torch.utils.data.dataloader.DataLoader,
              config: Dict[str, Any]) -> DeepAR:
    """
    Обучает модель DeepAR.

    Args:
        train_dataloader (torch.utils.data.dataloader.DataLoader): Загрузчик данных для обучения.
        val_dataloader (torch.utils.data.dataloader.DataLoader): Загрузчик данных для валидации.
        subset_data_loader (torch.utils.data.dataloader.DataLoader): Загрузчик данных для подмножества.
        config (Dict[str, Any]): Конфигурационные параметры.

    Returns:
        DeepAR: Обученная модель DeepAR.
    """
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator=config["accelerator"],
        enable_model_summary=True,
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=[pl.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )],
        limit_train_batches=300,
        limit_val_batches=100,
        enable_checkpointing=True,
        logger=config["logger"]
    )

    net = DeepAR.from_dataset(
        train_dataloader.dataset,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        rnn_layers=config["rnn_layers"],
        optimizer=config["optimizer"]
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    ar_predictions = net.predict(subset_data_loader,
                                 trainer_kwargs=dict(accelerator=config["accelerator"]),
                                 return_y=False)

    rmse = RMSE()(ar_predictions, val_dataloader.dataset["internet"])
    print(f"RMSE: {rmse}")

    return net


def make_infer_loader(df: pd.DataFrame, dataset: TimeSeriesDataSet, config: Dict[str, Any]) -> torch.utils.data.dataloader.DataLoader:
    """
    Создает загрузчик данных для инференса.

    Args:
        df (pd.DataFrame): Входные данные в виде DataFrame.
        dataset (TimeSeriesDataSet): Набор данных.
        config (Dict[str, Any]): Конфигурационные параметры.

    Returns:
        torch.utils.data.dataloader.DataLoader: Загрузчик данных для инференса.
    """
    test_df = df[df.time_idx > df.time_idx.max() - config['context_length']]
    start_idx = test_df.time_idx.max() + 1
    for square_id in test_df.square_id.unique():
        data = []
        for d in range(config['prediction_length']):
            data.append({
                "square_id": square_id,
                "time_idx": start_idx + d,
                "internet": 0
            })
        data = pd.DataFrame(data)
        test_df = pd.concat([test_df, data])

    test_df = test_df.reset_index()

    infer_dataset = TimeSeriesDataSet.from_dataset(dataset, test_df, predict_mode=True)

    infer_loader = infer_dataset.to_dataloader(batch_size=test_df[config['id_column']].nunique())

    return infer_loader


def make_forecasts(net: DeepAR, infer_loader: torch.utils.data.dataloader.DataLoader) -> torch.Tensor:
    """
    Создает прогнозы с использованием обученной модели.

    Args:
        net (DeepAR): Обученная модель DeepAR.
        infer_loader (torch.utils.data.dataloader.DataLoader): Загрузчик данных для инференса.

    Returns:
        torch.Tensor: Прогнозы модели.
    """
    x, y = next(iter(infer_loader))
    return net.forward(x)


def main(config: Dict[str, Any]):
    """
    Главная функция программы для обучения и создания прогнозов с использованием модели DeepAR.

    Args:
        config (Dict[str, Any]): Конфигурационные параметры.
    """
    data = load_data(config["data_file"],
                     config['date_column'],
                     config['id_column'],
                     config['target_column'])
    print(data.shape)
    print(data.info())
    training, train_dataloader, val_dataloader = process_data(data, config)

    if config["fit_model"]:
        net = fit_model(train_dataloader, val_dataloader, subset_data_loader, config)
        net.save(config["model_path"])
    else:
        net = DeepAR.load_from_checkpoint(config["model_path"])

    infer_loader = make_infer_loader(data, training, config)
    print(make_forecasts(net, infer_loader))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Программа для обучения и создания прогнозов с использованием модели DeepAR")
    parser.add_argument("config", help="Путь к файлу конфигурации")

    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as file:
        config = json.load(file)

    main(config)
