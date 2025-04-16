# import wandb
# import torch
# from NCF_model import NCF
# from train import Trainer
# from dataset import DatasetFactory
#
# def train_sweep():
#     # Initialize wandb
#     wandb.init()
#     config = wandb.config
#
#     # Setup device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     config.device = device
#
#     # Setup paths
#     train_path = "/tmp/data/recsys_data_and_test_files/user_item_rating_train.csv"
#     test_path = "/tmp/data/recsys_data_and_test_files/warm_items_rating_prediction_test_format.csv"
#     metadata_path = "/tmp/data/recsys_data_and_test_files/items_metadata.jsonl"
#
#     # Create datasets using factory
#     factory = DatasetFactory(train_path, test_path, metadata_path)
#     train_dataset, val_dataset, test_dataset = factory.create_datasets()
#
#     model = NCF(
#
#     )
#
#     trainer = Trainer(model, train_dataset, val_dataset, config)
#     final_rmse = trainer.train()
#     wandb.log({"final_val_rmse": final_rmse})
#
# if __name__ == "__main__":
#     train_sweep()