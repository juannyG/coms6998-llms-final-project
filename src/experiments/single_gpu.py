from torch.utils.data import DataLoader

from models.simple import SimpleTransformerDecoder
from datasets.synthetic import SyntheticDataset


def run_single_gpu_experiment(conf):
    model = SimpleTransformerDecoder(
        conf["vocab_size"],
        conf["d_model"],
        conf["n_heads"],
        conf["n_layers"],
        conf["d_ff"],
        conf["seq_len"]
    ).to("cuda")

    print(f"Using configuration: {conf}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    dataset = SyntheticDataset(
        n_samples=10000,
        seq_len=conf["seq_len"],
        vocab_size=conf["vocab_size"]
    )
    loader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )


