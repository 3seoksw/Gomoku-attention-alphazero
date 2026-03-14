import torch
from models.policy_value_model import PolicyValueModel
from trainer.trainer import Trainer
from datetime import datetime


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f" {device} loaded\n")

    model = PolicyValueModel()

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = Trainer(
        model=model,
        name="alphazero",
        device=device,
        lr=1e-4,
        n_simulations=200,
        log_dir=f"runs/{time}",
        log_every=20,
        eval_every=50,
    )
    trainer.fit(n_episodes=3000, n_evals=30, verbose=True)


if __name__ == "__main__":
    main()
