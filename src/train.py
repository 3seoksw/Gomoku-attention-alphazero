import torch
from models.policy_value_model import PolicyValueModel
from models.attn_model import AttnPolicyValue
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
    model = AttnPolicyValue(n_blocks=1)

    time = datetime.now().strftime("%Y%m%d_%H%M%S")
    trainer = Trainer(
        model=model,
        device=device,
        lr=3e-4,
        n_simulations=200,
        log_dir=f"runs/{time}",
        log_every=10,
        eval_every=50,
        baseline_dir="runs/alphazero/best_model.pth",
    )
    trainer.fit(n_episodes=5000, n_evals=40, verbose=True)


if __name__ == "__main__":
    main()
