import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(trLoss: list, valLoss: list, title='Loss over Epochs', root='./results', model_name=None, tag: str = "") -> str:
    """
    Plots the training and validation loss over epochs.

    Parameters:
    - trLoss: List of training loss values.
    - valLoss: List of validation loss values.
    - title: Title of the plot (default is 'Loss over Epochs').
    - root: Root directory to save the plot.
    - model_name: Optional name of the model to include in the title.
    - tag: Optional tag to include in the filename.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(trLoss, label='Training Loss', color='blue')
    plt.plot(valLoss, label='Validation Loss', color='orange')
    if model_name:
        plt.title(f'{title} - {model_name}')
    else:
        plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    outpath = f"{root}\\{title.replace(' ', '_')}_{tag}.png"
    plt.savefig(outpath)

    # also save losses to a csv file for later analysis
    loss_df = pd.DataFrame({'Training Loss': trLoss, 'Validation Loss': valLoss})
    loss_df.to_csv(f"{root}\\losses_{tag}.csv", index=False)

    return outpath