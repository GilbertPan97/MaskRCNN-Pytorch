import os
import datetime
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments
import matplotlib.pyplot as plt


def plot_loss_and_lr(train_loss, learning_rate, save_dir):
    """
    Plots the training loss and learning rate over time and saves the plot as an image file.

    Parameters:
    train_loss (list): A list of training loss values.
    learning_rate (list): A list of learning rate values.
    save_dir (str): Directory where the plot image will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    if not train_loss or not learning_rate:
        raise ValueError("Both train_loss and learning_rate lists must be non-empty.")
    if len(train_loss) != len(learning_rate):
        raise ValueError("train_loss and learning_rate lists must have the same length.")

    try:
        x = list(range(len(train_loss)))

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(x, train_loss, marker='o', linestyle='-', color='crimson', label='Loss', linewidth=2)
        ax1.set_xlabel("Step", fontsize=14)
        ax1.set_ylabel("Loss", color='crimson', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='crimson')
        ax1.set_title("Training Loss and Learning Rate Over Time", fontsize=16, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.7)

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, marker='x', linestyle='--', color='navy', label='Learning Rate', linewidth=2)
        ax2.set_ylabel("Learning Rate", color='navy', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='navy')
        ax2.set_xlim(0, len(train_loss) - 1)

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=12)

        fig.subplots_adjust(right=0.85)
        fig_name = 'loss_and_lr_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        save_path = os.path.join(save_dir, fig_name)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved loss and learning rate curve at {save_path}")

    except Exception as e:
        print(f"Failed to save loss and learning rate curve: {e}")


def plot_map(mAP, save_dir):
    """
    Plots the mAP curve and saves it as an image file.

    Parameters:
    mAP (list): A list of mAP values.
    save_dir (str): Directory where the plot image will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    if not mAP:
        raise ValueError("The mAP list is empty. Please provide valid data to plot.")

    try:
        x = list(range(len(mAP)))

        plt.figure(figsize=(10, 6))
        plt.plot(x, mAP, marker='o', linestyle='-', color='darkorange', label='mAP', linewidth=2)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('mAP', fontsize=14)
        plt.title('Evaluation mAP Over Epochs', fontsize=16, fontweight='bold')
        plt.xlim(0, len(mAP) - 1)
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=12)

        fig_name = 'mAP_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        save_path = os.path.join(save_dir, fig_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Successfully saved mAP curve at {save_path}")

    except Exception as e:
        print(f"Failed to save mAP curve: {e}")

