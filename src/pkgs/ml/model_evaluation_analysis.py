# """
# #++++++++++++++++++++++++++++++++++++++++++++++

#     Project: Part of final project for Georgia Tech Institute of Technology course DL, CS 7643. 

#     Totality of this code is non-proprietary and may be used at will. 

# #++++++++++++++++++++++++++++++++++++++++++++++


# Description: 

# @brief Defines model training results analysis steps (saving loss, accuracy training curves for example).

# @author: Greg Zdor (gzdor@icloud.com)

# @date Date_Of_Creation: 4/16/2023 

# @date Last_Modification 4/16/2023 

# No Copyright - use at will

# """

# System level imports 
import os 

# Data science imports 
import pandas as pd 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Visualization packages 
import matplotlib.pyplot as plt 
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def color_legend_texts(leg):
    """Color legend texts based on color of corresponding lines"""
    for line, txt in zip(leg.get_lines(), leg.get_texts()):
        txt.set_color(line.get_color())  


def model_evaluation(current_logs_csv: str, current_experiment_path: str, n_epochs: int) -> None : 
    
    try: 
        
        # Load current training results
        df = pd.read_csv(current_logs_csv)
        
        # Note - the NaN values in ths dataframe are because it logs train acc/loss every n steps within an epoch, 
        # but logs other metrics less frequently, so it assigns NaNs as placeholders 
        
        tr_acc = df[df['avg_tr_acc'].notnull()]['avg_tr_acc']
        val_acc = df[df['avg_val_acc'].notnull()]['avg_val_acc']

        tr_loss = df[df['avg_tr_loss'].notnull()]['avg_tr_loss']
        val_loss = df[df['avg_val_loss'].notnull()]['avg_val_loss']
        
        # Accuracy curve
        plt.figure(figsize=(10,8))
        plt.title(f'Training and Validation Accuracy Across {n_epochs} Epochs', fontsize = 18)
        plt.xlabel('Epochs', fontsize = 14)
        plt.ylabel(f'Performance ("accuracy")', fontsize = 16)
        ax = plt.gca()

        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(tr_acc, label = ' train', color = color)
        plt.plot(val_acc, '--', label = ' val', color = color)

        legend = plt.legend(bbox_to_anchor = (1.05, 1.0), loc = 'upper left')

        ax = plt.gca()
        ax.set_ylim([0, 1])
        plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        color_legend_texts(legend)
        plt.tight_layout()
        save_path = os.path.join(current_experiment_path, "loss_curve.png")
        plt.savefig(save_path)
        plt.close()

        # Loss curve
        plt.figure(figsize=(10,8))
        plt.title(f'Training and Validation Loss Across {n_epochs} Epochs', fontsize = 18)
        plt.xlabel('Epochs', fontsize = 14)
        plt.ylabel(f'Performance ("Loss")', fontsize = 16)
        ax = plt.gca()

        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(tr_loss, label = ' train', color = color)
        plt.plot(val_loss, '--', label = ' val', color = color)

        legend = plt.legend(bbox_to_anchor = (1.05, 1.0), loc = 'upper left')

        ax = plt.gca()
        plt.grid(axis = 'y', linestyle = '--', linewidth = 0.5)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        color_legend_texts(legend)
        plt.tight_layout()
        save_path = os.path.join(current_experiment_path, "loss_curve.png")
        plt.savefig(save_path)
        plt.close()


        #TODO IMPLEMENT THE FOLLOWING 

        
        # # Run predictions 
        # predictor = pl.Trainer()
        # predictions_all_batches = predictor.predict(model, dataloaders = data_module)
        
        
        # # Create  confusion matrix
        # predictions = None 
        # labels = None 
        # class_names = ['device1', 'device2'] #TODO parse from dataset 

        # confusion_matx = confusion_matrix(labels, predictions, labels = class_names)

        # confusion_matrix_view = ConfusionMatrixDisplay(confusion_matrix = confusion_matx, 
        #                                             display_labels = class_names)

        # confusion_matrix_view.plot() 

        # plt.show() 
        
    except: 
        
        print(f'Could not find current experiment csv results on disk, namely file \n\n {current_logs_csv}\n\nskipping analysis')