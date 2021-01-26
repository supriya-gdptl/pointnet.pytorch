import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_loss_acc(loss_acc_file, output_file):
    data = pd.read_csv(loss_acc_file, sep=',')

    train_acc = data[data['mode'] == 'train']['acc']
    train_loss = data[data['mode'] == 'train']['loss']

    test_acc = data[data['mode'] == 'test']['acc']
    test_loss = data[data['mode'] == 'test']['loss']

    x_axis = data[data['mode'] == 'train']['epoch']

    fig,(a1,a2) = plt.subplots(2,1)
    a1.plot(x_axis, train_acc, 'bo-',label='train accuracy')
    a1.plot(x_axis, test_acc,'ro-', label='test accuracy')

    a1.set(xlabel='epochs', ylabel='accuracy')
    a1.legend()

    a2.plot(x_axis, train_loss,'bo-', label='train loss')
    a2.plot(x_axis, test_loss, 'ro-', label='test loss')
    a2.set_title('train and validation loss')
    a2.set(xlabel='epochs', ylabel='loss')
    a2.legend()
    fig.tight_layout()
    plt.savefig(output_file)


if __name__ == '__main__':
    # folder = "/home/supriya/MY_HOME/SFU_PHD/research/SceneGraphNet/data/mp3d_data/my_vico_output"
    import sys
    folder = sys.argv[1]
    loss_file = [fld for fld in os.listdir(folder) if fld.startswith("loss_acc")]
    loss_file = os.path.join(folder, loss_file[0])
    output_file = os.path.join(folder, "loss_plot.png")
    plot_loss_acc(loss_acc_file=loss_file, output_file=output_file)

