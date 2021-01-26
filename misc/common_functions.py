import os
import sys
from pathlib import Path


def replace_file_extension(input_file):
    """
    read {train/val/test/trainval}.txt file (downloaded from https://github.com/fxia22/pointnet.pytorch/issues/52#issuecomment-561013797)
    These files contain list of .ply models. ModelNet40 dataset now contains .off files instead of .ply
    This methods replaces the model file extensions to .off
    :param input_file: string: location of {train/val/test/trainval}.txt
    :return:
    """
    path = Path(input_file)
    text = path.read_text()
    if ".ply" in text:
        text = text.replace(".ply", ".off")
        # write in the same file
        path.write_text(text)
        print(f"Extension .ply replaced with .off in file {input_file}")


def write_log(logfile, msg):
    logfile.write(f'{msg}\n')
    print(msg)
    logfile.flush()


def write_loss_acc(loss_acc_file, mode, epoch, loss, acc):
    loss_acc_file.write(f'{mode},{epoch},{loss:.6f},{acc:.6f}\n')
    loss_acc_file.flush()

def main():
    # if __name__ == '__main__':
    dataset_folder = sys.argv[1]
    replace_file_extension(os.path.join(dataset_folder, 'train.txt'))
    replace_file_extension(os.path.join(dataset_folder, 'val.txt'))
    replace_file_extension(os.path.join(dataset_folder, 'test.txt'))
    replace_file_extension(os.path.join(dataset_folder, 'trainval.txt'))
