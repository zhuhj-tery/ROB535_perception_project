import glob
import csv
from model import CNN
import checkpoint
import torch, torchvision
import torchvision.transforms as transforms
import os, shutil
from glob import glob
from tqdm import tqdm
from train import evaluate_loop


def evaluate(config, model):
    # create temporary folder to store images
    tmp_path = "tmp_test"
    if os.path.exists(tmp_path) and os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path)
    dest_folder = tmp_path+"/test/0"
    os.makedirs(dest_folder)
    test_data_path = config["dataset_path"]
    img_files = glob(test_data_path+'/*/*_image.jpg')
    img_name_list = []
    for i in tqdm(range(len(img_files))):
        f_name = img_files[i].replace(test_data_path+"/", "")
        f_name = f_name.replace("_image.jpg", "")
        img_name_list.append(f_name)
        shutil.copy(img_files[i], dest_folder)
        dst_file = os.path.join(dest_folder, img_files[i].split("/")[-1])
        new_dst_file_name = os.path.join(dest_folder, str(i).zfill(4)+".jpg")
        os.rename(dst_file, new_dst_file_name)
    print("finish creating temporary folder to store test images")
    # create valication set
    transform_list = [
        # resize the image to 32x32x3
        transforms.Resize(32),
        # convert image to PyTorch tensor
        transforms.ToTensor(),
        # normalize the image (use self.x_mean and self.x_std)
        transforms.Normalize(config["x_mean"], config["x_std"]),
    ]
    trans = transforms.Compose(transform_list)
    eval_set = torchvision.datasets.ImageFolder(os.path.join(tmp_path, 'test'), transform=trans)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=config["batch_size"], shuffle=False)
    print("finish creating data loader")
    # load latest checkpoint
    model, start_epoch, stats = checkpoint.restore_checkpoint(model, config['ckpt_path'], force=False)
    print("finish loading model")
    # evaluate
    _, y_pred, _ = evaluate_loop(eval_loader, model)
    print("finish evaluating")
    # store results
    with open(config["result_path"], "w") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["guid/image", "label"])
        for i in range(len(y_pred)):
            re = [img_name_list[i], int(y_pred[i])]
            csv_writer.writerow(re)
    print("finish logging results")


if __name__ == '__main__':
    # evaluate
    config_eval = {
        'dataset_path': 'test',
        "x_mean": [0.36210934, 0.35751695, 0.34721091],
        "x_std": [0.25470638, 0.2477791, 0.23893978],
        "batch_size": 4,
        'ckpt_path': 'checkpoints/cnn',
        "result_path": "result_test.csv"
    }
    evaluate(config_eval, CNN())