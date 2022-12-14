import argparse
import os
from tqdm import tqdm
import pickle
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from model import Encoder, Classifier
from utils import get_device, set_random_seeds, MyRandomRotation
from shared import norevisit_stages, revisit_stages
import method


def train_epoch(encoder, classifier, device, train_loader, optimizer):
    encoder.train()
    classifier.train()

    total_train_loss = 0
    total_size = 0
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(classifier(encoder(data)), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        batch_size = data.shape[0]
        total_train_loss += loss.item() * batch_size
        total_size += batch_size

    total_train_loss /= total_size
    return total_train_loss


@torch.no_grad()
def test_epoch(encoder, classifier, device, test_loader):
    encoder.eval()
    classifier.eval()

    total_test_loss = 0
    total_correct = 0
    total_size = 0

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        output = F.log_softmax(classifier(encoder(data)), dim=1)
        loss = F.nll_loss(output, target, reduction='sum')
        total_test_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        total_correct += pred.eq(target.view_as(pred)).sum().item()
        total_size += data.shape[0]

    total_test_loss /= total_size
    total_correct /= total_size

    return total_test_loss, total_correct

def get_tgt_loader(stage, start, end, degree_sum, args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        MyRandomRotation(stage)
    ])
    tgt_train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                       transform=transform)
    tgt_train_dataset = torch.utils.data.Subset(tgt_train_dataset, range(len(tgt_train_dataset) * start // degree_sum, len(tgt_train_dataset) * end // degree_sum))
    tgt_val_dataset = datasets.MNIST(args.data_dir, train=False,
                                      transform=transform)
    tgt_val_dataset = torch.utils.data.Subset(tgt_val_dataset, range(len(tgt_val_dataset) * start // degree_sum, len(tgt_val_dataset) * end // degree_sum))
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_dataset, batch_size=args.batch_size, shuffle=True)
    tgt_val_loader = torch.utils.data.DataLoader(tgt_val_dataset, batch_size=args.batch_size, shuffle=False)
    tgt_all_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([tgt_train_dataset, tgt_val_dataset]), batch_size=512, shuffle=False) # this is for the test phase of prequential evaluation

    return tgt_train_loader, tgt_val_loader, tgt_all_loader


def main(args):
    device = get_device(args.gpuID)
    set_random_seeds(args.model_seed)

    if args.revisit:
        stages = revisit_stages
        revisit = "revisit"
    else:
        stages = norevisit_stages
        revisit = "norevisit"

    replay = "replay" if args.replay else "noreplay"
    encoder = Encoder().to(device)
    classifier = Classifier().to(device)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        MyRandomRotation(stages[0])
    ])
    degree_sum = sum([s[1] - s[0] for s in stages])
    degree_cdf = [0]
    for stage in stages:
        degree_cdf.append(degree_cdf[-1] + stage[1] - stage[0])
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True,
                                   transform=transform)
    train_dataset = torch.utils.data.Subset(train_dataset, range(len(train_dataset) * degree_cdf[0] // degree_sum, len(train_dataset) * degree_cdf[1] // degree_sum))

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * 0.9), len(train_dataset) - int(len(train_dataset) * 0.9)])
    test_dataset = datasets.MNIST(args.data_dir, train=False,
                                  transform=transform)
    test_dataset = torch.utils.data.Subset(test_dataset, range(len(test_dataset) * degree_cdf[0] // degree_sum, len(train_dataset) * degree_cdf[1] // degree_sum))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # train(encoder, classifier, train_loader, val_loader, device, args)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)

    for e in range(1, args.train_epochs + 1):
        train_loss = train_epoch(encoder, classifier, device, train_loader, optimizer)
        val_loss, val_acc = test_epoch(encoder, classifier, device, val_loader)
        print(
            f'Epoch:{e}/{args.train_epochs} Train Loss: {round(train_loss, 3)}, Val Loss: {round(val_loss, 3)}, Val Accuracy: {round(val_acc, 3)}')

    test_loss_list, test_acc_list = [], []
    test_loss, test_acc = test_epoch(encoder, classifier, device, test_loader)
    # train source
    print("Source stage:", stages[0])
    print(f"Source domain test loss {round(test_loss, 3)} test accuracy {round(test_acc, 3)}")
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

    if args.method == "jan":
        adapter = method.JointAdaptationNetwork(encoder, classifier, train_loader, val_loader, device, args)
    elif args.method == "dann":
        adapter = method.DomainAdversarialNetwork(encoder, classifier, train_loader, val_loader, device, args)
    elif args.method == "cst":
        adapter = method.CycleSelfTrainer(encoder, classifier, train_loader, val_loader, device, args)
    elif args.method == "cbst" or args.method == "crst":
        adapter = method.ClassBalancedSelfTrainer(encoder, classifier, train_loader, val_loader, device, args)
    elif args.method == "cbst-tgt" or args.method == "crst-tgt":
        adapter = method.ClassBalancedSelfTargetTrainer(encoder, classifier, device, args)


    for i, stage in enumerate(stages[1:]):
        os.makedirs(os.path.join(args.ckpt_dir, revisit, str(stage[0]) + "_" + str(stage[1])) , exist_ok=True)
        torch.save({"encoder": encoder,
                    "classifier": classifier},
                   os.path.join(args.ckpt_dir, revisit, str(stage[0]) + "_" + str(stage[1]), args.method + "_" + str(args.model_seed) + "_" + replay + ".pt"))

        tgt_train_loader, tgt_val_loader, tgt_all_loader = get_tgt_loader(stage, degree_cdf[i+1], degree_cdf[i+2], degree_sum, args)
        tgt_test_loss, tgt_test_acc = test_epoch(encoder, classifier, device, tgt_all_loader)
        test_loss_list.append(tgt_test_loss)
        test_acc_list.append(tgt_test_acc)
        print(f"Current stage: {stage} Prequential test loss {round(tgt_test_loss, 3)} test acc {round(tgt_test_acc, 3)}")

        if i == len(stages) - 2:
            break

        if args.method == "dann":
            lambda_coeff_list = [0.1, 0.3, 0.5]
            adapter.adapt(tgt_train_loader, tgt_val_loader, lambda_coeff_list, i, args)
            encoder, classifier = adapter.get_encoder_classifier()

        elif args.method == "jan":
            lambda_coeff_list = [0.5, 1, 5]
            adapter.adapt(tgt_train_loader, tgt_val_loader, lambda_coeff_list, i, args, test_epoch_fn=test_epoch)
            encoder, classifier = adapter.get_encoder_classifier()
        elif args.method == "cst":
            lambda_coeff_list = [0.1, 0.5, 1]
            adapter.adapt(tgt_train_loader, tgt_val_loader, lambda_coeff_list, i, args)
            encoder, classifier = adapter.get_encoder_classifier()
        elif args.method == "cbst":
            reg_weight_list = [0]
            adapter.adapt(tgt_train_loader, tgt_val_loader, reg_weight_list, i, args)
            encoder, classifier = adapter.get_encoder_classifier()
        elif args.method == "crst":
            reg_weight_list = [0.5]
            adapter.adapt(tgt_train_loader, tgt_val_loader, reg_weight_list, i, args)
            encoder, classifier = adapter.get_encoder_classifier()
        elif args.method == "cbst-tgt":
            reg_weight_list = [0]
            adapter.adapt(tgt_train_loader, tgt_val_loader, reg_weight_list, i, args)
            encoder, classifier = adapter.get_encoder_classifier()
        elif args.method == "crst-tgt":
            reg_weight_list = [0.5]
            adapter.adapt(tgt_train_loader, tgt_val_loader, reg_weight_list, i, args)
            encoder, classifier = adapter.get_encoder_classifier()
        elif args.method == "fixed":
            pass



    print("Test Loss List:", test_loss_list, "Avg Loss:", sum(test_loss_list) / len(test_loss_list))
    print("Test Acc List:", test_acc_list, "Avg Acc:", sum(test_acc_list) / len(test_acc_list))

    os.makedirs(os.path.join(args.result_dir, revisit), exist_ok=True)
    with open(os.path.join(args.result_dir, revisit, args.method + "_" + str(args.model_seed) + "_" + replay + "_loss_list"), "wb") as fp:
        pickle.dump(test_loss_list, fp)
    with open(os.path.join(args.result_dir, revisit, args.method + "_" + str(args.model_seed) + "_" + replay + "_acc_list"), "wb") as fp:
        pickle.dump(test_acc_list, fp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--revisit', dest='revisit', help="whether test data include previous rotation angles", action='store_true')
    parser.set_defaults(revisit=False)
    parser.add_argument("--data_dir", type=str, help="path to data directory")
    parser.add_argument("--log_dir", type=str, help="path to log directory", default="runs")
    parser.add_argument("--ckpt_dir", type=str, help="path to model checkpoints", default="checkpoints")
    parser.add_argument("--result_dir", type=str, help="path to experiment results", default="results")
    parser.add_argument("--method", type=str, help="method for domain adaptation")
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--batch_size", type=int, help="batch size", default=512)
    parser.add_argument("--train_epochs", type=int, help="number of training epochs", default=10)
    parser.add_argument("--adapt_epochs", type=int, help="number of adaptation epochs", default=100)
    parser.add_argument("--adapt_lr", type=float, help="learning rate during adaptation", default=1e-3)
    parser.add_argument('--replay', dest='replay', action='store_true')
    parser.set_defaults(replay=False)
    parser.add_argument("--buffer_size", type=int, help="size of buffer", default=1024)
    parser.add_argument("--replay_batch_size", type=int, help="replay batch size", default=256)
    parser.add_argument("--model_seed", type=int, help="seed for random number generator", default=42)
    parser.add_argument("--gpuID", type=int, help="which device to use", default=0)
    args = parser.parse_args()

    print(args)
    main(args)
