# continual-uda

This repository compares various unsupervised domain adaptation algorithms (UDA) for continual gradual domain adaptation.
Implemented algorithms include:
* Joint Adaptation Network
* Domain Adversarial Neural Network
* Class-balanced Self-training
* Confidence-regularized Self-training
* Cycle Self-trianing

To run the experiments, go to rotate_mnist/ and run
```
python main.py --data_dir path/to/root_datadir/ --method cbst
```
