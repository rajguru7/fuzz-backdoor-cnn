# Demo

```
python version - 3.9.10
tensorflow version - 2.8.0
```

Files in CIFAR10 folder
```
Model3 - LeNet5
Model4 - STRIP model working with DLFUZZ
Model4_1 - STRIP model not working with DLFUZZ
Model3x - LeNet5 with poisoned data
Model4x - STRIP model with poisoned data
```

## To run

generate adversarial examples for CIFAR10
```
cd CIFAR10

Format:
python3 gen_diff.py [1] [2] [3] [4] [5] [6]

Example: 
python3 gen_diff.py 0 0.25 10 0602 3 model3

#meanings of arguments
[1] -> the list of neuron selection strategies
[2] -> the activation threshold of a neuron
[3] -> the number of neurons selected to cover
[4] -> the folder holding the adversarial examples generated
[5] -> the number of times for mutation on each seed
[6] -> the DL model under test (possible values: model3, model4, model3x, model4x)
```

