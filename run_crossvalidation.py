import argparse
from run_single import train as single_train, get_args
from run_double import train as double_train
from neural_net.cnn_configurations import validation_dict
import sys

losses = ['bcemse', 'dicemse', 'bdmse', 'bsmse', 'siousiou', 'sioumse', 'bcef1mse']

def main():
    ls = int(sys.argv[1])
    del sys.argv[1]
    args = get_args()
    for k in validation_dict.keys():
        for seed in [1,2,3]:
            for model in ['unet', 'segnet', 'nestedunet', 'attentionunet']:
                args.model_name = model
                args.seed = seed
                args.key = k
                args.losses = losses[ls]
                double_train(args)
                print("\n\n\n")
            
if __name__ == '__main__':
    main()
    
