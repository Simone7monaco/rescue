import argparse
from pathlib import Path
from run_single import train as single_train, get_args
from run_double import train as double_train
from neural_net.cnn_configurations import validation_dict
import sys

losses = ['bcemse', 'dicemse', 'bdmse', 'bsmse', 'siousiou', 'sioumse', 'bcef1mse']

def main():
    if len(sys.argv) > 1:
        ls = int(sys.argv[1]) 
        del sys.argv[1]
    else:
        ls = None

    args = get_args()
    for seed in [1,2,3]:
        for model in ['unet', 'segnet', 'nestedunet', 'attentionunet'][::-1]:
            for k in list(validation_dict.keys()):
                for l in losses:
                    args.model_name = model
                    args.seed = seed
                    args.key = k
                    args.losses = l #losses[ls] if ls is not None else None
    #                 args.losses = 'bcemse'
                    name = f'test_double-{args.model_name}_{args.key}_{args.losses}_{args.seed}'.lower()
#                     if not Path(f"../data/new_ds_logs/Propaper/legion/{name}").exists():
#                         continue

    #                 print(next(Path(f"../data/new_ds_logs/Propaper/legion/{name}").glob('reg*best*')))

                    print(f'>> run_double {" ".join([f"--{k}={v}" for k, v in vars(args).items()])}\n')
                    double_train(args)
    #                 single_train(args)
                    print("\n\n\n")
            
if __name__ == '__main__':
    main()
    



# import argparse
# from run_single import train as single_train, get_args
# from run_double import train as double_train
# from neural_net.cnn_configurations import validation_dict
# import sys

# losses = ['bcemse', 'dicemse', 'bdmse', 'bsmse', 'siousiou', 'sioumse', 'bcef1mse']

# def main():
#     if len(sys.argv) > 1:
#         ls = int(sys.argv[1]) 
#         del sys.argv[1]
#     else:
#         ls = None

#     args = get_args()
# #     for seed in [1,2,3]:
#     for seed in [1]:
#         for k in list(validation_dict.keys())[::-1]:
# #             for model in ['unet', 'segnet', 'nestedunet']:
#             for model in ['attentionunet']:
#                 args.model_name = model
#                 args.seed = seed
#                 args.key = k
#                 args.losses = losses[ls] if ls is not None else None
# #                 args.losses = 'bcemse'
#                 print(f'>> run_double {" ".join([f"--{k}={v}" for k, v in vars(args).items()])}\n')
#                 double_train(args)
# #                 single_train(args)
#                 print("\n\n\n")
            
# if __name__ == '__main__':
#     main()
    
