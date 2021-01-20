import parser
import argparse

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="data", type=str, metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--log_path', type=str, default="train_log", help="log path for saving models and logs")
    parser.add_argument('--arch', metavar='ARCH', default='resnet50',
                        type=str,
                        help='model architecture: (default: resnet50)')
    parser.add_argument('--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay of SGD solver')
    parser.add_argument('-p', '--print_freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--world_size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training,rank of total threads, 0 to args.world_size-1')
    parser.add_argument('--dist_url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', type=int, default=1,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument("--nodes_num", type=int, default=1, help="number of nodes to use")

    # Baseline: moco specific configs:
    parser.add_argument('--moco_dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco_k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco_t', default=0.2, type=float,
                        help='softmax temperature (default: 0.2)')
    parser.add_argument('--mlp', type=int, default=1,
                        help='use mlp head')
    parser.add_argument('--cos', type=int, default=1,
                        help='use cosine lr schedule')
    parser.add_argument('--choose', type=str, default=None,
                        help="choose gpu for training, default:None(Use all available GPUs)")

    #clsa parameter configuration
    parser.add_argument('--alpha', type=float, default=1,
                        help="coefficients for DDM loss")
    parser.add_argument('--aug_times', type=int, default=5,
                        help="random augmentation times in strong augmentation")
    # idea from swav#adds crops for it
    parser.add_argument("--nmb_crops", type=int, default=[1, 1, 1, 1, 1], nargs="+",
                        help="list of number of crops (example: [2, 6])")  # when use 0 denotes the multi crop is not applied
    parser.add_argument("--size_crops", type=int, default=[224, 192, 160, 128, 96], nargs="+",
                        help="crops resolutions (example: [224, 96])")
    parser.add_argument("--min_scale_crops", type=float, default=[0.2, 0.172, 0.143, 0.114, 0.086], nargs="+",
                        help="min scale crop argument in RandomResizedCrop ")
    parser.add_argument("--max_scale_crops", type=float, default=[1.0, 0.86, 0.715, 0.571, 0.429], nargs="+",
                        help="max scale crop argument in RandomResizedCrop ")
    parser.add_argument("--pick_strong", type=int, default=[0, 1, 2, 3, 4], nargs="+",
                        help="specify the strong augmentation that will be used ")
    parser.add_argument("--pick_weak", type=int, default=[0, 1, 2, 3, 4], nargs="+",
                        help="specify the weak augmentation that will be used ")
    parser.add_argument("--clsa_t", type=float, default=0.2, help="temperature used for ddm loss")
    parser.add_argument("--sym",type=int,default=0,help="symmetrical loss apply or not (default:False)")
    args = parser.parse_args()
    params = vars(args)
    return args,params