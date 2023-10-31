def add_arguments(parser):
    '''
    Arguments.
    '''
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_features', type=int, default=84*84*4)
    parser.add_argument('--n_actions',type=int, default=4)
    parser.add_argument('--train', action='store_true', help='whether train DQN')
    parser.add_argument('--test', action='store_true', help='whether test DQN')
    parser.add_argument('--double', action='store_true')
    parser.add_argument('--duel', action='store_true')

    return parser