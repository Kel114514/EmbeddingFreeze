# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse

from recbole.quick_start import run_recbole


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')
    parser.add_argument('--pretr_emb_epoch', type=int, default=None, help='number of epoch for embedding reuse')

    args, _ = parser.parse_known_args()
    parameter_dict = {
        'neg_sampling': None,
        'pretrain_embedding_epoch': args.pretr_emb_epoch,
        'train_batch_size': 512,
        'topk': [10, 20],                 # ensure 20 is in topk
        'metrics': ['Hit', 'MRR', 'Recall'],    # request HR and MRR
        'log_filename': f'{args.model}-{args.dataset}-{"base" if args.pretr_emb_epoch is None else "freeze"}.log',
    }
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
