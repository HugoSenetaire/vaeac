from argparse import ArgumentParser
from importlib import import_module
from math import ceil
from os import replace
import os
from os.path import exists, join
from shutil import copy
from sys import stderr
from tabnanny import check

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try :
    from .datasets import load_dataset
    from .train_utils import extend_batch, get_validation_iwae
    from .VAEAC import VAEAC
except :
    from datasets import load_dataset
    from train_utils import extend_batch, get_validation_iwae
    from VAEAC import VAEAC

#CMD : python train.py --model_dir celeba_model --epochs 10 --root_dir ..\..\..\Dataset\CELEBA\ --train_dataset celeba_train --validation_dataset celeba_val

# Makes checkpoint of the current state.
# The checkpoint contains current epoch (in the current run),
# VAEAC and optimizer parameters, learning history.
# The function writes checkpoint to a temporary file,
# and then replaces last_checkpoint.tar with it, because
# the replacement operation is much more atomic than
# the writing the state to the disk operation.
# So if the function is interrupted, last checkpoint should be
# consistent.
def make_checkpoint(epoch, model_dir, model, optimizer, validation_iwae, train_vlb):

    filename = join(model_dir, 'last_checkpoint.tar')
    with open(filename + '.bak', "wb") as f :
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_iwae': validation_iwae,
            'train_vlb': train_vlb,
        }, f)
    replace(filename + '.bak', filename)


def load_model(model_dir, use_cuda = False, train = True, load_best = True):
    # Default parameters which are not supposed to be changed from user interface

    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, so maybe it is time to set it
    # to the number of CPU cores in the system.


    # import the module with the model networks definitions,
    # optimization settings, and a mask generator
    import sys
    sys.path.append(model_dir) # @TODO: fix this, that's ugly
    model_module = import_module('model')
    

    # build VAEAC on top of the imported networks
    model = VAEAC(
        model_module.reconstruction_log_prob,
        model_module.proposal_network,
        model_module.prior_network,
        model_module.generative_network
    )
    if use_cuda:
        model = model.cuda()

    
    # build optimizer and import its parameters
    optimizer = model_module.optimizer(model.parameters())
    batch_size = model_module.batch_size
    vlb_scale_factor = getattr(model_module, 'vlb_scale_factor', 1)


    # import mask generator
    mask_generator = model_module.mask_generator


    # a list of validation IWAE estimates
    validation_iwae = []
    # a list of running variational lower bounds on the train set
    train_vlb = []
    # the length of two lists above is the same because the new
    # values are inserted into them at the validation checkpoints only

    if load_best :
        checkpoint_name = 'best_checkpoint.tar'
    else :
        checkpoint_name = 'last_checkpoint.tar'
     # load the last checkpoint, if it exists
    if exists(join(model_dir, checkpoint_name)):
        location = 'cuda' if use_cuda else 'cpu'
        checkpoint = torch.load(join(model_dir, checkpoint_name),
                                map_location=location)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        validation_iwae = checkpoint['validation_iwae']
        train_vlb = checkpoint['train_vlb']
    else :
        if not train :
            raise Exception('No checkpoint found')
    
    
    return model_module, model, optimizer, batch_size, vlb_scale_factor, mask_generator, validation_iwae, train_vlb

def train_vaeac(model, model_dir, epochs, dataloader, val_dataloader, mask_generator, optimizer, validation_iwae, train_vlb, vlb_scale_factor, batch_size, validation_iwae_num_samples = 25, validations_per_epoch = 5, use_cuda = False, verbose = True):
    # number of batches after which it is time to do validation
    validation_batches = ceil(len(dataloader) / validations_per_epoch)
    # main train loop
    for epoch in range(epochs):

        iterator = dataloader
        avg_vlb = 0
        if verbose:
            print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
            iterator = tqdm(iterator)

        # one epoch
        for i, batch in enumerate(iterator):
            # the time to do a checkpoint is at start and end of the training
            # and after processing validation_batches batches
            
            if any([
                        i == 0 and epoch == 0,
                        i % validation_batches == validation_batches - 1,
                        i + 1 == len(dataloader)
                    ]):
                val_iwae = get_validation_iwae(val_dataloader, mask_generator,
                                            batch_size, model,
                                            validation_iwae_num_samples,
                                            verbose)
                validation_iwae.append(val_iwae)
                train_vlb.append(avg_vlb)



                make_checkpoint(epoch, model_dir, model, optimizer, validation_iwae, train_vlb)

                # if current model validation IWAE is the best validation IWAE
                # over the history of training, the current checkpoint is copied
                # to best_checkpoint.tar
                # copying is done through a temporary file, i. e. firstly last
                # checkpoint is copied to temporary file, and then temporary file
                # replaces best checkpoint, so even best checkpoint should be
                # consistent even if the script is interrupted
                # in the middle of copying
                if max(validation_iwae[::-1]) <= val_iwae:
                    src_filename = join(model_dir, 'last_checkpoint.tar')
                    dst_filename = join(model_dir, 'best_checkpoint.tar')
                    copy(src_filename, dst_filename + '.bak')
                    replace(dst_filename + '.bak', dst_filename)

                if verbose:
                    print(file=stderr)
                    print(file=stderr)

            # if batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch = extend_batch(batch, dataloader, batch_size)

            # generate mask and do an optimizer step over the mask and the batch
            mask = mask_generator(batch)
            optimizer.zero_grad()
            if use_cuda:
                batch = batch.cuda()
                mask = mask.cuda()
            vlb = model.batch_vlb(batch, mask).mean()
            (-vlb / vlb_scale_factor).backward()
            optimizer.step()

            # update running variational lower bound average
            avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
            if verbose:
                iterator.set_description('Train VLB: %g' % avg_vlb)


if __name__ == '__main__':

    parser = ArgumentParser(description='Train VAEAC to inpaint.')

    parser.add_argument('--model_dir', type=str, action='store', required=True,
                        help='Directory with model.py. ' +
                            'It must be a directory in the root ' +
                            'of this repository. ' +
                            'The checkpoints are saved ' +
                            'in this directory as well. ' +
                            'If there are already checkpoints ' +
                            'in the directory, the training procedure ' +
                            'is resumed from the last checkpoint ' +
                            '(last_checkpoint.tar).')

    parser.add_argument('--root_dir', type=str, action='store', required=True,)

    parser.add_argument('--epochs', type=int, action='store', required=True,
                        help='Number epochs to train VAEAC.')

    parser.add_argument('--train_dataset', type=str, action='store',
                        required=True,
                        help='Dataset of images for training VAEAC to inpaint ' +
                            '(see load_datasets function in datasets.py).')

    parser.add_argument('--validation_dataset', type=str, action='store',
                        required=True,
                        help='Dataset of validation images for VAEAC ' +
                            'log-likelihood IWAE estimate ' +
                            '(see load_datasets function in datasets.py).')

    parser.add_argument('--validation_iwae_num_samples', type=int, action='store',
                        default=25,
                        help='Number of samples per object to estimate IWAE ' +
                            'on the validation set. Default: 25.')

    parser.add_argument('--validations_per_epoch', type=int, action='store',
                        default=5,
                        help='Number of IWAE estimations on the validation set ' +
                            'per one epoch on the training set. Default: 5.')

    args = parser.parse_args()

    verbose = True
    num_workers = 0
    use_cuda = torch.cuda.is_available()
    model_module, model, optimizer, batch_size, vlb_scale_factor, mask_generator, validation_iwae, train_vlb = load_model(args.model_dir, use_cuda=use_cuda)

    
    # load train and validation datasets
    train_dataset = load_dataset(args.train_dataset, args.root_dir,)
    validation_dataset = load_dataset(args.validation_dataset, args.root_dir)

    # build dataloaders on top of datasets
    dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True, drop_last=False,
                            num_workers=num_workers)


    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size,
                                shuffle=True, drop_last=False,
                                num_workers=num_workers)

    train_vaeac(model,
                args.model_dir,
                args.epochs,
                dataloader,
                val_dataloader,
                mask_generator,
                optimizer,
                validation_iwae,
                train_vlb,
                vlb_scale_factor,
                batch_size,
                validation_iwae_num_samples = args.validation_iwae_num_samples,
                validations_per_epoch = args.validations_per_epoch,
                use_cuda = use_cuda,
                verbose = verbose)