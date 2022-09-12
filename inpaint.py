from argparse import ArgumentParser
from importlib import import_module
from os import makedirs
from os.path import join

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
from tqdm import tqdm

try :
    from .datasets import load_dataset, ZipDatasets
    from .train_utils import extend_batch_tuple
    from .VAEAC import VAEAC
    from .train import load_model
except :
    from datasets import load_dataset, ZipDatasets
    from train_utils import extend_batch_tuple
    from VAEAC import VAEAC
    from train import load_model

# saves inpainting to file
def save_img(img, path):
    ToPILImage()((img / 2 + 0.5).clamp(0, 1).cpu()).save(path)


def impute(model, sampler, batch, masks, use_cuda=False, nb_samples = 1):

    init_shape = batch.shape[0]

    # if batch size is less than batch_size, extend it with objects
    # from the beginning of the dataset
    # batch_tuple_extended = extend_batch_tuple(batch, dataloader,
                                            # batch_size)
    # batch_extended, masks_extended = batch_tuple_extended
    # TODO: Why do they need to extend the batch size here ?
    batch_extended = batch
    masks_extended = masks
    if next(model.parameters()).is_cuda:
        batch_extended = batch_extended.to(next(model.parameters()).device)
        masks_extended = masks_extended.to(next(model.parameters()).device)
        batch = batch.to(next(model.parameters()).device)
        masks = masks.to(next(model.parameters()).device)
        

    # compute imputation distributions parameters
    samples_params = model.generate_samples_params(batch_extended,
                                                masks_extended,
                                                nb_samples)
    samples_params = samples_params[:init_shape].flatten(0,1) # Flatten the number of samples and the batch size
    img_samples = sampler(samples_params)

    return img_samples, samples_params

#CMD : python inpaint.py --model_dir celeba_model --root_dir ..\..\..\Dataset\CELEBA\ --dataset celeba_test --masks celeba_inpainting_masks --out_dir exps\results\

def save_imputed_images(batch, mask, multiple_img_samples, samples_params, batch_size, num_samples, other_shape, out_dir, image_num = 0):
    multiple_img_samples = multiple_img_samples.reshape(batch_size, num_samples, *other_shape)

    for groundtruth, mask, multiple_img_sample, img_samples_params in zip(batch, masks,multiple_img_samples, samples_params):
        # save groundtruth image
        save_img(groundtruth,
                join(out_dir, '%05d_groundtruth.jpg' % image_num))

        # to show mask on the model input we use gray color
        model_input_visualization = torch.tensor(groundtruth)
        model_input_visualization[mask.byte()] = 0.5

        # save model input visualization
        save_img(model_input_visualization,
                join(out_dir, '%05d_input.jpg' % image_num))

        # in the model input the unobserved part is zeroed
        model_input = torch.tensor(groundtruth)
        model_input[mask.byte()] = 0

        img_samples = multiple_img_sample
        for i, sample in enumerate(img_samples):
            sample[1 - mask.byte()] = 0
            sample += model_input 
            sample_filename = join(out_dir,
                                '%05d_sample_%03d.jpg' % (image_num, i))
            save_img(sample, sample_filename)

        image_num += 1
   


if __name__ == "__main__":
    parser = ArgumentParser(description='Inpaint images using a given model.')

    parser.add_argument('--model_dir', type=str, action='store', required=True,
                        help='Directory with a model and its checkpoints. ' +
                            'It must be a directory in the root ' +
                            'of this repository.')

    parser.add_argument('--num_samples', type=int, action='store', default=5,
                        help='Number of different inpaintings per image.')

    parser.add_argument('--root_dir', type=str, action='store', required=True,
                        help='Number of different inpaintings per image.')

    parser.add_argument('--dataset', type=str, action='store', required=True,
                        help='The name of dataset of images to inpaint ' +
                            '(see load_datasets function in datasets.py)')

    parser.add_argument('--masks', type=str, action='store', required=True,
                        help='The name of masks dataset of the same length ' +
                            'as the images dataset. White color (i. e. one ' +
                            'in each channel) means a pixel to inpaint.')

    parser.add_argument('--out_dir', type=str, action='store', required=True,
                        help='The name of directory where to save ' +
                            'inpainted images.')

    parser.add_argument('--use_last_checkpoint', action='store_true',
                        default=False,
                        help='By default the model with the best ' +
                            'validation IWAE (best_checkpoint.tar) is used ' +
                            'to generate inpaintings. This flag indicates ' +
                            'that the last model (last_checkpoint.tar) ' +
                            'should be used instead.')

    args = parser.parse_args()

    # Default parameters which are not supposed to be changed from user interface
    use_cuda = torch.cuda.is_available()
    verbose = True
    # Non-zero number of workers cause nasty warnings because of some bug in
    # multiprocess library. It might be fixed now, so maybe it is time to set it
    # to the number of CPU cores in the system.
    num_workers = 0

    model_module, model, optimizer, batch_size, vlb_scale_factor, mask_generator, validation_iwae, train_vlb = load_model(args.model_dir, use_cuda=use_cuda)
    sampler = model_module.sampler

    # load images and masks datasets, build a dataloader on top of them
    dataset = load_dataset(args.dataset, args.root_dir)
    masks = load_dataset(args.masks, args.root_dir)
    dataloader = DataLoader(ZipDatasets(dataset, masks), batch_size=batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=num_workers)
    



    # create directory for inpaintings, if not exists
    makedirs(args.out_dir, exist_ok=True)

    iterator = dataloader
    if verbose:
        iterator = tqdm(iterator)
    with torch.no_grad():

        image_num = 0
        for batch_tuple in iterator:
            batch, masks = batch_tuple
            batch_size = batch.shape[0]
            other_shape = batch.shape[1:]
            multiple_img_samples, samples_params = impute(model, sampler, batch, masks, nb_samples = args.num_samples)
            save_imputed_images(batch, masks, multiple_img_samples, samples_params, batch_size, args.num_samples, other_shape, args.out_dir, image_num)