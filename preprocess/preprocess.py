import argparse
import glob
import os
import shutil
import sys
import zipfile

from PIL import ImageFile
import dask
from dask.diagnostics import ProgressBar
from tqdm import tqdm

sys.path.insert(0, '../common/')
from prepare_dataset import FileHandler, Preprocessor
from progress.bar import Bar


ImageFile.LOAD_TRUNCATED_IMAGES = True

@dask.delayed
def preprocess_dask(preprocessor, file_handler, bar, file, source_path):
    """
    Preprocess image using dask. This function calls the preprocess function.
    Args:
        preprocessor: Preprocessor object.
        file_handler: FileHandler object.
        bar: Progress bar.
        file: Image file.
        source_path: Source path.
    """
    preprocess(preprocessor, file_handler, bar, file, source_path)

def preprocess(preprocessor, file_handler, bar, file, source_path):
    """
    Preprocess image.
    Args:
        preprocessor: Preprocessor object.
        file_handler: FileHandler object.
        bar: Progress bar.
        file: Image file.
        source_path: Source path.
    """ 
    for i in range(1, len(file.namelist()) + 1):
        img = file_handler.image_open(file, i)
        preprocessed_image = preprocessor.get_image_without_background(img)
        path = os.path.join(source_path, file.namelist()[i].split('/')[-1]).replace('jpeg', 'png')
        file_handler.image_write(path, preprocessed_image)
        bar.next()
 


file_handler = FileHandler()
preprocessor = Preprocessor()
parser = argparse.ArgumentParser(description='Preprocess retina images.')
parser.add_argument('--path', type=str, help='Path to the data. (in .zip format))', default='../train.zip')
parser.add_argument('--save_path', type=str, help='Path to save the preprocessed images.', default='./train_preprocessed')
parser.add_argument('--save_as_zip', action='store_true', help='Save the preprocessed images as a zip file.')
args = parser.parse_args()

save_path = file_handler.fold_dir(args.save_path)
data = glob.glob(args.path)
try:
    file = zipfile.ZipFile(data[0])
except:
    file = zipfile.ZipFile(data)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

print('\nPreprocessing images using dask...\n')
bar = Bar('Processing', max=len(file.namelist())-1)
print('\nComputing delayed tasks...')
delayed_tasks = []

# get all files in args.save_path
files = []
for r, d, f in os.walk(args.save_path):
    for file_in in f:
        files.append(os.path.join(r, file_in))

for i in range(1, len(file.namelist())):
    img = file_handler.image_open(file, i)
    path = os.path.join('./train_preprocessed', file.namelist()[i].split('/')[-1]).replace('jpeg', 'png')
    if path not in files:
        delayed_tasks.append(preprocessor.get_image_without_background(img, path))
    bar.next()
bar.finish()

print('\nExecuting ',len(delayed_tasks),'delayed tasks...\n')

progress_bar = tqdm(total=len(delayed_tasks), desc='Progress', unit='task', bar_format='{l_bar}{bar}|')
total = dask.delayed(delayed_tasks)
with ProgressBar():
    result = total.compute()
    progress_bar.update(1)
progress_bar.close()
print('\nDone.\n')

if args.save_as_zip:
    print('Saving as zip file...')
    shutil.make_archive(args.save_path, 'zip', args.save_path)
    shutil.rmtree(args.save_path)
