from PIL import Image
import glob
import nn.utils as utils
from pdf2image import convert_from_path

pdf_dir = utils.REMOTE_ROOT + '/saved_plots/lin'

datasets = ['a9a', 'duke', 'gisette_scale', 'madelon', 'mushrooms', 'w8a', 'phishing']
files = {(key, 'vr'): [] for key in datasets} | {(key, 'no_vr'): [] for key in datasets}


for file in glob.glob(pdf_dir + '/*.pdf'):
    for d in datasets:
        if d in file and 'no_vr' in file:
            files[(d, 'no_vr')].append(file)
        elif d in file and 'vr' in file:
            files[(d, 'vr')].append(file)


for key in files:
    images = []
    for file in sorted(files[key]):
        im = convert_from_path(file)[0]
        images.append(im)

    widths, heights = [im.width for im in images], [im.height for im in images]

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    new_im.save('lin_short/' + str(key) + '.jpg')





