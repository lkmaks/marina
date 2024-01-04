import glob

files = glob.glob('data/*')
with open('divs.html', 'w') as out:
    for file in files:
        if file.find('main') != -1:
            out.write(f'<div class="pdf-item" data-pdf="{file}"></div>\n')

