import os

from logger import _logger

class WebMaker(object):
    r"""Helper class for making a webpage"""

    def __init__(self, title):
        self.title = title
        self.text = []


    def add_text(self, line=''):
        self.text.append(line)


    def add_h1(self, line):
        self.add_text()
        self.text.append('# ' + line)
        self.add_text()


    def add_h2(self, line):
        self.add_text()
        self.text.append('## ' + line)
        self.add_text()


    def add_h3(self, line):
        self.add_text()
        self.text.append('### ' + line)
        self.add_text()


    def add_figure(self, basedir, src, title, width=400, height=400):
        path = os.path.join(basedir, src)
        if not os.path.exists(path):
            _logger.warning(f'Figure {path} not found when making the webpage')
            self.text.append(f'<textarea name="a" style="width:{width}px;height:{height}px;">{title}</textarea>')
        else:
            self.text.append(f'<img src="{src}" title="{title}" alt="{title}" style="width:{width}px;height:{height}px;"/>')


    def add_pdf(self, basedir, src, title, width=400, height=400):
        path = os.path.join(basedir, src)
        if not os.path.exists(path):
            _logger.warning(f'Pdf {path} not found when making the webpage')
            self.text.append(f'<textarea name="a" style="width:{width}px;height:{height}px;">{title}</textarea>')
        else:
            self.text.append(f'<object data="{src}" type="application/pdf" width="{width}px" height="{height}px"></object>')

    def add_textarea(self, text, width=400, height=400):
        # should first end the current zero-md block to add textarea blocks (otherwise text is interpreted by MD syntax)
        self.text.append('</xmp>\n    </template>\n</zero-md>')
        self.text.append(f'<textarea name="a" style="width:{width}px;height:{height}px;">{text}</textarea>')
        self.text.append('<zero-md>\n    <template>\n<xmp>')

    def write_to_file(self, dst, filename='index.html'):
        if not os.path.exists(dst):
            os.makedirs(dst)
        with open(os.path.join(dst, filename), 'w') as fw:
            fw.write(self.get_webpage_template().replace('__TITLE__', self.title).replace('__TEXT__', '\n'.join(self.text)))


    def get_webpage_template(self):
        return '''<!doctype html>
<html lang="">
<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <!-- Edit your site info here -->
    <meta name="description" content="EXAMPLE SITE DESCRIPTION">
    <title>__TITLE__</title>

    <script src="https://cdn.jsdelivr.net/npm/@webcomponents/webcomponentsjs@2/webcomponents-loader.min.js"></script>
    <script type="module" src="https://cdn.jsdelivr.net/gh/zerodevx/zero-md@1/src/zero-md.min.js"></script>

    <style>
    /* Edit your header styles here */
    header { font-family: sans-serif; font-size: 20px; text-align: center; position: fixed; width: 100%; line-height: 42px; top: 0; left: 0; background-color: #424242; color: white; }
    body { box-sizing: border-box; min-width: 200px; max-width: 2000px; margin: 56px auto 0 auto; padding: 45px; }
    .year-buttons { font-family: sans-serif; margin: 0 0 18px 0; }
    .year-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 7px; align-items: center; }
    .year-group { display: inline-flex; flex-wrap: wrap; gap: 6px; margin-right: 12px; }
    .year-buttons button {
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 5px;
        padding: 5px 9px;
        font-size: 13px;
        cursor: pointer;
        color: #1f2933;
    }
    .year-buttons button:hover { filter: brightness(0.95); }
    .era-run2 { background: #d8ecff; }
    .era-run3a { background: #ddf4df; }
    .era-run3b { background: #fff0c2; }
    .era-run2.dark { background: #9ecbef; }
    .era-run3a.dark { background: #a9d8ad; }
    .era-run3b.dark { background: #e0c56e; }
    @media (max-width: 767px) {
        header { font-size: 15px; }
        body { padding: 15px; }
    }
    </style>
</head>
<body>
<div class="year-buttons">
<div class="year-row">
<span class="year-group">
<button class="era-run2" onclick="replaceYear('2016APV_v9')">2016APV_v9</button>
<button class="era-run2" onclick="replaceYear('2016_v9')">2016_v9</button>
<button class="era-run2" onclick="replaceYear('2017_v9')">2017_v9</button>
<button class="era-run2" onclick="replaceYear('2018_v9')">2018_v9</button>
</span>
<span class="year-group">
<button class="era-run3a" onclick="replaceYear('2022_v12')">2022_v12</button>
<button class="era-run3a" onclick="replaceYear('2022EE_v12')">2022EE_v12</button>
<button class="era-run3a" onclick="replaceYear('2023_v12')">2023_v12</button>
<button class="era-run3a" onclick="replaceYear('2023BPix_v12')">2023BPix_v12</button>
<button class="era-run3a" onclick="replaceYear('2022Comb_v12')">2022Comb_v12</button>
<button class="era-run3a" onclick="replaceYear('2023Comb_v12')">2023Comb_v12</button>
</span>
</div>
<div class="year-row">
<span class="year-group">
<button class="era-run2 dark" onclick="replaceYear('2016APV_v15')">2016APV_v15</button>
<button class="era-run2 dark" onclick="replaceYear('2016_v15')">2016_v15</button>
<button class="era-run2 dark" onclick="replaceYear('2017_v15')">2017_v15</button>
<button class="era-run2 dark" onclick="replaceYear('2018_v15')">2018_v15</button>
</span>
<span class="year-group">
<button class="era-run3a dark" onclick="replaceYear('2022_v15')">2022_v15</button>
<button class="era-run3a dark" onclick="replaceYear('2022EE_v15')">2022EE_v15</button>
<button class="era-run3a dark" onclick="replaceYear('2023_v15')">2023_v15</button>
<button class="era-run3a dark" onclick="replaceYear('2023BPix_v15')">2023BPix_v15</button>
<button class="era-run3a dark" onclick="replaceYear('2022Comb_v15')">2022Comb_v15</button>
<button class="era-run3a dark" onclick="replaceYear('2023Comb_v15')">2023Comb_v15</button>
</span>
<span class="year-group">
<button class="era-run3b dark" onclick="replaceYear('2024_v15')">2024_v15</button>
<button class="era-run3b dark" onclick="replaceYear('2025_v15')">2025_v15</button>
</span>
</div>
</div>
<script>
function replaceYear(y) {
    const pat = '(2016APV|2016|2017|2018|2022EE|2022Comb|2022|2023BPix|2023Comb|2023|2024|2025)(?:_v[0-9]+)?';
    location.href=location.href.replace(new RegExp(pat), y);
}
</script>

<!-- Edit your Markdown URL file location here -->
<zero-md>
    <!-- Declare `<template>` element as a child of `<zero-md>` -->
    <template>
    <!-- Wrap your markdown string inside an `<xmp>` tag -->
<xmp>
----------
__TEXT__
</xmp>
    </template>
</zero-md>

    <!-- Edit your header title here -->
    <header class="header">__TITLE__</header>

</body>
</html>'''
