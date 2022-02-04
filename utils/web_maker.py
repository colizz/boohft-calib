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
    @media (max-width: 767px) {
        header { font-size: 15px; }
        body { padding: 15px; }
    }
    </style>
</head>
<body>
<button onclick="replaceYear(2016APV)">2016APV</button>
<button onclick="replaceYear(2016)">2016</button>
<button onclick="replaceYear(2017)">2017</button>
<button onclick="replaceYear(2018)">2018</button>
<script>
function replaceYear(y) {
    location.href=location.href.replace(new RegExp('(2016APV|2016|2017|2018)'), y);
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