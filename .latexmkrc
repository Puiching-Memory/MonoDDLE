# Use XeLaTeX for Chinese documents.
# This allows `latexmk -pdf README.zh-CN.tex` to work in this project.
$pdf_mode = 1;
$pdflatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
