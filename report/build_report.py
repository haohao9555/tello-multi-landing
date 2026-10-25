"""Build the editable/HTML report and the canonical signed XeLaTeX PDF."""
from pathlib import Path
import json,shutil,subprocess,sys
R=Path(__file__).resolve().parent
# Keep reusable signature assets before replacing any generated PDF.
sys.path.insert(0,'/tmp/rl-report-deps')
import pymupdf
pdf=R/'Project_Report.pdf';sig=R/'overleaf/signatures'
if pdf.exists() and not all((sig/(n+'.png')).exists() for n in ['chenghao','bo','junqi']):
 old=pymupdf.open(pdf);last=old[-1];images=sorted(last.get_images(),key=lambda im:last.get_image_rects(im[0])[0].y0)
 if len(images)==3:
  sig.mkdir(parents=True,exist_ok=True)
  for name,im in zip(['chenghao','bo','junqi'],images):
   pix=pymupdf.Pixmap(old,im[0]);pix=pymupdf.Pixmap(pix,pymupdf.Pixmap(old,im[1])) if im[1] else pix
   pix.save(sig/(name+'.png'))
 old.close()
for required in ['pandoc','latexmk','xelatex']:
 if not shutil.which(required):raise SystemExit(f'Missing report dependency: {required}. Install Pandoc and XeLaTeX, or compile the Overleaf project online.')
subprocess.run([sys.executable,str(R/'build_editable_report.py')],check=True)
subprocess.run([sys.executable,str(R/'export_overleaf.py')],check=True)
with (R/'overleaf/compile.log').open('w') as log:
 subprocess.run(['latexmk','-cd','-xelatex','-interaction=nonstopmode','-halt-on-error',str(R/'overleaf/main.tex')],stdout=log,stderr=subprocess.STDOUT,check=True)
subprocess.run([sys.executable,str(R/'build_math_docx.py')],check=True)
shutil.copyfile(R/'overleaf/main.pdf',pdf)
shutil.copyfile(pdf,R/'Group74_Final_Signed_Report.pdf')
d=pymupdf.open(pdf)
assert 'Project Report' in d[0].get_text() and 'Part E' not in d[0].get_text()
assert all(len(p.get_text())>20 for p in d)
if (sig/'chenghao.png').exists():assert len(d[-1].get_images())==3
print(json.dumps({'pdf':str(pdf),'pages':len(d),'engine':'XeLaTeX','signatures':len(d[-1].get_images())}))
