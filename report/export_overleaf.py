"""Create a self-contained XeLaTeX project with equations and automatic references."""
from pathlib import Path
import re,shutil,subprocess,json
R=Path(__file__).resolve().parent;O=R/'overleaf';O.mkdir(exist_ok=True);(O/'figures').mkdir(exist_ok=True)
s=(R/'Project_Report.md').read_text()
abstract=re.search(r'## Abstract\n\n(.*?)\n\n## 1\.',s,re.S)[1]
body=s[s.index('## 1. Introduction'):s.index('## 5. References')]
appendix=s[s.index('### A. Individual Contribution'):s.index('**Signatures**')]
math={
'`a_xy = clip(k_v (v_cmd − v), ±a_max)_xy − (d_1 + d_2 ||v_xy − w||)(v_xy − w)`':r'''\begin{equation}\label{eq:dynamics}
\begin{aligned}
\mathbf a_{xy}={}&[\operatorname{clip}(k_v(\mathbf v_{\mathrm{cmd}}-\mathbf v),-a_{\max},a_{\max})]_{xy}\\
&-(d_1+d_2\lVert\mathbf v_{xy}-\mathbf w\rVert_2)(\mathbf v_{xy}-\mathbf w).
\end{aligned}
\end{equation}''',
'`r_i = 5(C_i,t − C_i,t+1) − 0.02 − 0.05 ||u_i,t − u_i,t−1||² + event and proximity terms`':r'''\begin{equation}\label{eq:reward}
\begin{aligned}
r_{i,t}={}&5(C_{i,t}-C_{i,t+1})-0.02-0.05\lVert\mathbf u_{i,t}-\mathbf u_{i,t-1}\rVert_2^2\\
&+r^{\mathrm{event}}_{i,t}+r^{\mathrm{prox}}_{i,t}.
\end{aligned}
\end{equation}''',
'`C_i = (horizontal target distance + 0.5 × height) / length_scale`':r'$C_{i,t}=(\lVert\mathbf p^{xy}_{i,t}-\mathbf p_i^\star\rVert_2+0.5\max(z_{i,t},0))/L$',
'`length_scale = 10`':r'$L=10$',
'`rho = exp(log pi_new − log pi_old)`':r'$\rho_{i,t}(\theta)=\pi_\theta(\widetilde{\mathbf a}_{i,t}\mid o_{i,t})/\pi_{\theta_{\mathrm{old}}}(\widetilde{\mathbf a}_{i,t}\mid o_{i,t})$',
'`L_actor = −mean[min(rho × A, clip(rho, 1 − epsilon, 1 + epsilon) × A)]`':r'''\begin{equation}\label{eq:ppo}
\mathcal L_{\mathrm{actor}}(\theta)=-\mathbb E_{(i,t)\in\mathcal B}\!\left[
\min\!\left\{\rho_{i,t}(\theta)\widehat A_{i,t},
\operatorname{clip}(\rho_{i,t}(\theta),1-\epsilon,1+\epsilon)\widehat A_{i,t}\right\}\right].
\end{equation}''',
'`A`':r'$\widehat A_{i,t}$','`epsilon = 0.1`':r'$\epsilon=0.1$',
'`delta_t = r_t + gamma × continuation_t × V_next − V_t`':r'$\delta_{i,t}=\widetilde r_{i,t}+\gamma m_tV_{i,t+1}-V_{i,t}$',
'`A_t = delta_t + gamma × lambda × continuation_t × A_next`':r'$\widehat A_{i,t}=\delta_{i,t}+\gamma\lambda m_t\widehat A_{i,t+1}$'}
for a,b in math.items():body=body.replace(a,b)
body=body.replace('Here '+math['`C_i = (horizontal target distance + 0.5 × height) / length_scale`']+'; M30 uses '+math['`length_scale = 10`']+'.',r'''The progress cost is

\begin{equation}\label{eq:cost}
C_{i,t}=\frac{\lVert\mathbf p^{xy}_{i,t}-\mathbf p_i^\star\rVert_2+0.5\max(z_{i,t},0)}{L}.
\end{equation}

Here $L=1$ for Tello and $L=10$ for M30.''')
ratio='Let the likelihood ratio be '+math['`rho = exp(log pi_new − log pi_old)`']+'.'
body=body.replace(ratio,r'''The likelihood ratio is

\begin{equation}\label{eq:ratio}
\rho_{i,t}(\theta)=\frac{\pi_\theta(\widetilde{\mathbf a}_{i,t}\mid o_{i,t})}{\pi_{\theta_{\mathrm{old}}}(\widetilde{\mathbf a}_{i,t}\mid o_{i,t})}.
\end{equation}

''')
gae='A residual has the form '+math['`delta_t = r_t + gamma × continuation_t × V_next − V_t`']+', and the backward recursion uses '+math['`A_t = delta_t + gamma × lambda × continuation_t × A_next`']+'.'
body=body.replace(gae,r'''The temporal-difference residual and advantage recursion are

\begin{align}
\delta_{i,t}&=\widetilde r_{i,t}+\gamma m_t V_{i,t+1}-V_{i,t},\label{eq:td}\\
\widehat A_{i,t}&=\delta_{i,t}+\gamma\lambda m_t\widehat A_{i,t+1}.\label{eq:gae}
\end{align}

Here $\widetilde r_{i,t}$ is the normalised rollout reward and $m_t$ is the scene-level continuation indicator. The value estimates are expressed in the same reward units before these residuals are formed.''')
marker='Commands are rotated back to world coordinates and scaled to physical velocity units.'
body=body.replace(marker,marker+r'''

The Gaussian raw action and bounded command are related by

\begin{equation}\label{eq:action}
\widetilde{\mathbf a}_{i,t}\sim\mathcal N(\symbf\mu_\theta(o_{i,t}),\operatorname{diag}(\symbf\sigma_i^2)),\qquad
\mathbf u_{i,t}=2\operatorname{sigmoid}(\widetilde{\mathbf a}_{i,t})-\mathbf 1.
\end{equation}
''')
marker='Four-drone success is the fraction of episodes in which all four aircraft land successfully.'
body=body.replace(marker,r'''Let $b_{e,i}$ indicate a successful landing by aircraft $i$ in episode $e$. With $E=100$ evaluated episodes and $N=4$ aircraft,

\begin{equation}\label{eq:success}
S_{\mathrm{fleet}}=\frac{1}{E}\sum_{e=1}^{E}\prod_{i=1}^{N}b_{e,i},\qquad
S_{\mathrm{drone}}=\frac{1}{EN}\sum_{e=1}^{E}\sum_{i=1}^{N}b_{e,i}.
\end{equation}

'''+marker)
def tex(text):
 return subprocess.run(['pandoc','-f','markdown+raw_tex+tex_math_dollars','-t','latex','--wrap=none'],input=text,text=True,capture_output=True,check=True).stdout.strip().replace(r'\textasciitilde{}\ref',r'~\ref')
def esc(t):
 m={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
 return ''.join(m.get(c,c) for c in t)
figures={};tables={}
def fig(m):
 n,caption,relative=m.groups();p=R/relative
 if p.with_suffix('.pdf').exists():p=p.with_suffix('.pdf')
 shutil.copy2(p,O/'figures'/p.name);label='fig:'+p.stem;figures[n]=label
 return '\n\n'+r'\begin{figure}[H]'+'\n'+r'\centering'+'\n'+r'\includegraphics[max width=\linewidth,max height=.62\textheight,keepaspectratio]{figures/'+p.name+'}\n'+r'\caption{'+esc(caption)+'}\n'+r'\label{'+label+'}\n'+r'\end{figure}'+'\n\n'+(r'\clearpage'+'\n\n' if p.stem=='architecture' else '')
def tab(m):
 n,caption,block=m.groups();rows=[]
 for row in block.strip().splitlines():
  cells=[c.strip() for c in row.strip().strip('|').split('|')]
  if all(re.fullmatch(':?-+:?',c) for c in cells):continue
  rows.append(cells)
 cols=len(rows[0]);tables[n]='tab:'+n
 lines=[r'\begin{table}[H]',r'\centering\small',r'\caption{'+esc(caption)+'}',r'\label{tab:'+n+'}',r'\begin{tabularx}{\linewidth}{@{}'+(r'>{\raggedright\arraybackslash}X'*cols)+'@{}}',r'\toprule']
 for i,row in enumerate(rows):
  cells=[esc(c) for c in row]
  if i==0:cells=[r'\textbf{'+c+'}' for c in cells]
  lines.append(' & '.join(cells)+r' \\')
  if i==0:lines.append(r'\midrule')
 lines += [r'\bottomrule',r'\end{tabularx}',r'\end{table}']
 return '\n\n'+'\n'.join(lines)+'\n\n'
word_source='## Abstract\n\n'+abstract+'\n\n'+body+'\n\n'+s[s.index('## 5. References'):]
word_source=re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}',lambda m:'\n\n$$\n'+m[1]+'\n$$\n\n',word_source,flags=re.S)
word_source=re.sub(r'\\begin\{align\}(.*?)\\end\{align\}',lambda m:'\n\n$$\n'+r'\begin{aligned}'+m[1]+r'\end{aligned}'+'\n$$\n\n',word_source,flags=re.S)
word_source=re.sub(r'\\label\{[^}]+\}','',word_source).replace(r'\symbf',r'\boldsymbol')
word_source=re.sub(r'^(#{2,4}) ',lambda m:'#'*(len(m[1])-1)+' ',word_source,flags=re.M)
word_source=re.sub(r'\$\$(.*?)\$\$',lambda m:'$$\n'+'\n'.join(line for line in m[1].splitlines() if line.strip())+'\n$$',word_source,flags=re.S)
(O/'word_math.md').write_text(word_source)
(O/'web_math.md').write_text(s[:s.index('## Abstract')]+word_source)
body=re.sub(r'!\[Figure (\d+)\. (.*?)\]\((.*?)\)',fig,body)
body=re.sub(r'\*\*Table (\d+)\. (.*?)\*\*\n\n((?:\|[^\n]*\|\n)+)',tab,body)
body=re.sub(r'\bFigure (\d+)\b',lambda m:r'Figure~\ref{'+figures[m[1]]+'}',body)
body=body.replace('Tables 7 and 8',r'Tables~\ref{tab:7} and~\ref{tab:8}')
body=re.sub(r'\bTable (\d+)\b',lambda m:r'Table~\ref{'+tables[m[1]]+'}',body)
keys={'1':'ppo2017','2':'mappo2022','3':'gae2016','4':'gazebo'}
body=re.sub(r'\[(\d+(?:,\s*\d+)*)\]',lambda m:r'\cite{'+','.join(keys[n.strip()] for n in m[1].split(','))+'}' if all(n.strip() in keys for n in m[1].split(',')) else m[0],body)
body=re.sub(r'^(#{2,4}) ',lambda m:'#'*(len(m[1])-1)+' ',body,flags=re.M)
body=re.sub(r'^(#{1,3}) \d+(?:\.\d+)*\.? ',r'\1 ',body,flags=re.M)
body=body.replace('## System Architecture Design',r'\clearpage'+'\n\n## System Architecture Design')
(O/'content.tex').write_text(tex(body)+'\n');(O/'abstract.tex').write_text(tex(abstract)+'\n')
appendix=appendix.replace('### A. Individual Contribution','# Individual Contribution').replace('### B. Individual Contribution Split',r'\clearpage'+'\n\n# Individual Contribution Split')
(O/'appendices.tex').write_text(tex(appendix)+r'''
\medskip
\noindent\textbf{Signatures}\par\medskip
\signaturefield{Chenghao Zhuang}{chenghao}
\signaturefield{Bo Zhao}{bo}
\signaturefield{Junqi Zhang}{junqi}
''')
(O/'references.bib').write_text(r'''@misc{ppo2017,author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},title={Proximal Policy Optimization Algorithms},year={2017},howpublished={arXiv:1707.06347},note={\url{https://arxiv.org/abs/1707.06347}}}
@inproceedings{mappo2022,author={Yu, Chao and Velu, Akash and Vinitsky, Eugene and Gao, Jiaxuan and Wang, Yu and Bayen, Alexandre and Wu, Yi},title={The Surprising Effectiveness of {PPO} in Cooperative, Multi-Agent Games},booktitle={Advances in Neural Information Processing Systems},year={2022},note={\url{https://arxiv.org/abs/2103.01955}}}
@inproceedings{gae2016,author={Schulman, John and Moritz, Philipp and Levine, Sergey and Jordan, Michael and Abbeel, Pieter},title={High-Dimensional Continuous Control Using Generalized Advantage Estimation},booktitle={International Conference on Learning Representations},year={2016},note={\url{https://arxiv.org/abs/1506.02438}}}
@misc{gazebo,author={{Gazebo}},title={{Gazebo Sim}: A Robotic Simulator},howpublished={Official project documentation},year={2026},note={Accessed 25 October 2026. \url{https://gazebosim.org/libs/sim/}}}
''')
(O/'main.tex').write_text(r'''% Compile with XeLaTeX. Import this project ZIP directly into Overleaf.
\documentclass[11pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{Latin Modern Mono}
\usepackage{amsmath,amssymb,mathtools,unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{microtype,setspace}
\setstretch{1.12}
\usepackage{graphicx,float,booktabs,tabularx,longtable,array,calc}
\usepackage[export]{adjustbox}
\usepackage[section]{placeins}
\usepackage{caption,titlesec}
\captionsetup{font=small,labelfont=bf,justification=raggedright,singlelinecheck=false,skip=6pt}
\titleformat{\section}{\Large\sffamily\bfseries}{\thesection}{.7em}{}
\titleformat{\subsection}{\large\sffamily\bfseries}{\thesubsection}{.7em}{}
\titleformat{\subsubsection}{\normalsize\sffamily\bfseries}{\thesubsubsection}{.7em}{}
\titlespacing*{\section}{0pt}{18pt}{9pt}
\titlespacing*{\subsection}{0pt}{14pt}{6pt}
\titlespacing*{\subsubsection}{0pt}{11pt}{4pt}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\setlength{\emergencystretch}{2em}
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}
\usepackage{xcolor,xurl}
\definecolor{linkblue}{HTML}{175D7B}
\usepackage[unicode,colorlinks=true,linkcolor=linkblue,urlcolor=linkblue,citecolor=linkblue]{hyperref}
\hypersetup{pdftitle={Robust Autonomous Multi-UAV Landing under Wind Disturbances Using PPO},pdfauthor={Group 74}}
\urlstyle{same}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\passthrough}[1]{#1}
\newcommand{\signaturefield}[2]{\noindent\makebox[34mm][l]{#1:}%
\ifdefined\HideSignatures\rule[-5mm]{0pt}{16mm}\else
\IfFileExists{signatures/#2.png}{\raisebox{-5mm}{\includegraphics[height=16mm]{signatures/#2.png}}}{\rule[-5mm]{0pt}{16mm}}\fi
\par\vspace{5mm}}
\begin{document}
\begin{titlepage}
\hypersetup{pageanchor=false}\thispagestyle{empty}\vspace*{18mm}
{\Large\sffamily 43008: Reinforcement Learning\par}\vspace{12mm}
{\Huge\sffamily\bfseries Project Report\par}\vspace{12mm}
{\LARGE\sffamily\bfseries Robust Autonomous Multi-UAV Landing under Wind Disturbances Using PPO\par}\vspace{14mm}
{\large\textbf{Group number:} 74\par}\vspace{10mm}
\begin{tabularx}{\textwidth}{@{}XX@{}}
\toprule\textbf{Team member}&\textbf{Student ID}\\\midrule
Chenghao Zhuang&25965245\\[5pt]
Bo Zhao&25971060\\[5pt]
Junqi Zhang&25971159\\\bottomrule
\end{tabularx}
\end{titlepage}
\pagenumbering{roman}\hypersetup{pageanchor=true}\tableofcontents\clearpage
\phantomsection\section*{Abstract}\addcontentsline{toc}{section}{Abstract}
{\small\input{abstract.tex}}\clearpage
\pagenumbering{arabic}\input{content.tex}\clearpage
\refstepcounter{section}\renewcommand{\refname}{\thesection\quad References}\phantomsection\addcontentsline{toc}{section}{\protect\numberline{\thesection}References}
\bibliographystyle{ieeetr}\bibliography{references}\clearpage
\appendix\input{appendices.tex}
\end{document}
''')
(O/'README.md').write_text('''# Group 74 — Overleaf project

Import this ZIP into Overleaf. Select `main.tex` as the main document and **XeLaTeX** as the compiler.

Local compilation:

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The report uses native numbered equations, automatic figure/table references and a BibTeX bibliography. The private bundle includes the original transparent signature images. The public repository omits these personal images; signing fields remain blank if they are absent. No date or Part E label appears on the cover or footer. Contributions are 33.3% each, without a total row.

Implementation: https://github.com/haohao9555/tello-multi-landing
''')
(O/'.latexmkrc').write_text("$pdf_mode = 5;\n")
(O/'overleaf.json').write_text(json.dumps({'compiler':'xelatex','main':'main.tex'},indent=2))
print(json.dumps({'project':str(O),'figures':len(figures),'tables':len(tables),'abstract_words':len(abstract.split())}))
