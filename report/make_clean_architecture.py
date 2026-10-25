"""Draw concise architecture diagrams with module names and dimensions."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle
ROOT=Path(__file__).resolve().parent

def box(ax,x,y,w,h,title,detail='',color='#eaf2f5'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07',linewidth=1.4,edgecolor='#557184',facecolor=color))
 ax.text(x+w/2,y+h*.64 if detail else y+h/2,title,ha='center',va='center',fontsize=14,fontweight='bold',color='#173e53')
 if detail:ax.text(x+w/2,y+h*.26,detail,ha='center',va='center',fontsize=11,color='#405562')
def arrow(ax,a,b,label='',dashed=False):
 ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#476474','linestyle':'--' if dashed else '-'})
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=10,color='#405562',bbox={'fc':'white','ec':'none','pad':1})
def save(fig,name):
 for ext in ('png','pdf'):fig.savefig(ROOT/'assets'/f'{name}.{ext}',dpi=230,bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(12,6));ax.set(xlim=(0,12),ylim=(0,6));ax.axis('off')
ax.text(.2,5.7,'MAPPO architecture · M30',fontsize=18,fontweight='bold',color='#173e53')
ax.text(.2,5.3,'Policy execution',fontsize=12,color='#557184')
box(ax,.25,3.7,2.35,1.1,'4-drone scene','Wind + dynamics')
box(ax,3.25,3.7,2.2,1.1,'Observations','37 features / drone')
box(ax,6.15,3.7,2.3,1.1,'Shared actor','128 → 128 → 128')
box(ax,9.15,3.7,2.35,1.1,'Velocity actions','vx, vy, vz')
arrow(ax,(2.67,4.25),(3.18,4.25));arrow(ax,(5.52,4.25),(6.08,4.25));arrow(ax,(8.52,4.25),(9.08,4.25))
ax.plot([10.32,10.32,1.42,1.42],[3.63,3.12,3.12,3.63],color='#476474',lw=1.5)
arrow(ax,(1.42,3.12),(1.42,3.63))
ax.add_patch(FancyBboxPatch((.1,.35),11.55,2.25,boxstyle='round,pad=.08',linestyle='--',ec='#b48b48',fc='#fffcf4',linewidth=1.3))
ax.text(.3,2.23,'Training only',fontsize=12,color='#8a652c')
box(ax,.5,.75,2.5,1.05,'Global state + ID','30 + 4 features',color='#f5ecd9')
box(ax,3.6,.75,2.5,1.05,'Central critic','128 → 128 → 1',color='#f5ecd9')
box(ax,6.8,.75,2.5,1.05,'GAE + PPO','Value / policy update',color='#f5ecd9')
arrow(ax,(3.07,1.28),(3.53,1.28));arrow(ax,(6.17,1.28),(6.73,1.28))
ax.plot([9.37,10.65,10.65,7.3],[1.28,1.28,2.9,2.9],color='#8a652c',ls='--',lw=1.5)
arrow(ax,(7.3,2.9),(7.3,3.63),dashed=True)
ax.text(9.65,2.05,'Actor weights',fontsize=10,ha='center',color='#8a652c')
arrow(ax,(1.9,3.63),(1.9,1.87),dashed=True)
save(fig,'architecture')
fig,axs=plt.subplots(1,2,figsize=(11.5,4.1))
for ax,title,input_title,detail in [(axs[0],'MAPPO','Global state + ID','30 + 4 features'),(axs[1],'PPO (IPPO)','Agent observation','34 (Tello) / 37 (M30)')]:
 ax.set(xlim=(0,5.5),ylim=(0,4));ax.axis('off');ax.text(2.75,3.7,title,ha='center',fontsize=18,fontweight='bold',color='#173e53')
 box(ax,.3,2.25,4.9,.9,input_title,detail)
 box(ax,.3,.55,4.9,.9,'Value network','128 → 128 → 1',color='#f5ecd9')
 arrow(ax,(2.75,2.18),(2.75,1.52))
fig.text(.5,.02,'Same actor and PPO settings · Different critic inputs',ha='center',fontsize=13,color='#405562')
fig.tight_layout(rect=(0,.07,1,1));save(fig,'critic_comparison')
print('Simplified architecture diagrams saved.')
