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
fig,axs=plt.subplots(1,2,figsize=(12,7.5))
for ax,method,critic_input,input_size,critic_title in [
 (axs[0],'MAPPO','Global state + ID','30 + 4 features','Central critic'),
 (axs[1],'PPO (IPPO)','Agent observation','34 / 37 features','Local critic')]:
 ax.set(xlim=(0,5.8),ylim=(0,7));ax.axis('off')
 ax.add_patch(FancyBboxPatch((.12,.28),5.5,6.3,boxstyle='round,pad=.06',ec='#c7d5de',fc='#ffffff'))
 ax.text(2.88,6.83,method,ha='center',fontsize=19,fontweight='bold',color='#173e53')
 box(ax,.35,5.05,2.3,1.1,'Agent observation','34 / 37 features')
 box(ax,.35,3.1,2.3,1.1,'Shared actor','128 → 128 → 128')
 box(ax,.35,1.15,2.3,1.1,'Velocity actions','vx, vy, vz')
 box(ax,3.1,5.05,2.25,1.1,critic_input,input_size,color='#f5ecd9')
 box(ax,3.1,3.1,2.25,1.1,critic_title,'128 → 128 → 1',color='#f5ecd9')
 box(ax,3.1,1.15,2.25,1.1,'GAE + PPO','Training only',color='#f5ecd9')
 arrow(ax,(1.5,4.98),(1.5,4.27));arrow(ax,(1.5,3.03),(1.5,2.32))
 arrow(ax,(4.22,4.98),(4.22,4.27));arrow(ax,(4.22,3.03),(4.22,2.32))
 ax.plot([3.03,2.87,2.87],[1.7,1.7,3.65],ls='--',color='#8a652c',lw=1.6)
 arrow(ax,(2.87,3.65),(2.72,3.65),dashed=True)
 ax.text(2.88,.62,'Policy execution     |     Value learning',ha='center',fontsize=10,color='#557184')
fig.suptitle('MAPPO and PPO architectures',fontsize=21,fontweight='bold',color='#173e53',y=.99)
fig.text(.5,.925,'Tello: 34 observation features     •     M30: 37 observation features',ha='center',fontsize=13,color='#405562')
fig.text(.5,.025,'Same actor design; separately trained models. Actor weights are shared across the four drones within each run.',ha='center',fontsize=10,color='#405562')
fig.tight_layout(rect=(0,.055,1,.91),w_pad=1.0);save(fig,'architecture')
fig,axs=plt.subplots(1,2,figsize=(11.5,4.1))
for ax,title,input_title,detail in [(axs[0],'MAPPO','Global state + ID','30 + 4 features'),(axs[1],'PPO (IPPO)','Agent observation','34 (Tello) / 37 (M30)')]:
 ax.set(xlim=(0,5.5),ylim=(0,4));ax.axis('off');ax.text(2.75,3.7,title,ha='center',fontsize=18,fontweight='bold',color='#173e53')
 box(ax,.3,2.25,4.9,.9,input_title,detail)
 box(ax,.3,.55,4.9,.9,'Value network','128 → 128 → 1',color='#f5ecd9')
 arrow(ax,(2.75,2.18),(2.75,1.52))
fig.text(.5,.02,'Same actor and PPO settings · Different critic inputs',ha='center',fontsize=13,color='#405562')
fig.tight_layout(rect=(0,.07,1,1));save(fig,'critic_comparison')
print('Simplified architecture diagrams saved.')
