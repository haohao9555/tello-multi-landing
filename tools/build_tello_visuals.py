"""Build original Tello-inspired visual meshes without altering simulation physics.

Reproducible OBJ geometry in metres. This is an illustrative model, not DJI CAD.
Only <visual> nodes are replaced. Existing links, joints, inertials, collision
geometry and control plugins are retained exactly.
"""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'src/tello_sim_description/models/tello'
OUT=MODEL/'meshes';OUT.mkdir(exist_ok=True)


def save(name, vertices, faces):
    v=np.asarray(vertices,dtype=float);normals=np.zeros_like(v)
    for face in faces:
        a,b,c=v[list(face)];normal=np.cross(b-a,c-a)
        for i in face:normals[i]+=normal
    normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    with (OUT/f'{name}.obj').open('w') as f:
        f.write('# Original AeroLab visual geometry; units: metres\n')
        for p in v:f.write('v '+' '.join(f'{x:.8f}' for x in p)+'\n')
        for n in normals:f.write('vn '+' '.join(f'{x:.8f}' for x in n)+'\n')
        for face in faces:f.write('f '+' '.join(f'{i+1}//{i+1}' for i in face)+'\n')


def shell(name,rings,segments=96):
    vertices=[];faces=[]
    for z,a,b,power in rings:
        for i in range(segments):
            t=2*math.pi*i/segments;c,s=math.cos(t),math.sin(t)
            vertices.append((a*math.copysign(abs(c)**(2/power),c),b*math.copysign(abs(s)**(2/power),s),z))
    for j in range(len(rings)-1):
        for i in range(segments):
            a=j*segments+i;b=j*segments+(i+1)%segments;c=b+segments;d=a+segments
            faces.extend([(a,b,c),(a,c,d)])
    for ring,reverse in [(0,True),(len(rings)-1,False)]:
        center=len(vertices);vertices.append((0,0,rings[ring][0]))
        for i in range(segments):
            a=ring*segments+i;b=ring*segments+(i+1)%segments
            faces.append((center,b,a) if reverse else (center,a,b))
    save(name,vertices,faces)


def torus(name,radius,tube,start=0,end=2*math.pi):
    vertices=[];faces=[];n,m=128,10
    for i in range(n+1):
        t=start+(end-start)*i/n
        for j in range(m):
            p=2*math.pi*j/m;r=radius+tube*math.cos(p)
            vertices.append((r*math.cos(t),r*math.sin(t),tube*math.sin(p)))
    for i in range(n):
        for j in range(m):
            a=i*m+j;b=i*m+(j+1)%m;c=b+m;d=a+m
            faces.extend([(a,c,b),(a,d,c)])
    save(name,vertices,faces)


def propeller():
    vertices=[];faces=[]
    for sign in [1,-1]:
        offset=len(vertices)
        for side in [-1,1]:
            for i in range(33):
                t=i/32;x=.003+.033*t;width=.0015+.0032*math.sin(math.pi*t)**.65
                sweep=.003*math.sin(math.pi*t)
                for edge in [-1,1]:
                    y=edge*width+sweep;z=side*.00045 + y*(.26-.18*t)
                    vertices.append((sign*x,sign*y,z))
        for side in range(2):
            for i in range(32):
                a=offset+side*66+i*2;b=a+2
                faces.extend([(a,b,b+1),(a,b+1,a+1)] if side else [(a,b+1,b),(a,a+1,b+1)])
        for i in range(32):
            for edge in [0,1]:
                a=offset+i*2+edge;b=a+2;c=b+66;d=a+66
                faces.extend([(a,b,c),(a,c,d)])
        for i in [0,32]:
            a=offset+i*2;faces.extend([(a,a+66,a+67),(a,a+67,a+1)])
    save('propeller',vertices,faces)


shell('shell', [(-.001,.029,.016,4),(.003,.033,.019,4),(.012,.032,.018,4),
               (.018,.029,.016,3.5),(.020,.023,.012,3),(.0208,.013,.006,2)])
shell('chassis',[(-.018,.021,.012,3),(-.015,.030,.017,4),(-.007,.033,.020,4),(.002,.033,.020,4)])
shell('battery',[(-.001,.024,.010,4),(.000,.025,.011,4),(.001,.024,.010,4)])
shell('camera_bezel',[(-.005,.006,.010,4),(.004,.006,.010,4)])
torus('guard',.0385,.0012)
torus('lens_rim',.0042,.00065)
propeller()

# Convert a font outline to a small native mesh for a crisp top-shell label.
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.tri import Triangulation
word=TextPath((0,0),'TELLO',size=1,prop=FontProperties(family='DejaVu Sans',weight='bold'))
polygons=word.to_polygons()
vertices=np.vstack([poly[:-1] if np.allclose(poly[0],poly[-1]) else poly for poly in polygons])
vertices=np.unique(vertices,axis=0)
tri=Triangulation(vertices[:,0],vertices[:,1])
centers=vertices[tri.triangles].mean(axis=1)
inside=np.zeros(len(centers),dtype=bool)
for polygon in polygons:inside ^= MPath(polygon).contains_points(centers)
faces=tri.triangles[inside].tolist()
lo,hi=vertices.min(axis=0),vertices.max(axis=0)
vertices=(vertices-(lo+hi)/2)*(.026/(hi[0]-lo[0]))
save('shell_label',np.c_[vertices,np.zeros(len(vertices))],faces)

WHITE='0.90 0.92 0.94 1';DARK='0.065 0.08 0.10 1';ARM='0.13 0.15 0.18 1';METAL='0.48 0.52 0.56 1';BLACK='0.025 0.033 0.045 1'

def visual(link,name,shape,pose='0 0 0 0 0 0',color=DARK,metal=.0,rough=.45):
    v=ET.SubElement(link,'visual',name=name);ET.SubElement(v,'pose').text=pose
    geom=ET.SubElement(v,'geometry')
    if shape[0]=='mesh':ET.SubElement(ET.SubElement(geom,'mesh'),'uri').text=f'model://tello/meshes/{shape[1]}.obj'
    elif shape[0]=='box':ET.SubElement(ET.SubElement(geom,'box'),'size').text=shape[1]
    elif shape[0]=='cylinder':
        c=ET.SubElement(geom,'cylinder');ET.SubElement(c,'radius').text=str(shape[1]);ET.SubElement(c,'length').text=str(shape[2])
    material=ET.SubElement(v,'material')
    for tag,text in [('ambient',color),('diffuse',color),('specular','0.3 0.3 0.3 1')]:ET.SubElement(material,tag).text=text
    pbr=ET.SubElement(ET.SubElement(material,'pbr'),'metal');ET.SubElement(pbr,'metalness').text=str(metal);ET.SubElement(pbr,'roughness').text=str(rough)
    return v


def build():
    path=MODEL/'model.sdf';tree=ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)));model=tree.getroot().find('model')
    for link in model.findall('link'):
        for child in list(link.findall('visual')):link.remove(child)
    base=model.find("link[@name='base_link']")
    visual(base,'body',('mesh','shell'),color=WHITE,rough=.24)
    visual(base,'chassis',('mesh','chassis'),color=DARK,rough=.6)
    visual(base,'battery_seam',('mesh','battery'),'-.006 0 .0203 0 0 0',color='0.30 0.33 0.36 1')
    visual(base,'battery_top',('mesh','battery'),'-.006 0 .0208 0 0 0',color=WHITE,rough=.27)
    visual(base,'shell_label',('mesh','shell_label'),'-.006 0 .022 0 0 0',color=ARM,rough=.45)
    visual(base,'fleet_accent',('box','0.014 0.026 0.0006'),'0.015 0 .0202 0 0 0',color='0.9 0.9 0.9 1',rough=.3)
    visual(base,'front_camera_housing',('mesh','camera_bezel'),'.033 0 -.004 0 1.5707963 0',color=BLACK)
    visual(base,'camera_lens_rim',('mesh','lens_rim'),'.0375 0 -.004 0 1.5707963 0',color=METAL,metal=.85,rough=.2)
    visual(base,'camera_glass',('cylinder',.0038,.0008),'.038 0 -.004 0 1.5707963 0',color='0.025 0.12 0.20 1',metal=.8,rough=.08)
    visual(base,'lens_inner',('cylinder',.0018,.0009),'.0385 0 -.004 0 1.5707963 0',color='0.01 0.025 0.04 1',rough=.05)
    visual(base,'status_led',('cylinder',.001,.001),'.034 .012 -.004 0 1.5707963 0',color='0.05 0.9 0.45 1',rough=.2)
    for i in range(7):
        visual(base,f'vent_{i}',('box','0.0008 0.001 0.005'),f'.0315 {-.009+i*.003} .009 0 0 0',color=BLACK)
    for x,y in [(.04,-.04),(-.04,.04),(.04,.04),(-.04,-.04)]:
        name=f'{x}_{y}';yaw=math.atan2(y,x)
        visual(base,'arm_'+name,('box','0.045 0.008 0.007'),f'{x*.64} {y*.64} -.002 0 0 {yaw}',color=ARM)
        visual(base,'motor_'+name,('cylinder',.0062,.015),f'{x} {y} .006 0 0 0',color=DARK,metal=.25)
        visual(base,'motor_collar_'+name,('cylinder',.0064,.003),f'{x} {y} .012 0 0 0',color=METAL,metal=.75,rough=.26)
        visual(base,'guard_'+name,('mesh','guard'),f'{x} {y} .013 0 0 0',color=DARK)
        for a in [yaw-.95,yaw,yaw+.95]:
            cx=x+.020*math.cos(a);cy=y+.020*math.sin(a)
            visual(base,f'guard_brace_{name}_{a}',('box','0.037 0.0015 0.002'),f'{cx} {cy} .012 0 0 {a}',color=DARK)
        visual(base,'landing_strut_'+name,('cylinder',.002,.018),f'{x} {y} -.007 0 0 0',color=DARK)
        visual(base,'landing_foot_'+name,('cylinder',.004,.005),f'{x} {y} -.018 0 0 0',color=BLACK)
        visual(base,'screw_'+name,('cylinder',.0011,.001),f'{x*.47} {y*.40} .018 0 0 0',color=METAL,metal=.8)
    for i in range(4):
        rotor=model.find(f"link[@name='rotor_{i}']")
        visual(rotor,'sculpted_propeller',('mesh','propeller'),'0 0 .0018 0 0 0',color=BLACK,rough=.32)
        visual(rotor,'hub',('cylinder',.0038,.0035),'0 0 .002 0 0 0',color=ARM,metal=.25)
        visual(rotor,'hub_cap',('cylinder',.0014,.0038),'0 0 .002 0 0 0',color=METAL,metal=.8)
    ET.indent(tree,space='  ')
    tree.write(path,encoding='utf-8',xml_declaration=True)

if __name__=='__main__':build()
