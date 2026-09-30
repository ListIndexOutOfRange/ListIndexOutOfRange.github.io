"""Rebuild the pricing landscapes from the model in optimization_slides_plan.md.
No dependencies. Run: python3 scripts/cnphge_visuals.py
"""
import math, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'static/cnphge'
OUT=ROOT/'assets'; OUT.mkdir(exist_ok=True)
def sigmoid(z): return 1/(1+math.exp(-z))
def model(p,a,campaign=False):
    h=sigmoid(.12*(a-90))
    s=sigmoid(-.3*p+.4*math.log1p(a)+(2.5*h if campaign else 0))
    da=.4/(1+a)+(2.5*.12*h*(1-h) if campaign else 0)
    j=-(p-4)*100*s+a
    return j,(-100*s+(p-4)*30*s*(1-s),1-(p-4)*100*s*(1-s)*da)
def best_price(a,campaign=False):
    lo,hi=4.,20.
    for _ in range(65):
        p=(lo+hi)/2
        if model(p,a,campaign)[1][0]<0:lo=p
        else:hi=p
    return (lo+hi)/2
# For each advertising budget, profit has a unique interior maximum in p:
# beta*(p-c)*(1-sigma(z)) is strictly increasing in p. Reduce global search
# to the profiled one-dimensional cost in a, including boundary values.
def profile(a,c): return model(best_price(a,c),a,c)[0]
def golden(lo,hi,c):
    for _ in range(100):
        x=lo+(hi-lo)*.381966;y=lo+(hi-lo)*.618034
        if profile(x,c)<profile(y,c):hi=y
        else:lo=x
    a=(lo+hi)/2;return [best_price(a,c),a,profile(a,c)]
def extrema(c):
    vals=[profile(i/10,c) for i in range(1801)]
    return [golden((i-1)/10,(i+1)/10,c) for i in range(1,1800) if vals[i]<vals[i-1] and vals[i]<vals[i+1]]
def descent(start,c=False,eta=.08,n=220):
    p,a=start; result=[]
    for k in range(n+1):
        j,g=model(p,a,c); result.append([p,a,j])
        # Gradient in u=(p-4)/16, v=a/180, for F=J/100.
        gu,gv=g[0]*16/100,g[1]*180/100
        step=eta
        for _ in range(35):
            pn=max(4,min(20,p-step*gu*16));an=max(0,min(180,a-step*gv*180))
            if model(pn,an,c)[0]<=j+1e-10:break
            step*=.5
        p,a=pn,an
    return result
base=extrema(False); multi=extrema(True)
assert len(base)==1 and len(multi)==2
trace=descent((17,145))
low=descent((6,12),True,n=500);high=descent((18,165),True,n=500)
assert abs(trace[-1][2]-base[0][2])<.001
assert abs(low[-1][2]-multi[0][2])<.001 and abs(high[-1][2]-multi[1][2])<.001
# Check analytic derivatives against central finite differences.
for c in [False,True]:
    for p,a in [(8,20),(12,85),(16,140)]:
        _,g=model(p,a,c)
        for axis in [0,1]:
            h=1e-4; v=[p,a];v[axis]+=h;plus=model(*v,c)[0];v[axis]-=2*h;minus=model(*v,c)[0]
            assert abs((plus-minus)/(2*h)-g[axis])<1e-5
for t in [trace,low,high]:assert all(b[2]<=a[2]+1e-8 for a,b in zip(t,t[1:]))
C={'ink':'#172e40','teal':'#087e83','orange':'#b35c36','purple':'#6656aa','muted':'#5b6e79'}
def txt(x,y,t,col='#5b6e79',size=20,anchor='start'):
    return f'<text x="{x:.2f}" y="{y:.2f}" fill="{col}" font-size="{size}" text-anchor="{anchor}" font-family="Arial,sans-serif">{t}</text>'
def line(x,y,X,Y,col='#d5e0e3',width=1.5):return f'<path d="M{x:.2f},{y:.2f}L{X:.2f},{Y:.2f}" fill="none" stroke="{col}" stroke-width="{width}"/>'
def dot(x,y,col,r=6):return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{col}" stroke="white" stroke-width="2"/>'
def xy(p,a):return 75+(p-4)/16*555,365-a/180*310
def path(t,col):
    return '<path d="'+' '.join(('M' if i==0 else 'L')+f'{xy(p,a)[0]:.2f},{xy(p,a)[1]:.2f}' for i,(p,a,*_) in enumerate(t))+'" fill="none" stroke="'+col+'" stroke-width="3.5"/>'
def wrap(s):return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 440">'+s+'</svg>'
def terrain(c=False):
    levels=[-70,-60,-40,-20,0,40,80,120,160] if not c else [-430,-400,-320,-240,-160,-100,-75,-60,-20,40,120]
    # Marching squares: contour segments are interpolated from the actual J.
    n=85; grid=[[model(4+16*i/n,180*j/n,c)[0] for i in range(n+1)] for j in range(n+1)]
    s=txt(75,25,'Budget publicitaire a (€)',size=21)
    for lev in levels:
        d=[]
        for j in range(n):
            for i in range(n):
                corners=[(i,j),(i+1,j),(i+1,j+1),(i,j+1)];cross=[]
                for k in range(4):
                    u,v=corners[k],corners[(k+1)%4];z=grid[u[1]][u[0]];Z=grid[v[1]][v[0]]
                    if (z<=lev<Z) or (Z<=lev<z):
                        f=(lev-z)/(Z-z);cross.append(xy(4+16*(u[0]+f*(v[0]-u[0]))/n,180*(u[1]+f*(v[1]-u[1]))/n))
                for k in range(0,len(cross)-1,2):
                    x,y=cross[k];X,Y=cross[k+1];d.append(f'M{x:.1f},{y:.1f}L{X:.1f},{Y:.1f}')
        col='#087e83' if lev<0 else '#bccfd4'
        s+=f'<path d="{" ".join(d)}" fill="none" stroke="{col}" stroke-width="1.5" opacity=".7"/>'
    s+=line(75,55,75,365)+line(75,365,630,365)
    for p in [4,8,12,16,20]:s+=txt(xy(p,0)[0],392,str(p),anchor='middle')
    for a in [0,60,120,180]:s+=txt(62,xy(4,a)[1]+6,str(a),anchor='end')
    s+=txt(630,429,'Prix p (€)',anchor='end',size=21)
    return s
base_map=terrain();multi_map=terrain(True)
def save(name,s): (OUT/(name+'.svg')).write_text(wrap(s))
def mark_min(s,m,label,col):
    x,y=xy(*m[:2]);return s+dot(x,y,col,7)+txt(x+12,y-12,label,col,21)
save('landscape-base',mark_min(base_map,base[0],'Minimum',C['teal']))
save('landscape-search',base_map+dot(*xy(17,145),C['orange'],8)+txt(*xy(17,145),' ?',C['orange'],32))
# Gradient arrows are drawn in normalized coordinates, with the coordinate
# aspect ratio accounted for when mapping vectors to screen space.
p,a=14,95;j,g=model(p,a);gu,gv=g[0]*16/100,g[1]*180/100;norm=math.hypot(gu,gv);u,v=gu/norm,gv/norm
x,y=xy(p,a)
s=base_map+'<defs><marker id="up" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0L7,3L0,6Z" fill="#b35c36"/></marker><marker id="down" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0L7,3L0,6Z" fill="#087e83"/></marker></defs>'
for sign,col,id,label in [(1,C['orange'],'up','∇J : monter'),(-1,C['teal'],'down','−∇J : descendre')]:
    X,Y=x+sign*u*100,y-sign*v*310/555*100
    s+=f'<path d="M{x},{y}L{X},{Y}" stroke="{col}" stroke-width="3" marker-end="url(#{id})"/>'+txt(X+10,Y-10 if sign==1 else Y+25,label,col,20)
save('landscape-gradient',s+dot(x,y,C['ink'],7))
save('landscape-trajectory',mark_min(base_map+path(trace,C['orange']),base[0],'Minimum',C['teal'])+dot(*xy(*trace[0][:2]),C['orange'],7))
save('landscape-empty',base_map)
save('landscape-campaign',mark_min(mark_min(multi_map,multi[0],'A',C['orange']),multi[1],'B',C['teal']))
save('landscape-local',mark_min(mark_min(multi_map+path(low,C['orange'])+path(high,C['teal']),multi[0],'A · local',C['orange']),multi[1],'B · global',C['teal']))
s=multi_map
for start in [(5,5),(18,20),(8,55),(5,145),(18,165),(12,110)]:
    t=descent(start,True,n=400);col=C['orange'] if t[-1][1]<90 else C['teal'];s+=path(t,col)+dot(*xy(*start),col,5)
save('landscape-starts',s)
# Cost history for the exact same trajectory, not an illustrative sketch.
s=txt(75,25,'Critère J (€)',size=21)
xx=lambda k:75+k/len(trace)*555; yy=lambda j:365-(j+80)/240*310
for z in [-80,0,80,160]:s+=line(75,yy(z),630,yy(z))+txt(62,yy(z)+6,str(z),anchor='end')
for k in [0,50,100,150,200]:s+=txt(xx(k),394,str(k),anchor='middle')
s+='<path d="'+' '.join(('M' if i==0 else 'L')+f'{xx(i):.2f},{yy(t[2]):.2f}' for i,t in enumerate(trace))+'" fill="none" stroke="#087e83" stroke-width="3"/>'+txt(630,429,'Itération',anchor='end')
save('cost-history',s)
# Controlled step-size comparison on the original one-dimensional quadratic.
s=txt(75,25,'Prix p (€)',size=21)
x=lambda k:75+k/20*470;y=lambda p:365-(p-2)/20*310
for p in [4,8,12,16,20]:s+=line(75,y(p),545,y(p))+txt(62,y(p)+6,str(p),anchor='end')
for k in [0,5,10,15,20]:s+=txt(x(k),394,str(k),anchor='middle')
for eta,col,label in [(.005,C['purple'],'Petit'),(.08,C['teal'],'Adapté'),(.2,C['orange'],'Trop grand')]:
    ps=[5.]
    for _ in range(20):ps.append(ps[-1]-eta*10*(ps[-1]-12))
    s+='<polyline points="'+' '.join(f'{x(k)},{y(p)}' for k,p in enumerate(ps))+'" fill="none" stroke="'+col+'" stroke-width="3"/>'
    s+=txt(562,y(ps[-1])+6,label,col,20)
s+=txt(545,429,'Itération',anchor='end');save('step-sizes',s)
data={'baseMinimum':base[0],'campaignMinima':multi,'trajectory':trace,'localTrajectory':low,'globalTrajectory':high}
(ROOT/'js/landscape-data.js').write_text('/* Generated by scripts/cnphge_visuals.py. */\nwindow.CNPHGE_OPTIMIZATION = '+json.dumps(data,separators=(',',':'))+';\n')
print(json.dumps({'base':base,'campaign':multi,'gradient_checks':'passed','descent_checks':'passed','final_gradient':model(*trace[-1][:2])[1]},indent=2))
