/* Local, dependency-free 3D data charts. The SVG mesh is an orthographic
   projection of sampled objective values; paths are computed iterations. */
(() => {
  'use strict';
  const C = {ink:'#172e40', teal:'#087e83', blue:'#376bb3', purple:'#6656aa', orange:'#b35c36', grid:'#c7d8dc', muted:'#5b6e79'};
  const clamp = (v,a,b) => Math.max(a,Math.min(b,v));
  const text = (x,y,t,c=C.muted,size=22,anchor='start') => `<text x="${x}" y="${y}" fill="${c}" font-size="${size}" text-anchor="${anchor}">${t}</text>`;
  const line = (a,b,c=C.grid,w=1.5,dash='') => `<path d="M${a.join(',')}L${b.join(',')}" fill="none" stroke="${c}" stroke-width="${w}" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
  const dot = (p,c,r=7) => `<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="${c}" stroke="white" stroke-width="2.5"/>`;
  const sigmoid = t => 1/(1+Math.exp(-t));
  const convex = (u,v) => {const x=2*u-1,y=2*v-1;return (x*x+1.6*y*y+.5*x*y)/2;};
  function convexTrace() {
    let x=-.95,y=-.95; const trace=[];
    for(let k=0;k<=32;k++) {
      trace.push([(x+1)/2,(y+1)/2,convex((x+1)/2,(y+1)/2)]);
      const gx=x+.25*y, gy=.25*x+1.6*y;
      x=clamp(x-.18*gx,-1,1);y=clamp(y-.18*gy,-1,1);
    }
    return trace;
  }
  // Pedagogical adjustment: deepen basin A in the objective itself, not just
  // in the drawing. Gradients, minima and displayed profits use this same cost.
  const basinA=[(8.42393129213717-4)/16,32.729208383659/180];
  const basinBonus=(u,v)=>180*Math.exp(-((u-basinA[0])**2/.025+(v-basinA[1])**2/.018));
  function campaign(u,v) {
    const p=4+16*u,a=180*v;
    const q=sigmoid(-.3*p+.4*Math.log1p(a)+2.5*sigmoid(.12*(a-90)));
    return a-(p-4)*100*q-basinBonus(u,v);
  }
  function campaignGradient(u,v) {
    const p=4+16*u,a=180*v,s=sigmoid(.12*(a-90));
    const q=sigmoid(-.3*p+.4*Math.log1p(a)+2.5*s),dq=q*(1-q);
    const bonus=basinBonus(u,v)/100;
    return [16*(-q+(p-4)*.3*dq)+2*bonus*(u-basinA[0])/.025,
      180*(.01-(p-4)*dq*(.4/(1+a)+.3*s*(1-s)))+2*bonus*(v-basinA[1])/.018];
  }
  function campaignTrace(p,a) {
    let u=(p-4)/16,v=a/180;const trace=[];
    for(let k=0;k<=160;k++) {
      const cost=campaign(u,v);trace.push([u,v,cost]);
      const [gu,gv]=campaignGradient(u,v);let step=.004,nu=u,nv=v;
      for(let j=0;j<30;j++) {
        nu=clamp(u-step*gu,0,1);nv=clamp(v-step*gv,0,1);
        if(campaign(nu,nv)<=cost+1e-10) break;
        step/=2;
      }
      u=nu;v=nv;
    }
    return trace;
  }
  const bowlTrace=convexTrace();
  const starts=[[5,5],[18,20],[8,55],[5,145],[18,165],[12,110]];
  const paths=starts.map(([p,a])=>campaignTrace(p,a));
  const trajectoryColors=paths.map(t=>t.at(-1)[1]<.5?C.orange:C.purple);
  const minima=[campaignTrace(8.42393129213717,32.729208383659).at(-1),campaignTrace(13.026926952937455,132.69997351125932).at(-1)];
  function buildSurface(el) {
    const kind=el.dataset.surface, isBowl=kind==='convex';
    const fn=isBowl?convex:campaign;
    const z = cost => isBowl?cost/1.55:(cost+450)/630;
    // Separate camera azimuths: a diagonal corner faces the audience for the
    // bowl; the low-budget side faces the audience to expose basin A.
    const azimuth=(isBowl?-40:-115)*Math.PI/180;
    const cosine=Math.cos(azimuth),sine=Math.sin(azimuth);
    const floor=(u,v)=>[390+((u-.5)*cosine-(v-.5)*sine)*400,
      300+((u-.5)*sine+(v-.5)*cosine)*135];
    const project=(u,v,cost)=>{const [x,y]=floor(u,v);return [x,y-z(cost)*195];};
    const frontU=sine>0?1:0,frontV=cosine>0?1:0;
    let svg='';
    // Ground plane establishes the two decisions independently of height.
    for(let i=0;i<=4;i++) {const q=i/4;svg+=line(floor(q,0),floor(q,1),'#e0e8ea')+line(floor(0,q),floor(1,q),'#e0e8ea');}
    svg+=line(floor(0,frontV),floor(1,frontV),C.muted,2)+line(floor(frontU,0),floor(frontU,1),C.muted,2);
    // Sparse wireframe: smooth sampled curves, with no opaque surface hiding
    // the far side of either basin. Each line lies on the actual objective.
    for(const axis of [0,1])for(let grid=0;grid<=12;grid++) {
      const fixed=grid/12;
      const points=Array.from({length:101},(_,k)=>{
        const u=axis===0?k/100:fixed,v=axis===0?fixed:k/100;
        return project(u,v,fn(u,v)).join(',');
      }).join(' ');
      const boundary=grid===0||grid===12;
      svg+=`<polyline points="${points}" fill="none" stroke="${axis===0?'#087e83':'#65979f'}" stroke-width="${boundary?2.2:1.6}" stroke-opacity="${boundary?'.9':'.72'}"/>`;
    }
    svg+=line([67,236],[67,51],C.muted,2)+line([67,51],[61,63],C.muted,2)+line([67,51],[73,63],C.muted,2);
    svg+=text(38,30,isBowl?'Coût relatif':'Coût',C.ink,23)+text(50,79,'Plus haut',C.muted,17,'end')+text(50,233,'Plus bas',C.teal,17,'end');
    const priceMid=floor(.5,frontV),budgetMid=floor(frontU,.5);
    const priceSide=priceMid[0]<budgetMid[0]?-1:1;
    svg+=text(priceMid[0]+priceSide*22,priceMid[1]+46,'Prix (€)',C.ink,23,'middle');
    svg+=text(budgetMid[0]-priceSide*25,budgetMid[1]+46,'Budget publicitaire (€)',C.ink,23,'middle');
    for(const [u,label] of [[0,'4'],[1,'20']]) {
      const point=floor(u,frontV),shared=u===frontU;
      svg+=text(point[0]+(shared?priceSide*17:priceSide*15),point[1]+(shared?29:17),label,C.muted,19,priceSide<0?'end':'start');
    }
    for(const [v,label] of [[0,'0'],[1,'180']]) {
      const point=floor(frontU,v),shared=v===frontV;
      svg+=text(point[0]-(shared?priceSide*17:priceSide*15),point[1]+(shared?29:17),label,C.muted,19,priceSide<0?'start':'end');
    }
    const mm=isBowl?[[.5,.5,0]]:minima;
    mm.forEach((m,i)=>{
      const p=project(...m),color=isBowl?C.teal:i===0?C.orange:C.purple;
      svg+=dot(p,color,8);
      const label=isBowl?[p[0]+70,p[1]+46]:i===0?[p[0]-52,p[1]+38]:[p[0]+58,p[1]+49];
      svg+=line([p[0]+5,p[1]+(isBowl?8:0)],[label[0],label[1]-12],color,1.5);
      svg+=text(label[0],label[1],isBowl?'Minimum':i===0?'A':'B',color,26,'middle').replace('<text ', '<text style="paint-order:stroke;stroke:#f6f8f8;stroke-width:5px;stroke-linejoin:round" ');
    });
    el.innerHTML=svg+'<g class="surface-trails"></g>';
    return {el,project,layer:el.querySelector('.surface-trails')};
  }
  const surfaces={};
  document.querySelectorAll('[data-surface]').forEach(el=>{surfaces[el.id]=buildSurface(el);});
  function trail(surface,traces,colors,step) {
    let svg='';
    traces.forEach((trace,i)=>{
      const end=Math.min(step,trace.length-1),screen=trace.slice(0,end+1).map(p=>surface.project(...p));
      const points=screen.map(p=>p.join(',')).join(' '),c=colors[i];
      svg+=`<polyline points="${points}" fill="none" stroke="white" stroke-width="6" stroke-linejoin="round"/><polyline points="${points}" fill="none" stroke="${c}" stroke-width="3" stroke-linejoin="round"/>`;
      svg+=dot(screen[0],c,4);
      if(traces.length===1)screen.slice(1).forEach(p=>{svg+=dot(p,c,3);});
      svg+=dot(screen.at(-1),c,8);
    });
    surface.layer.innerHTML=svg;
  }
  function history(step) {
    const el=document.getElementById('convergence-history'),x=k=>30+290*k/32,y=c=>112-c/(bowlTrace[0][2]*1.08)*87;
    let svg=line([30,22],[30,112])+line([30,112],[330,112]);
    svg+=text(32,17,'Coût',C.muted,20)+text(330,144,'Étapes',C.muted,20,'end');
    svg+=`<polyline points="${bowlTrace.slice(0,step+1).map((p,k)=>[x(k),y(p[2])].join(',')).join(' ')}" fill="none" stroke="${C.teal}" stroke-width="3"/>`;
    svg+=dot([x(step),y(bowlTrace[step][2])],C.orange,5);
    el.innerHTML=svg;
  }
  const renderers={
    'convex-descent':t=>trail(surfaces['convex-descent'],[bowlTrace],[C.orange],t),
    'convex-convergence':t=>{trail(surfaces['convex-convergence'],[bowlTrace],[C.orange],t);history(t);},
    'campaign-trajectories':t=>trail(surfaces['campaign-trajectories'],paths,trajectoryColors,t*2)
  };
  const controllers=[];
  document.querySelectorAll('[data-animation]').forEach(control=>{
    const id=control.dataset.animation,render=renderers[id],input=control.querySelector('input'),play=control.querySelector('.play-animation'),reset=control.querySelector('.reset-animation'),output=control.querySelector('output');
    let timer=null;
    const pause=()=>{clearInterval(timer);timer=null;play.textContent='Lire';play.setAttribute('aria-pressed','false');};
    const update=()=>{const t=Number(input.value);render(t);output.textContent=`${id==='campaign-trajectories'?t*2:t} / ${id==='campaign-trajectories'?Number(input.max)*2:input.max}`;input.setAttribute('aria-valuetext',`Étape ${id==='campaign-trajectories'?t*2:t}`);};
    play.addEventListener('click',()=>{
      if(timer){pause();return;}
      if(Number(input.value)>=Number(input.max)) input.value=0;
      play.textContent='Pause';play.setAttribute('aria-pressed','true');update();
      timer=setInterval(()=>{input.value=Number(input.value)+1;update();if(Number(input.value)>=Number(input.max))pause();},id==='campaign-trajectories'?120:220);
    });
    reset.addEventListener('click',()=>{pause();input.value=0;update();});
    input.addEventListener('input',()=>{pause();update();});
    control.querySelectorAll('button,input').forEach(e=>e.addEventListener('keydown',event=>event.stopPropagation()));
    controllers.push({pause,input,update,section:control.closest('section')});update();
  });
  // Presenter-controlled playback avoids surprise movement and also supports
  // reduced-motion users. Navigation and background tabs always pause playback.
  Reveal.on('slidechanged',()=>controllers.forEach(c=>c.pause()));
  document.addEventListener('visibilitychange',()=>{if(document.hidden)controllers.forEach(c=>c.pause());});
  const finishForPrint=()=>controllers.forEach(c=>{c.pause();c.input.value=c.input.max;c.update();});
  window.addEventListener('beforeprint',finishForPrint);
  if(new URLSearchParams(location.search).has('print-pdf'))finishForPrint();
})();
