/* Pricing examples from optimization_pricing_example.md. All charts use the
   actual functions, and all iterations use their analytic derivatives. */
(() => {
  'use strict';
  const color = {ink:'#172e40', muted:'#5b6e79', grid:'#d5e0e3', blue:'#376bb3', teal:'#087e83', purple:'#6656aa', orange:'#b35c36'};
  const format = (n, digits=0) => n.toLocaleString('fr-FR', {minimumFractionDigits:digits, maximumFractionDigits:digits});
  const demand = p => 100 - 5*p;
  const profit = p => (p-4)*demand(p);
  const criterion = p => -profit(p);
  const groups = [[60,8,10],[40,15,15],[20,22,8]];
  const marketDemand = p => groups.reduce((sum,[a,m,v]) => sum + a*Math.exp(-((p-m)**2)/v),0);
  const marketCost = p => -(p-4)*marketDemand(p);
  const marketGradient = p => -marketDemand(p) - (p-4)*groups.reduce((sum,[a,m,v]) => sum + a*Math.exp(-((p-m)**2)/v)*(-2*(p-m)/v),0);
  const trajectories = [7,14,20].map(start => {
    const prices = [start];
    for (let k=0; k<120; k++) prices.push(prices[k] - .008*marketGradient(prices[k]));
    return prices;
  });
  const minima = trajectories.map(prices => prices[prices.length-1]);
  const text = (x,y,label,fill=color.muted,anchor='start',size=22) => `<text x="${x}" y="${y}" fill="${fill}" text-anchor="${anchor}" font-size="${size}">${label}</text>`;
  const line = (x1,y1,x2,y2,stroke=color.grid,dash='') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}" stroke-width="1.5" ${dash ? `stroke-dasharray="${dash}"`:''}/>`;
  const dot = (x,y,fill,r=7) => `<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="#f6f8f8" stroke-width="2.5"/>`;
  function chart(id,{xmin,xmax,ymin,ymax,xticks,yticks,xlabel='Prix p (€)',height=420,right=28}) {
    const el = document.getElementById(id);
    const box = {left:75,top:30,right:720-right,bottom:height-70};
    const x = v => box.left+(v-xmin)/(xmax-xmin)*(box.right-box.left);
    const y = v => box.bottom-(v-ymin)/(ymax-ymin)*(box.bottom-box.top);
    let content = '';
    for (const v of yticks) content += line(box.left,y(v),box.right,y(v)) + text(box.left-13,y(v)+7,format(v),color.muted,'end',21);
    content += line(box.left,box.top,box.left,box.bottom)+line(box.left,box.bottom,box.right,box.bottom);
    for (const v of xticks) content += line(x(v),box.bottom,x(v),box.bottom+7)+text(x(v),box.bottom+32,format(v),color.muted,'middle',21);
    content += text(box.right,height-7,xlabel,color.muted,'end',22);
    return {
      x,y,box,
      add: markup => {content += markup;},
      curve(fn,stroke,dash='') {
        // Sampling includes the L1 breakpoint p=10 exactly for the comparison.
        const values = Array.from({length:481},(_,i)=>xmin+(xmax-xmin)*i/480);
        content += `<polyline points="${values.map(v=>`${x(v).toFixed(2)},${y(fn(v)).toFixed(2)}`).join(' ')}" fill="none" stroke="${stroke}" stroke-width="4" stroke-linejoin="round" ${dash?`stroke-dasharray="${dash}"`:''}/>`;
      },
      render: () => {el.innerHTML = content;}
    };
  }
  const d = chart('demand-plot',{xmin:4,xmax:20,ymin:0,ymax:85,xticks:[4,8,12,16,20],yticks:[0,20,40,60,80]});
  d.curve(demand,color.blue);
  for (const [p,label,dx,dy] of [[10,'10 € : 50 unités',18,-17],[15,'15 € : 25 unités',18,-17]]) {
    d.add(line(d.x(p),d.y(0),d.x(p),d.y(demand(p)),color.teal,'5 5'));
    d.add(dot(d.x(p),d.y(demand(p)),color.teal));
    d.add(text(d.x(p)+dx,d.y(demand(p))+dy,label,color.teal,'start',24));
  }
  d.render();
  function renderProfit() {
    const p = Number(document.getElementById('price').value);
    const c = chart('profit-plot',{xmin:4,xmax:20,ymin:0,ymax:360,xticks:[4,8,12,16,20],yticks:[0,100,200,300]});
    c.curve(profit,color.blue);
    c.add(line(c.x(12),c.y(0),c.x(12),c.y(320),color.teal,'5 5'));
    c.add(dot(c.x(12),c.y(320),color.teal,8));
    c.add(text(c.x(12),c.y(320)-20,'Maximum : 320 €',color.teal,'middle',24));
    c.add(line(c.x(p),c.y(0),c.x(p),c.y(profit(p)),color.orange,'4 5'));
    c.add(dot(c.x(p),c.y(profit(p)),color.orange,8));
    c.render();
    document.getElementById('price-value').textContent = `${format(p,1)} €`;
    document.getElementById('price').setAttribute('aria-valuetext',`${format(p,1)} euros`);
    document.getElementById('profit-value').textContent = `${format(demand(p),1)} unités / profit : ${format(profit(p),2)} €`;
  }
  const compare = chart('criteria-plot',{xmin:8,xmax:14,ymin:-330,ymax:-80,xticks:[8,9,10,11,12,13,14],yticks:[-300,-200,-100]});
  compare.add(line(compare.x(10),compare.box.top,compare.x(10),compare.box.bottom,color.muted,'4 5'));
  compare.add(text(compare.x(10)+12,compare.box.top+18,'Référence : 10 €',color.muted,'start',21));
  for (const [fn,best,stroke,dash] of [[criterion,12,color.blue,''],[p=>criterion(p)+10*Math.abs(p-10),11,color.teal,'10 5'],[p=>criterion(p)+10*(p-10)**2,32/3,color.purple,'3 5']]) {
    compare.curve(fn,stroke,dash);
    compare.add(dot(compare.x(best),compare.y(fn(best)),stroke,8));
  }
  compare.render();
  function drawMarket(id,height=420,annotate=true) {
    const c = chart(id,{xmin:4,xmax:28,ymin:annotate?-600:-500,ymax:0,xticks:[4,8,12,16,20,24,28],yticks:annotate?[-600,-400,-200,0]:[-500,-250,0],height});
    c.curve(marketCost,color.ink);
    minima.forEach((p,i)=>{
      const stroke = i===1 ? color.teal : color.orange;
      c.add(dot(c.x(p),c.y(marketCost(p)),stroke,8));
      if(annotate) {
        c.add(text(c.x(p),c.y(marketCost(p))+29,i===1?'Global':'Local',stroke,'middle',24));
        c.add(text(c.x(p),c.y(marketCost(p))+55,`${format(p,2)} €`,stroke,'middle',21));
      }
    });
    c.render();
  }
  drawMarket('market-plot');
  drawMarket('nonconvex-plot',360,false);
  const cv = chart('convex-plot',{xmin:4,xmax:20,ymin:-350,ymax:0,xticks:[4,8,12,16,20],yticks:[-300,-150,0],height:360});
  cv.curve(criterion,color.blue);
  cv.add(dot(cv.x(12),cv.y(-320),color.teal,8));
  cv.render();
  const it = chart('convergence-plot',{xmin:0,xmax:20,ymin:5,ymax:25,xticks:[0,5,10,15,20],yticks:[5,10,15,20,25],xlabel:'Itération',right:150});
  trajectories.forEach((prices,i)=>{
    const stroke = [color.orange,color.teal,color.purple][i];
    const values = prices.slice(0,21);
    it.add(`<polyline points="${values.map((p,k)=>`${it.x(k)},${it.y(p)}`).join(' ')}" fill="none" stroke="${stroke}" stroke-width="3.5"/>`);
    values.forEach((p,k)=>it.add(dot(it.x(k),it.y(p),stroke,4.5)));
    it.add(text(it.x(0)+9,it.y(prices[0])+(i===0?25:30),`${prices[0]} €`,stroke,'start',23));
    it.add(text(it.x(20)+16,it.y(minima[i])-5,`${format(minima[i],2)} €`,stroke,'start',25));
    it.add(text(it.x(20)+16,it.y(minima[i])+22,i===1?'global':'local',stroke,'start',21));
  });
  it.add(text(it.box.left,19,'Prix p (€)',color.muted,'start',22));
  it.render();
  const slider = document.getElementById('price');
  slider.addEventListener('input',renderProfit);
  slider.addEventListener('keydown',event=>event.stopPropagation());
  renderProfit();
})();
